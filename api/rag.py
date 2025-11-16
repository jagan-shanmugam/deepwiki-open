import logging
import re
from dataclasses import dataclass
from typing import Any, List, Tuple, Dict, Optional
from uuid import uuid4

import dspy

from api.dspy_components import get_embedder, FAISSRetriever
from api.prompts import RAG_SYSTEM_PROMPT as system_prompt, RAG_TEMPLATE

# Configure logging
logger = logging.getLogger(__name__)

# Maximum token limit for embedding models
MAX_INPUT_TOKENS = 7500  # Safe threshold below 8192 token limit


# Conversation classes for memory management
@dataclass
class UserQuery:
    query_str: str


@dataclass
class AssistantResponse:
    response_str: str


@dataclass
class DialogTurn:
    id: str
    user_query: UserQuery
    assistant_response: AssistantResponse


class CustomConversation:
    """Custom implementation of Conversation for managing dialog history"""

    def __init__(self):
        self.dialog_turns = []

    def append_dialog_turn(self, dialog_turn):
        """Safely append a dialog turn to the conversation"""
        if not hasattr(self, 'dialog_turns'):
            self.dialog_turns = []
        self.dialog_turns.append(dialog_turn)


class Memory:
    """Simple conversation memory management."""

    def __init__(self):
        self.current_conversation = CustomConversation()

    def __call__(self) -> Dict:
        """Return the conversation history as a dictionary."""
        all_dialog_turns = {}
        try:
            if hasattr(self.current_conversation, 'dialog_turns'):
                if self.current_conversation.dialog_turns:
                    logger.info(f"Memory content: {len(self.current_conversation.dialog_turns)} turns")
                    for i, turn in enumerate(self.current_conversation.dialog_turns):
                        if hasattr(turn, 'id') and turn.id is not None:
                            all_dialog_turns[turn.id] = turn
                            logger.info(f"Added turn {i+1} with ID {turn.id} to memory")
                        else:
                            logger.warning(f"Skipping invalid turn object in memory: {turn}")
                else:
                    logger.info("Dialog turns list exists but is empty")
            else:
                logger.info("No dialog_turns attribute in current_conversation")
                self.current_conversation.dialog_turns = []
        except Exception as e:
            logger.error(f"Error accessing dialog turns: {str(e)}")
            try:
                self.current_conversation = CustomConversation()
                logger.info("Recovered by creating new conversation")
            except Exception as e2:
                logger.error(f"Failed to recover: {str(e2)}")

        logger.info(f"Returning {len(all_dialog_turns)} dialog turns from memory")
        return all_dialog_turns

    def add_dialog_turn(self, user_query: str, assistant_response: str) -> bool:
        """
        Add a dialog turn to the conversation history.

        Args:
            user_query: The user's query
            assistant_response: The assistant's response

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            dialog_turn = DialogTurn(
                id=str(uuid4()),
                user_query=UserQuery(query_str=user_query),
                assistant_response=AssistantResponse(response_str=assistant_response),
            )

            if not hasattr(self.current_conversation, 'append_dialog_turn'):
                logger.warning("current_conversation does not have append_dialog_turn method, creating new one")
                self.current_conversation = CustomConversation()

            if not hasattr(self.current_conversation, 'dialog_turns'):
                logger.warning("dialog_turns not found, initializing empty list")
                self.current_conversation.dialog_turns = []

            self.current_conversation.dialog_turns.append(dialog_turn)
            logger.info(f"Successfully added dialog turn, now have {len(self.current_conversation.dialog_turns)} turns")
            return True

        except Exception as e:
            logger.error(f"Error adding dialog turn: {str(e)}")
            try:
                self.current_conversation = CustomConversation()
                dialog_turn = DialogTurn(
                    id=str(uuid4()),
                    user_query=UserQuery(query_str=user_query),
                    assistant_response=AssistantResponse(response_str=assistant_response),
                )
                self.current_conversation.dialog_turns.append(dialog_turn)
                logger.info("Recovered from error by creating new conversation")
                return True
            except Exception as e2:
                logger.error(f"Failed to recover from error: {str(e2)}")
                return False


# DSPy Signature for RAG
class RAGSignature(dspy.Signature):
    """Answer questions based on the provided context from the repository."""

    # Input fields
    system_prompt: str = dspy.InputField(desc="System instructions for the assistant")
    context: str = dspy.InputField(desc="Relevant code and documentation from the repository")
    conversation_history: str = dspy.InputField(desc="Previous conversation turns")
    question: str = dspy.InputField(desc="User's question about the codebase")

    # Output fields
    rationale: str = dspy.OutputField(desc="Chain of thoughts for the answer")
    answer: str = dspy.OutputField(
        desc="Answer to the user query, formatted in markdown. DO NOT include ``` triple backticks fences."
    )


# Simple dataclass for backward compatibility
@dataclass
class RAGAnswer:
    rationale: str = ""
    answer: str = ""


class RAG(dspy.Module):
    """RAG with DSPy for repository-based question answering."""

    def __init__(self, provider="google", model=None, use_s3: bool = False):
        """
        Initialize the RAG component.

        Args:
            provider: Model provider to use (google, openai, openrouter, ollama, azure)
            model: Model name to use with the provider
            use_s3: Whether to use S3 for database storage (kept for compatibility, not used)
        """
        super().__init__()

        self.provider = provider
        self.model = model

        # Import the helper functions
        from api.config import get_embedder_type

        # Determine embedder type based on current configuration
        self.embedder_type = get_embedder_type()

        # Initialize components
        self.memory = Memory()
        self.embedder = get_embedder(embedder_type=self.embedder_type)

        # Initialize database manager
        self.initialize_db_manager()

        # Configure DSPy LM
        self._configure_dspy_lm()

        # Initialize DSPy module for generation
        self.generate_answer = dspy.ChainOfThought(RAGSignature)

    def _configure_dspy_lm(self):
        """Configure DSPy language model based on provider and model."""
        from api.config import get_model_config
        import os

        # Get model configuration
        generator_config = get_model_config(self.provider, self.model)
        model_kwargs = generator_config.get("model_kwargs", {})

        # Build model identifier based on provider
        if self.provider == "google":
            model_name = model_kwargs.get("model", "gemini-2.0-flash-exp")
            lm_identifier = f"gemini/{model_name}"
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            lm_kwargs = {"api_key": api_key} if api_key else {}

        elif self.provider == "openai":
            model_name = model_kwargs.get("model", "gpt-4o-mini")
            lm_identifier = f"openai/{model_name}"
            api_key = os.getenv("OPENAI_API_KEY")
            lm_kwargs = {"api_key": api_key} if api_key else {}

        elif self.provider == "ollama":
            model_name = model_kwargs.get("model", "llama3.2")
            lm_identifier = f"ollama_chat/{model_name}"
            api_base = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            lm_kwargs = {"api_base": api_base, "api_key": ""}

        elif self.provider == "azure":
            deployment_name = model_kwargs.get("model", "gpt-4o-mini")
            lm_identifier = f"azure/{deployment_name}"
            lm_kwargs = {
                "api_key": os.getenv("AZURE_API_KEY"),
                "api_base": os.getenv("AZURE_API_BASE"),
                "api_version": os.getenv("AZURE_API_VERSION", "2024-02-15-preview")
            }

        elif self.provider == "openrouter":
            model_name = model_kwargs.get("model", "anthropic/claude-3.5-sonnet")
            lm_identifier = f"openrouter/{model_name}"
            api_key = os.getenv("OPENROUTER_API_KEY")
            lm_kwargs = {"api_key": api_key} if api_key else {}

        else:
            # Default to Google
            logger.warning(f"Unknown provider {self.provider}, defaulting to Google")
            model_name = "gemini-2.0-flash-exp"
            lm_identifier = f"gemini/{model_name}"
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            lm_kwargs = {"api_key": api_key} if api_key else {}

        # Add common parameters
        if "temperature" in model_kwargs:
            lm_kwargs["temperature"] = model_kwargs["temperature"]
        if "max_tokens" in model_kwargs:
            lm_kwargs["max_tokens"] = model_kwargs["max_tokens"]

        # Create and configure DSPy LM
        try:
            lm = dspy.LM(lm_identifier, **lm_kwargs)
            dspy.configure(lm=lm)
            logger.info(f"Configured DSPy LM: {lm_identifier}")
        except Exception as e:
            logger.error(f"Error configuring DSPy LM: {e}")
            raise

    def initialize_db_manager(self):
        """Initialize the database manager with local storage"""
        from api.data_pipeline import DatabaseManager
        self.db_manager = DatabaseManager()
        self.transformed_docs = []
        self.retriever = None

    def prepare_retriever(
        self,
        repo_url_or_path: str,
        type: str = "github",
        access_token: str = None,
        excluded_dirs: List[str] = None,
        excluded_files: List[str] = None,
        included_dirs: List[str] = None,
        included_files: List[str] = None
    ):
        """
        Prepare the retriever for a repository.

        Args:
            repo_url_or_path: URL or local path to the repository
            type: Repository type (github, gitlab, bitbucket)
            access_token: Optional access token for private repositories
            excluded_dirs: Optional list of directories to exclude from processing
            excluded_files: Optional list of file patterns to exclude from processing
            included_dirs: Optional list of directories to include exclusively
            included_files: Optional list of file patterns to include exclusively
        """
        self.initialize_db_manager()
        self.repo_url_or_path = repo_url_or_path

        # Prepare database and get documents
        self.transformed_docs = self.db_manager.prepare_database(
            repo_url_or_path,
            type,
            access_token,
            embedder_type=self.embedder_type,
            excluded_dirs=excluded_dirs,
            excluded_files=excluded_files,
            included_dirs=included_dirs,
            included_files=included_files
        )

        logger.info(f"Loaded {len(self.transformed_docs)} documents for retrieval")

        if not self.transformed_docs:
            raise ValueError("No valid documents found. Cannot create retriever.")

        # Get the retriever from the database manager
        self.retriever = self.db_manager.retriever
        logger.info("Retriever prepared successfully")

    def call(self, query: str, language: str = "en") -> Tuple[RAGAnswer, List]:
        """
        Process a query using RAG.

        Args:
            query: The user's query
            language: Language for the response (default: "en")

        Returns:
            Tuple of (RAGAnswer, retrieved_documents)
        """
        try:
            if self.retriever is None:
                raise ValueError("Retriever not initialized. Call prepare_retriever first.")

            # Retrieve relevant documents
            retrieved_docs = self.retriever.retrieve(query)
            logger.info(f"Retrieved {len(retrieved_docs)} documents")

            # Format context from retrieved documents
            context_parts = []
            for i, doc in enumerate(retrieved_docs):
                text = doc.get('text', '')
                file_path = doc.get('metadata', {}).get('file_path', 'unknown')
                context_parts.append(f"[Document {i+1}] {file_path}:\n{text}")

            context = "\n\n".join(context_parts)

            # Format conversation history
            conv_history = self.memory()
            history_str = ""
            if conv_history:
                history_parts = []
                for turn_id, turn in conv_history.items():
                    user_q = turn.user_query.query_str
                    assistant_a = turn.assistant_response.response_str
                    history_parts.append(f"User: {user_q}\nAssistant: {assistant_a}")
                history_str = "\n\n".join(history_parts)

            # Generate answer using DSPy
            prediction = self.generate_answer(
                system_prompt=system_prompt,
                context=context,
                conversation_history=history_str,
                question=query
            )

            # Create RAGAnswer object
            rag_answer = RAGAnswer(
                rationale=prediction.rationale,
                answer=prediction.answer
            )

            # Add to memory
            self.memory.add_dialog_turn(query, prediction.answer)

            return rag_answer, retrieved_docs

        except Exception as e:
            logger.error(f"Error in RAG call: {str(e)}")

            # Create error response
            error_response = RAGAnswer(
                rationale="Error occurred while processing the query.",
                answer="I apologize, but I encountered an error while processing your question. Please try again or rephrase your question."
            )
            return error_response, []

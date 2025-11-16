import os
import subprocess
import json
import tiktoken
import logging
import base64
import glob
from urllib.parse import urlparse, urlunparse, quote
import requests
from requests.exceptions import RequestException
from typing import List, Dict, Any, Optional
from pathlib import Path

from api.config import configs, DEFAULT_EXCLUDED_DIRS, DEFAULT_EXCLUDED_FILES
from api.dspy_components import TextSplitter, get_embedder, FAISSRetriever

# Configure logging
logger = logging.getLogger(__name__)

# Maximum token limit for OpenAI embedding models
MAX_EMBEDDING_TOKENS = 8192


def get_deepwiki_root_path() -> str:
    """
    Get the root path for DeepWiki data storage.

    Returns:
        str: Path to ~/.deepwiki directory
    """
    return os.path.expanduser("~/.deepwiki")


def count_tokens(text: str, embedder_type: str = None, is_ollama_embedder: bool = None) -> int:
    """
    Count the number of tokens in a text string using tiktoken.

    Args:
        text (str): The text to count tokens for.
        embedder_type (str, optional): The embedder type ('openai', 'google', 'ollama').
                                     If None, will be determined from configuration.
        is_ollama_embedder (bool, optional): DEPRECATED. Use embedder_type instead.
                                           If None, will be determined from configuration.

    Returns:
        int: The number of tokens in the text.
    """
    try:
        # Handle backward compatibility
        if embedder_type is None and is_ollama_embedder is not None:
            embedder_type = 'ollama' if is_ollama_embedder else None

        # Determine embedder type if not specified
        if embedder_type is None:
            from api.config import get_embedder_type
            embedder_type = get_embedder_type()

        # Choose encoding based on embedder type
        if embedder_type == 'ollama':
            # Ollama typically uses cl100k_base encoding
            encoding = tiktoken.get_encoding("cl100k_base")
        elif embedder_type == 'google':
            # Google uses similar tokenization to GPT models for rough estimation
            encoding = tiktoken.get_encoding("cl100k_base")
        else:  # OpenAI or default
            # Use OpenAI embedding model encoding
            encoding = tiktoken.encoding_for_model("text-embedding-3-small")

        return len(encoding.encode(text))
    except Exception as e:
        # Fallback to a simple approximation if tiktoken fails
        logger.warning(f"Error counting tokens with tiktoken: {e}")
        # Rough approximation: 4 characters per token
        return len(text) // 4


def download_repo(repo_url: str, local_path: str, repo_type: str = None, access_token: str = None) -> str:
    """
    Downloads a Git repository (GitHub, GitLab, or Bitbucket) to a specified local path.

    Args:
        repo_type(str): Type of repository
        repo_url (str): The URL of the Git repository to clone.
        local_path (str): The local directory where the repository will be cloned.
        access_token (str, optional): Access token for private repositories.

    Returns:
        str: The output message from the `git` command.
    """
    try:
        # Check if Git is installed
        logger.info(f"Preparing to clone repository to {local_path}")
        subprocess.run(
            ["git", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Check if repository already exists
        if os.path.exists(local_path) and os.listdir(local_path):
            # Directory exists and is not empty
            logger.warning(f"Repository already exists at {local_path}. Using existing repository.")
            return f"Using existing repository at {local_path}"

        # Ensure the local path exists
        os.makedirs(local_path, exist_ok=True)

        # Prepare authentication for private repositories
        if access_token:
            # Parse the repository URL
            parsed_url = urlparse(repo_url)

            # Insert the access token into the URL
            # Format: https://token@domain/path
            authenticated_url = urlunparse((
                parsed_url.scheme,
                f"{quote(access_token, safe='')}@{parsed_url.netloc}",
                parsed_url.path,
                parsed_url.params,
                parsed_url.query,
                parsed_url.fragment
            ))
        else:
            authenticated_url = repo_url

        # Clone the repository
        logger.info("Starting repository clone...")
        result = subprocess.run(
            ["git", "clone", authenticated_url, local_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        logger.info("Repository cloned successfully")
        return result.stdout + result.stderr

    except subprocess.CalledProcessError as e:
        error_msg = f"Failed to clone repository: {e.stderr}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def read_all_documents(
    repo_path: str,
    embedder_type: str = None,
    is_ollama_embedder: bool = None,
    excluded_dirs: List[str] = None,
    excluded_files: List[str] = None,
    included_dirs: List[str] = None,
    included_files: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Reads all documents from a repository directory.

    Args:
        repo_path (str): The path to the repository directory
        embedder_type (str, optional): The embedder type ('openai', 'google', 'ollama')
        is_ollama_embedder (bool, optional): DEPRECATED. Use embedder_type instead.
        excluded_dirs (List[str], optional): List of directories to exclude from processing
        excluded_files (List[str], optional): List of file patterns to exclude from processing
        included_dirs (List[str], optional): List of directories to include exclusively
        included_files (List[str], optional): List of file patterns to include exclusively

    Returns:
        List[Dict]: List of document dictionaries with 'text' and 'metadata' fields
    """
    # Handle backward compatibility
    if embedder_type is None and is_ollama_embedder is not None:
        embedder_type = 'ollama' if is_ollama_embedder else None

    if embedder_type is None:
        from api.config import get_embedder_type
        embedder_type = get_embedder_type()

    # Use provided exclusions or defaults
    if excluded_dirs is None:
        excluded_dirs = DEFAULT_EXCLUDED_DIRS
    if excluded_files is None:
        excluded_files = DEFAULT_EXCLUDED_FILES

    documents = []

    # Walk through the repository
    for root, dirs, files in os.walk(repo_path):
        # Filter directories
        if excluded_dirs:
            dirs[:] = [d for d in dirs if d not in excluded_dirs and not any(d.startswith(ex) for ex in excluded_dirs)]

        if included_dirs:
            dirs[:] = [d for d in dirs if d in included_dirs or any(d.startswith(inc) for inc in included_dirs)]

        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, repo_path)

            # Apply file filters
            if excluded_files and any(file.endswith(ext) or file == ext for ext in excluded_files):
                continue

            if included_files and not any(file.endswith(ext) or file == ext for ext in included_files):
                continue

            try:
                # Try to read as text
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # Skip empty files
                if not content.strip():
                    continue

                # Check token count and skip if too large
                token_count = count_tokens(content, embedder_type)
                if token_count > MAX_EMBEDDING_TOKENS:
                    logger.warning(f"Skipping {relative_path}: {token_count} tokens exceeds limit")
                    continue

                doc = {
                    'text': content,
                    'metadata': {
                        'file_path': relative_path,
                        'full_path': file_path,
                        'source_type': 'code',
                        'token_count': token_count
                    }
                }
                documents.append(doc)

            except Exception as e:
                logger.warning(f"Error reading file {relative_path}: {e}")
                continue

    logger.info(f"Read {len(documents)} documents from {repo_path}")
    return documents


def prepare_data_pipeline(embedder_type: str = None, is_ollama_embedder: bool = None) -> tuple:
    """
    Prepares the data processing pipeline with text splitter and embedder.

    Args:
        embedder_type (str, optional): The embedder type ('openai', 'google', 'ollama').
                                     If None, will be determined from configuration.
        is_ollama_embedder (bool, optional): DEPRECATED. Use embedder_type instead.
                                           If None, will be determined from configuration.

    Returns:
        tuple: (text_splitter, embedder) - The data transformation components
    """
    from api.config import get_embedder_type

    # Handle backward compatibility
    if embedder_type is None and is_ollama_embedder is not None:
        embedder_type = 'ollama' if is_ollama_embedder else None

    # Determine embedder type if not specified
    if embedder_type is None:
        embedder_type = get_embedder_type()

    # Create text splitter
    splitter = TextSplitter(**configs["text_splitter"])

    # Create embedder
    embedder = get_embedder(embedder_type=embedder_type)

    return splitter, embedder


def transform_documents_and_save_to_db(
    documents: List[Dict[str, Any]],
    db_path: str,
    embedder_type: str = None,
    is_ollama_embedder: bool = None
) -> FAISSRetriever:
    """
    Transforms a list of documents and saves them to a FAISS index.

    Args:
        documents (list): A list of document dictionaries
        db_path (str): The path to save the FAISS index
        embedder_type (str, optional): The embedder type ('openai', 'google', 'ollama').
                                     If None, will be determined from configuration.
        is_ollama_embedder (bool, optional): DEPRECATED. Use embedder_type instead.
                                           If None, will be determined from configuration.

    Returns:
        FAISSRetriever: The retriever with built index
    """
    # Get the data transformation components
    splitter, embedder = prepare_data_pipeline(embedder_type, is_ollama_embedder)

    # Split documents into chunks
    logger.info(f"Splitting {len(documents)} documents into chunks...")
    chunked_documents = splitter.split_documents(documents)
    logger.info(f"Created {len(chunked_documents)} chunks")

    # Embed all chunks
    logger.info("Embedding chunks...")
    for doc in chunked_documents:
        embedding = embedder(doc['text'])
        doc['embedding'] = embedding

    # Create FAISS retriever and build index
    logger.info("Building FAISS index...")
    retriever = FAISSRetriever(
        embedder=embedder,
        top_k=configs.get("retriever", {}).get("top_k", 20),
        documents=chunked_documents
    )

    # Save to disk
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db_path_obj = Path(db_path).with_suffix('')  # Remove .pkl extension
    retriever.save(db_path_obj)
    logger.info(f"Saved FAISS index to {db_path_obj}")

    return retriever


def get_github_file_content(repo_url: str, file_path: str, access_token: str = None) -> str:
    """
    Retrieves the content of a file from a GitHub repository using the GitHub API.
    Supports both public GitHub (github.com) and GitHub Enterprise (custom domains).

    Args:
        repo_url (str): The URL of the GitHub repository
                       (e.g., "https://github.com/username/repo" or "https://github.company.com/username/repo")
        file_path (str): The path to the file within the repository (e.g., "src/main.py")
        access_token (str, optional): GitHub personal access token for private repositories

    Returns:
        str: The content of the file as a string

    Raises:
        ValueError: If the file cannot be fetched or if the URL is not a valid GitHub URL
    """
    try:
        # Parse the repository URL to support both github.com and enterprise GitHub
        parsed_url = urlparse(repo_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Not a valid GitHub repository URL")

        # Check if it's a GitHub-like URL structure
        path_parts = parsed_url.path.strip('/').split('/')
        if len(path_parts) < 2:
            raise ValueError("Invalid GitHub URL format - expected format: https://domain/owner/repo")

        owner = path_parts[-2]
        repo = path_parts[-1].replace(".git", "")

        # Determine the API base URL
        if parsed_url.netloc == "github.com":
            # Public GitHub
            api_base = "https://api.github.com"
        else:
            # GitHub Enterprise - API is typically at https://domain/api/v3/
            api_base = f"{parsed_url.scheme}://{parsed_url.netloc}/api/v3"

        # Construct API URL to get file content
        api_url = f"{api_base}/repos/{owner}/{repo}/contents/{file_path}"

        # Set up headers
        headers = {"Accept": "application/vnd.github.v3+json"}
        if access_token:
            headers["Authorization"] = f"token {access_token}"

        # Make API request
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()

        # Parse response
        file_data = response.json()

        # Decode content (GitHub API returns base64-encoded content)
        if "content" in file_data:
            content = base64.b64decode(file_data["content"]).decode("utf-8")
            return content
        else:
            raise ValueError(f"No content found in file {file_path}")

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            raise ValueError(f"File not found: {file_path}")
        elif e.response.status_code == 403:
            raise ValueError("Access denied. You may need to provide an access token.")
        else:
            raise ValueError(f"HTTP error occurred: {e}")
    except Exception as e:
        raise ValueError(f"Error fetching file from GitHub: {str(e)}")


def get_gitlab_file_content(repo_url: str, file_path: str, access_token: str = None) -> str:
    """
    Retrieves the content of a file from a GitLab repository using the GitLab API.
    Supports both gitlab.com and self-hosted GitLab instances.

    Args:
        repo_url (str): The URL of the GitLab repository
        file_path (str): The path to the file within the repository
        access_token (str, optional): GitLab personal access token

    Returns:
        str: The content of the file as a string

    Raises:
        ValueError: If the file cannot be fetched
    """
    try:
        parsed_url = urlparse(repo_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Not a valid GitLab repository URL")

        path_parts = parsed_url.path.strip('/').split('/')
        if len(path_parts) < 2:
            raise ValueError("Invalid GitLab URL format")

        # GitLab can have group/subgroup/repo structure
        owner = path_parts[-2]
        repo = path_parts[-1].replace(".git", "")

        # For nested groups, join all parts except the last one
        if len(path_parts) > 2:
            project_path = '/'.join(path_parts[:-1]) + '/' + repo
        else:
            project_path = f"{owner}/{repo}"

        # URL encode the project path
        encoded_project_path = quote(project_path, safe='')

        # Determine API base URL
        if parsed_url.netloc == "gitlab.com":
            api_base = "https://gitlab.com/api/v4"
        else:
            # Self-hosted GitLab
            api_base = f"{parsed_url.scheme}://{parsed_url.netloc}/api/v4"

        # URL encode the file path
        encoded_file_path = quote(file_path, safe='')

        # Construct API URL
        api_url = f"{api_base}/projects/{encoded_project_path}/repository/files/{encoded_file_path}/raw"

        headers = {}
        if access_token:
            headers["PRIVATE-TOKEN"] = access_token

        response = requests.get(api_url, headers=headers)
        response.raise_for_status()

        return response.text

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            raise ValueError(f"File not found: {file_path}")
        elif e.response.status_code == 401:
            raise ValueError("Authentication failed. Check your access token.")
        else:
            raise ValueError(f"HTTP error occurred: {e}")
    except Exception as e:
        raise ValueError(f"Error fetching file from GitLab: {str(e)}")


def get_bitbucket_file_content(repo_url: str, file_path: str, access_token: str = None) -> str:
    """
    Retrieves the content of a file from a Bitbucket repository using the Bitbucket API.

    Args:
        repo_url (str): The URL of the Bitbucket repository
        file_path (str): The path to the file within the repository
        access_token (str, optional): Bitbucket access token or app password

    Returns:
        str: The content of the file as a string

    Raises:
        ValueError: If the file cannot be fetched
    """
    try:
        parsed_url = urlparse(repo_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Not a valid Bitbucket repository URL")

        path_parts = parsed_url.path.strip('/').split('/')
        if len(path_parts) < 2:
            raise ValueError("Invalid Bitbucket URL format")

        workspace = path_parts[-2]
        repo_slug = path_parts[-1].replace(".git", "")

        # Bitbucket API v2
        api_url = f"https://api.bitbucket.org/2.0/repositories/{workspace}/{repo_slug}/src/master/{file_path}"

        headers = {}
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"

        response = requests.get(api_url, headers=headers)
        response.raise_for_status()

        return response.text

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            raise ValueError(f"File not found: {file_path}")
        elif e.response.status_code == 401:
            raise ValueError("Authentication failed. Check your access token.")
        else:
            raise ValueError(f"HTTP error occurred: {e}")
    except Exception as e:
        raise ValueError(f"Error fetching file from Bitbucket: {str(e)}")


def get_file_content_from_repo(
    repo_url: str,
    file_path: str,
    repo_type: str = "github",
    access_token: str = None
) -> str:
    """
    Generic function to get file content from any supported repository type.

    Args:
        repo_url (str): The URL of the repository
        file_path (str): The path to the file within the repository
        repo_type (str): Type of repository ('github', 'gitlab', or 'bitbucket')
        access_token (str, optional): Access token for private repositories

    Returns:
        str: The content of the file as a string

    Raises:
        ValueError: If repository type is not supported or file cannot be fetched
    """
    if repo_type == "github":
        return get_github_file_content(repo_url, file_path, access_token)
    elif repo_type == "gitlab":
        return get_gitlab_file_content(repo_url, file_path, access_token)
    elif repo_type == "bitbucket":
        return get_bitbucket_file_content(repo_url, file_path, access_token)
    else:
        raise ValueError("Unsupported repository type. Only GitHub, GitLab, and Bitbucket are supported.")


class DatabaseManager:
    """
    Manages the creation, loading, transformation, and persistence of FAISS retriever instances.
    """

    def __init__(self):
        self.retriever = None
        self.repo_url_or_path = None
        self.repo_paths = None

    def prepare_database(
        self,
        repo_url_or_path: str,
        repo_type: str = None,
        access_token: str = None,
        embedder_type: str = None,
        is_ollama_embedder: bool = None,
        excluded_dirs: List[str] = None,
        excluded_files: List[str] = None,
        included_dirs: List[str] = None,
        included_files: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Create a new database from the repository.

        Args:
            repo_type(str): Type of repository
            repo_url_or_path (str): The URL or local path of the repository
            access_token (str, optional): Access token for private repositories
            embedder_type (str, optional): Embedder type to use ('openai', 'google', 'ollama').
                                         If None, will be determined from configuration.
            is_ollama_embedder (bool, optional): DEPRECATED. Use embedder_type instead.
                                               If None, will be determined from configuration.
            excluded_dirs (List[str], optional): List of directories to exclude from processing
            excluded_files (List[str], optional): List of file patterns to exclude from processing
            included_dirs (List[str], optional): List of directories to include exclusively
            included_files (List[str], optional): List of file patterns to include exclusively

        Returns:
            List[Dict]: List of document dictionaries
        """
        # Handle backward compatibility
        if embedder_type is None and is_ollama_embedder is not None:
            embedder_type = 'ollama' if is_ollama_embedder else None

        self.reset_database()
        self._create_repo(repo_url_or_path, repo_type, access_token)
        return self.prepare_db_index(
            embedder_type=embedder_type,
            excluded_dirs=excluded_dirs,
            excluded_files=excluded_files,
            included_dirs=included_dirs,
            included_files=included_files
        )

    def reset_database(self):
        """
        Reset the database to its initial state.
        """
        self.retriever = None
        self.repo_url_or_path = None
        self.repo_paths = None

    def _extract_repo_name_from_url(self, repo_url_or_path: str, repo_type: str) -> str:
        # Extract owner and repo name to create unique identifier
        url_parts = repo_url_or_path.rstrip('/').split('/')

        if repo_type in ["github", "gitlab", "bitbucket"] and len(url_parts) >= 5:
            # GitHub URL format: https://github.com/owner/repo
            # GitLab URL format: https://gitlab.com/owner/repo or https://gitlab.com/group/subgroup/repo
            # Bitbucket URL format: https://bitbucket.org/owner/repo
            owner = url_parts[-2]
            repo = url_parts[-1].replace(".git", "")
            repo_name = f"{owner}_{repo}"
        else:
            repo_name = url_parts[-1].replace(".git", "")
        return repo_name

    def _create_repo(self, repo_url_or_path: str, repo_type: str = None, access_token: str = None) -> None:
        """
        Download and prepare all paths.
        Paths:
        ~/.deepwiki/repos/{owner}_{repo_name} (for url, local path will be the same)
        ~/.deepwiki/databases/{owner}_{repo_name}

        Args:
            repo_type(str): Type of repository
            repo_url_or_path (str): The URL or local path of the repository
            access_token (str, optional): Access token for private repositories
        """
        logger.info(f"Preparing repo storage for {repo_url_or_path}...")

        try:
            root_path = get_deepwiki_root_path()

            os.makedirs(root_path, exist_ok=True)
            # url
            if repo_url_or_path.startswith("https://") or repo_url_or_path.startswith("http://"):
                # Extract the repository name from the URL
                repo_name = self._extract_repo_name_from_url(repo_url_or_path, repo_type)
                logger.info(f"Extracted repo name: {repo_name}")

                save_repo_dir = os.path.join(root_path, "repos", repo_name)

                # Check if the repository directory already exists and is not empty
                if not (os.path.exists(save_repo_dir) and os.listdir(save_repo_dir)):
                    # Only download if the repository doesn't exist or is empty
                    download_repo(repo_url_or_path, save_repo_dir, repo_type, access_token)
                else:
                    logger.info(f"Repository already exists at {save_repo_dir}. Using existing repository.")
            else:  # local path
                repo_name = os.path.basename(repo_url_or_path)
                save_repo_dir = repo_url_or_path

            save_db_file = os.path.join(root_path, "databases", repo_name)
            os.makedirs(save_repo_dir, exist_ok=True)
            os.makedirs(os.path.dirname(save_db_file), exist_ok=True)

            self.repo_paths = {
                "save_repo_dir": save_repo_dir,
                "save_db_file": save_db_file,
            }
            self.repo_url_or_path = repo_url_or_path
            logger.info(f"Repo paths: {self.repo_paths}")

        except Exception as e:
            logger.error(f"Failed to create repository structure: {e}")
            raise

    def prepare_db_index(
        self,
        embedder_type: str = None,
        is_ollama_embedder: bool = None,
        excluded_dirs: List[str] = None,
        excluded_files: List[str] = None,
        included_dirs: List[str] = None,
        included_files: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Prepare the indexed database for the repository.

        Args:
            embedder_type (str, optional): Embedder type to use ('openai', 'google', 'ollama').
                                         If None, will be determined from configuration.
            is_ollama_embedder (bool, optional): DEPRECATED. Use embedder_type instead.
                                               If None, will be determined from configuration.
            excluded_dirs (List[str], optional): List of directories to exclude from processing
            excluded_files (List[str], optional): List of file patterns to exclude from processing
            included_dirs (List[str], optional): List of directories to include exclusively
            included_files (List[str], optional): List of file patterns to include exclusively

        Returns:
            List[Dict]: List of document dictionaries
        """
        # Handle backward compatibility
        if embedder_type is None and is_ollama_embedder is not None:
            embedder_type = 'ollama' if is_ollama_embedder else None

        # Get embedder for retriever
        embedder = get_embedder(embedder_type=embedder_type)

        # Check if database exists
        db_path = Path(self.repo_paths["save_db_file"])
        faiss_file = db_path.with_suffix('.faiss')
        pkl_file = db_path.with_suffix('.pkl')

        if faiss_file.exists() and pkl_file.exists():
            logger.info("Loading existing database...")
            try:
                self.retriever = FAISSRetriever(
                    embedder=embedder,
                    index_path=db_path
                )
                documents = self.retriever.documents
                if documents:
                    logger.info(f"Loaded {len(documents)} documents from existing database")
                    return documents
            except Exception as e:
                logger.error(f"Error loading existing database: {e}")
                # Continue to create a new database

        # Prepare the database
        logger.info("Creating new database...")
        documents = read_all_documents(
            self.repo_paths["save_repo_dir"],
            embedder_type=embedder_type,
            excluded_dirs=excluded_dirs,
            excluded_files=excluded_files,
            included_dirs=included_dirs,
            included_files=included_files
        )

        logger.info(f"Total documents: {len(documents)}")

        self.retriever = transform_documents_and_save_to_db(
            documents, self.repo_paths["save_db_file"], embedder_type=embedder_type
        )

        transformed_docs = self.retriever.documents
        logger.info(f"Total transformed documents: {len(transformed_docs)}")
        return transformed_docs

    def prepare_retriever(self, repo_url_or_path: str, repo_type: str = None, access_token: str = None):
        """
        Prepare the retriever for a repository.
        This is a compatibility method for the isolated API.

        Args:
            repo_type(str): Type of repository
            repo_url_or_path (str): The URL or local path of the repository
            access_token (str, optional): Access token for private repositories

        Returns:
            List[Dict]: List of document dictionaries
        """
        return self.prepare_database(repo_url_or_path, repo_type, access_token)

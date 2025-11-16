"""FAISS-based retriever for DSPy."""

import pickle
import faiss
import numpy as np
from typing import List, Optional, Callable, Any
from pathlib import Path
import dspy


class FAISSRetriever:
    """
    FAISS-based document retriever compatible with DSPy.

    This retriever stores documents and their embeddings in a FAISS index
    for efficient similarity search.
    """

    def __init__(
        self,
        embedder: dspy.Embedder,
        top_k: int = 20,
        documents: Optional[List[dict]] = None,
        index_path: Optional[Path] = None
    ):
        """
        Initialize FAISS retriever.

        Args:
            embedder: DSPy embedder for query embedding
            top_k: Number of top results to retrieve
            documents: List of documents with 'text' and 'embedding' fields
            index_path: Path to load/save FAISS index
        """
        self.embedder = embedder
        self.top_k = top_k
        self.documents = []
        self.index = None
        self.dimension = None
        self.index_path = index_path

        if documents:
            self.build_index(documents)
        elif index_path and index_path.exists():
            self.load(index_path)

    def build_index(self, documents: List[dict]):
        """
        Build FAISS index from documents.

        Args:
            documents: List of dicts with 'text' and 'embedding' fields
        """
        if not documents:
            raise ValueError("Cannot build index with empty documents list")

        self.documents = documents

        # Extract embeddings
        embeddings = []
        for doc in documents:
            if 'embedding' in doc:
                embeddings.append(doc['embedding'])
            else:
                raise ValueError("Document missing 'embedding' field")

        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Get dimension
        self.dimension = embeddings_array.shape[1]

        # Create FAISS index (using L2 distance)
        self.index = faiss.IndexFlatL2(self.dimension)

        # Add vectors to index
        self.index.add(embeddings_array)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[dict]:
        """
        Retrieve top-k most similar documents for a query.

        Args:
            query: Query text
            top_k: Number of results (overrides self.top_k if provided)

        Returns:
            List of document dicts with similarity scores
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        k = top_k if top_k is not None else self.top_k

        # Embed the query
        query_embedding = self.embedder(query)

        # Ensure it's the right shape
        query_vector = np.array([query_embedding], dtype=np.float32)

        # Search FAISS index
        distances, indices = self.index.search(query_vector, k)

        # Retrieve documents
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['score'] = float(distances[0][i])
                doc['rank'] = i + 1
                results.append(doc)

        return results

    def __call__(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """
        Retrieve documents and return their text content.

        This makes the retriever compatible with DSPy's Retrieve interface.

        Args:
            query: Query text
            top_k: Number of results

        Returns:
            List of document texts
        """
        results = self.retrieve(query, top_k)
        return [doc['text'] for doc in results]

    def save(self, path: Path):
        """
        Save FAISS index and documents to disk.

        Args:
            path: Base path for saving (will create .faiss and .pkl files)
        """
        if self.index is None:
            raise ValueError("No index to save")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_file = path.with_suffix('.faiss')
        faiss.write_index(self.index, str(index_file))

        # Save documents and metadata
        metadata = {
            'documents': self.documents,
            'dimension': self.dimension,
            'top_k': self.top_k
        }
        metadata_file = path.with_suffix('.pkl')
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)

        self.index_path = path

    def load(self, path: Path):
        """
        Load FAISS index and documents from disk.

        Args:
            path: Base path for loading (will read .faiss and .pkl files)
        """
        path = Path(path)

        # Load FAISS index
        index_file = path.with_suffix('.faiss')
        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_file}")

        self.index = faiss.read_index(str(index_file))

        # Load documents and metadata
        metadata_file = path.with_suffix('.pkl')
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)

        self.documents = metadata['documents']
        self.dimension = metadata['dimension']
        self.top_k = metadata.get('top_k', self.top_k)
        self.index_path = path

    def get_document_count(self) -> int:
        """Get the number of documents in the index."""
        return len(self.documents)

    def add_documents(self, new_documents: List[dict]):
        """
        Add new documents to existing index.

        Args:
            new_documents: List of dicts with 'text' and 'embedding' fields
        """
        if not new_documents:
            return

        # Extract embeddings
        new_embeddings = []
        for doc in new_documents:
            if 'embedding' not in doc:
                raise ValueError("Document missing 'embedding' field")
            new_embeddings.append(doc['embedding'])

        # Convert to numpy array
        embeddings_array = np.array(new_embeddings, dtype=np.float32)

        # Initialize index if it doesn't exist
        if self.index is None:
            self.dimension = embeddings_array.shape[1]
            self.index = faiss.IndexFlatL2(self.dimension)

        # Add to index
        self.index.add(embeddings_array)

        # Add to documents list
        self.documents.extend(new_documents)

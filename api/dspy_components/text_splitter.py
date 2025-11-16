"""Text splitter for document chunking."""

from typing import List


class TextSplitter:
    """
    Word-based text splitter compatible with DSPy.

    Splits text into chunks based on word count with configurable overlap.
    This maintains compatibility with the previous Adalflow text splitter behavior.
    """

    def __init__(
        self,
        split_by: str = "word",
        chunk_size: int = 350,
        chunk_overlap: int = 100
    ):
        """
        Initialize the text splitter.

        Args:
            split_by: Splitting method ('word' is the only supported method)
            chunk_size: Number of words per chunk
            chunk_overlap: Number of words to overlap between chunks
        """
        if split_by != "word":
            raise ValueError(f"Only 'word' splitting is supported, got: {split_by}")

        self.split_by = split_by
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        """
        Split text into chunks.

        Args:
            text: The text to split

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        words = text.split()

        if len(words) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk = ' '.join(chunk_words)
            chunks.append(chunk)

            # Move start position considering overlap
            start += (self.chunk_size - self.chunk_overlap)

        return chunks

    def split_documents(self, documents: List[dict]) -> List[dict]:
        """
        Split multiple documents into chunks.

        Args:
            documents: List of document dicts with 'text' and 'metadata' fields

        Returns:
            List of chunked documents with metadata preserved
        """
        chunked_docs = []

        for doc in documents:
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})

            chunks = self.split(text)

            for i, chunk in enumerate(chunks):
                chunked_doc = {
                    'text': chunk,
                    'metadata': {
                        **metadata,
                        'chunk_index': i,
                        'total_chunks': len(chunks)
                    }
                }
                chunked_docs.append(chunked_doc)

        return chunked_docs

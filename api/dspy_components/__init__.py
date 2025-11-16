"""DSPy components for DeepWiki."""

from .text_splitter import TextSplitter
from .embedder_utils import get_embedder, EmbedderType
from .faiss_retriever import FAISSRetriever

__all__ = [
    "TextSplitter",
    "get_embedder",
    "EmbedderType",
    "FAISSRetriever",
]

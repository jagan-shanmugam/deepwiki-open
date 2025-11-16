"""
Embedder utilities - wrapper for DSPy embedders.

This module provides a compatibility layer for the old embedder interface,
now powered by DSPy instead of Adalflow.
"""

# Simply re-export from DSPy components
from api.dspy_components.embedder_utils import (
    get_embedder,
    EmbedderType,
    embed_texts,
    get_embedding_dimension
)

__all__ = [
    'get_embedder',
    'EmbedderType',
    'embed_texts',
    'get_embedding_dimension'
]

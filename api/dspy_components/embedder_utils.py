"""Embedder utilities for DSPy."""

import os
import json
from enum import Enum
from typing import List, Optional
from pathlib import Path
import dspy


class EmbedderType(str, Enum):
    """Supported embedder types."""
    OPENAI = "openai"
    GOOGLE = "google"
    OLLAMA = "ollama"


def get_config_path(filename: str = "embedder.json") -> Path:
    """Get path to configuration file."""
    config_dir = os.getenv("DEEPWIKI_CONFIG_DIR")
    if config_dir:
        return Path(config_dir) / filename

    # Default to config directory relative to this file
    return Path(__file__).parent.parent / "config" / filename


def load_embedder_config() -> dict:
    """Load embedder configuration from JSON file."""
    config_path = get_config_path("embedder.json")
    with open(config_path, 'r') as f:
        return json.load(f)


def get_embedder(
    embedder_type: Optional[str] = None,
    **kwargs
) -> dspy.Embedder:
    """
    Get configured DSPy embedder based on type.

    Args:
        embedder_type: Type of embedder ('openai', 'google', 'ollama')
                      If None, uses DEEPWIKI_EMBEDDER_TYPE env var or defaults to 'openai'
        **kwargs: Additional arguments to pass to embedder

    Returns:
        Configured DSPy Embedder instance
    """
    if embedder_type is None:
        embedder_type = os.getenv('DEEPWIKI_EMBEDDER_TYPE', 'openai')

    embedder_type = embedder_type.lower()

    # Load configuration
    config = load_embedder_config()

    if embedder_type == EmbedderType.OPENAI:
        return _get_openai_embedder(config, **kwargs)
    elif embedder_type == EmbedderType.GOOGLE:
        return _get_google_embedder(config, **kwargs)
    elif embedder_type == EmbedderType.OLLAMA:
        return _get_ollama_embedder(config, **kwargs)
    else:
        raise ValueError(
            f"Unsupported embedder type: {embedder_type}. "
            f"Supported types: {[e.value for e in EmbedderType]}"
        )


def _get_openai_embedder(config: dict, **kwargs) -> dspy.Embedder:
    """Get OpenAI embedder."""
    embedder_config = config.get('embedder', {})
    model_kwargs = embedder_config.get('model_kwargs', {})

    model = model_kwargs.get('model', 'text-embedding-3-small')
    dimensions = model_kwargs.get('dimensions', 256)

    api_key = kwargs.get('api_key') or os.getenv('OPENAI_API_KEY')

    embedder_kwargs = {
        'dimensions': dimensions,
    }

    if api_key:
        embedder_kwargs['api_key'] = api_key

    # Add any custom base URL if provided
    api_base = kwargs.get('api_base') or os.getenv('OPENAI_API_BASE')
    if api_base:
        embedder_kwargs['api_base'] = api_base

    return dspy.Embedder(f'openai/{model}', **embedder_kwargs)


def _get_google_embedder(config: dict, **kwargs) -> dspy.Embedder:
    """Get Google embedder."""
    embedder_config = config.get('embedder_google', {})
    model_kwargs = embedder_config.get('model_kwargs', {})

    model = model_kwargs.get('model', 'text-embedding-004')

    # DSPy uses GEMINI_API_KEY for Google embeddings
    api_key = kwargs.get('api_key') or os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')

    embedder_kwargs = {}
    if api_key:
        embedder_kwargs['api_key'] = api_key

    return dspy.Embedder(f'gemini/{model}', **embedder_kwargs)


def _get_ollama_embedder(config: dict, **kwargs) -> dspy.Embedder:
    """Get Ollama embedder."""
    embedder_config = config.get('embedder_ollama', {})
    model_kwargs = embedder_config.get('model_kwargs', {})

    model = model_kwargs.get('model', 'nomic-embed-text')

    api_base = kwargs.get('api_base') or os.getenv('OLLAMA_HOST', 'http://localhost:11434')

    return dspy.Embedder(
        f'ollama/{model}',
        api_base=api_base,
        api_key=''  # Ollama doesn't require API key
    )


def embed_texts(
    texts: List[str],
    embedder: Optional[dspy.Embedder] = None,
    embedder_type: Optional[str] = None
) -> List[List[float]]:
    """
    Embed a list of texts.

    Args:
        texts: List of texts to embed
        embedder: Pre-configured embedder. If None, creates one using embedder_type
        embedder_type: Type of embedder to create if embedder is None

    Returns:
        List of embedding vectors
    """
    if embedder is None:
        embedder = get_embedder(embedder_type)

    embeddings = []
    for text in texts:
        embedding = embedder(text)
        embeddings.append(embedding)

    return embeddings


def get_embedding_dimension(embedder_type: Optional[str] = None) -> int:
    """
    Get the embedding dimension for a given embedder type.

    Args:
        embedder_type: Type of embedder

    Returns:
        Embedding dimension
    """
    if embedder_type is None:
        embedder_type = os.getenv('DEEPWIKI_EMBEDDER_TYPE', 'openai')

    embedder_type = embedder_type.lower()
    config = load_embedder_config()

    if embedder_type == EmbedderType.OPENAI:
        return config.get('embedder', {}).get('model_kwargs', {}).get('dimensions', 256)
    elif embedder_type == EmbedderType.GOOGLE:
        # Google embeddings have fixed dimensions
        return 768
    elif embedder_type == EmbedderType.OLLAMA:
        # Nomic embed text has 768 dimensions
        return 768
    else:
        raise ValueError(f"Unknown embedder type: {embedder_type}")

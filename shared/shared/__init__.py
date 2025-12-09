"""Shared modules for RAG system.

Provides common functionality for embedding and vector storage.
"""

from .embedding import EmbeddingModel, get_embedding_model
from .vector_store import VectorStore, get_vector_store

__all__ = [
    "EmbeddingModel",
    "get_embedding_model",
    "VectorStore",
    "get_vector_store",
]

__version__ = "0.1.0"

"""Embedding model wrapper for multilingual text embedding.

Uses intfloat/multilingual-e5-large-instruct model (1024 dimensions).
Supports instruction prefixes for optimal retrieval performance.
"""

from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Wrapper for sentence-transformers embedding model.

    Uses intfloat/multilingual-e5-large-instruct by default.
    Supports instruction prefixes: "passage:" for documents, "query:" for queries.
    """

    DEFAULT_MODEL = "intfloat/multilingual-e5-large-instruct"
    DEFAULT_DIMENSION = 1024

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """Initialize embedding model.

        Args:
            model_name: HuggingFace model name. Defaults to multilingual-e5-large-instruct.
            device: Device to use ('cpu', 'cuda', 'mps'). Auto-detected if None.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model: Optional[SentenceTransformer] = None
        self._device = device

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the model on first access."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self._device)
        return self._model

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

    def embed_documents(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """Embed documents for indexing.

        Automatically adds "passage: " prefix for E5 models.

        Args:
            texts: List of document texts to embed.
            batch_size: Batch size for encoding.
            show_progress: Whether to show progress bar.

        Returns:
            List of embedding vectors as float lists.
        """
        if not texts:
            return []

        # Add instruction prefix for E5 models
        prefixed_texts = [f"passage: {text}" for text in texts]

        embeddings = self.model.encode(
            prefixed_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a query for search.

        Automatically adds "query: " prefix for E5 models.

        Args:
            query: Query text to embed.

        Returns:
            Embedding vector as float list.
        """
        # Add instruction prefix for E5 models
        prefixed_query = f"query: {query}"

        embedding = self.model.encode(
            prefixed_query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    def compute_similarity(
        self,
        embedding1: list[float],
        embedding2: list[float],
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.

        Returns:
            Cosine similarity score between -1 and 1.
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Normalize and compute dot product (cosine similarity)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))


# Default model instance (lazy-loaded)
_default_model: Optional[EmbeddingModel] = None


def get_embedding_model(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> EmbeddingModel:
    """Get embedding model instance.

    Returns cached default instance if no custom parameters provided.

    Args:
        model_name: Optional custom model name.
        device: Optional device specification.

    Returns:
        EmbeddingModel instance.
    """
    global _default_model

    if model_name is None and device is None:
        if _default_model is None:
            _default_model = EmbeddingModel()
        return _default_model

    return EmbeddingModel(model_name=model_name, device=device)

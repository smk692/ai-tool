"""
Hugging Face Embedding Service

Provides text embedding functionality using sentence-transformers library.
Supports multilingual text (Korean, English, Japanese, Chinese, 50+ languages).
"""

import logging
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.models.embedding import EmbeddingConfiguration

logger = logging.getLogger(__name__)


class HuggingFaceEmbedding:
    """
    Hugging Face sentence-transformers based embedding service.

    Responsibilities:
    - Convert text to 384-dimensional embedding vectors
    - Optimize throughput with batch processing
    - Generate L2 normalized vectors for cosine similarity

    Example:
        >>> config = EmbeddingConfiguration()
        >>> service = HuggingFaceEmbedding(config)
        >>> vector = service.embed_text("PostgreSQL 트랜잭션")
        >>> len(vector)
        384
    """

    def __init__(self, config: EmbeddingConfiguration):
        """
        Initialize embedding service.

        Args:
            config: Embedding configuration (model name, device, etc.)

        Raises:
            RuntimeError: If model loading fails
        """
        self.config = config
        self.embedding_dim = config.embedding_dim

        try:
            # Load model with specified device
            self.model = SentenceTransformer(
                config.model_name,
                device=config.device.value if hasattr(config.device, 'value') else str(config.device)
            )

            logger.info(
                f"Initialized HuggingFaceEmbedding with model={config.model_name}, "
                f"device={config.device}, dim={self.embedding_dim}"
            )

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    def embed_text(self, text: str) -> List[float]:
        """
        Convert single text to embedding vector.

        Args:
            text: Input text (max 512 tokens, auto-truncated)

        Returns:
            384-dimensional L2 normalized embedding vector

        Raises:
            ValueError: If text is empty

        Example:
            >>> vector = service.embed_text("안녕하세요")
            >>> len(vector)
            384
        """
        # Validate input
        if not text or not text.strip():
            raise ValueError("Empty text cannot be embedded")

        try:
            # Generate embedding with L2 normalization
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2 normalization
                show_progress_bar=False
            )

            logger.debug(f"Embedded text (length={len(text)}) to {len(embedding)}-dim vector")

            return embedding.tolist()

        except Exception as e:
            logger.error(f"Text embedding failed: {e}")
            raise

    def embed_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """
        Convert multiple texts to embedding vectors using batch processing.

        Args:
            texts: List of input texts
            batch_size: Batch size for processing (default: config.batch_size)

        Returns:
            List of 384-dimensional embedding vectors (one per text)

        Raises:
            ValueError: If texts list is empty

        Example:
            >>> texts = ["한국어", "English", "日本語"]
            >>> vectors = service.embed_texts(texts)
            >>> len(vectors)
            3
        """
        # Validate input
        if not texts:
            raise ValueError("Empty text list cannot be embedded")

        # Use config batch size if not specified
        batch_size = batch_size or self.config.batch_size

        logger.info(f"Embedding {len(texts)} texts with batch_size={batch_size}")

        try:
            # Batch encode with progress bar
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2 normalization
                show_progress_bar=True  # tqdm progress bar
            )

            logger.info(f"Successfully embedded {len(texts)} texts")

            return embeddings.tolist()

        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        """
        Get embedding vector dimension.

        Returns:
            Embedding dimension (384 for paraphrase-multilingual-MiniLM-L12-v2)

        Example:
            >>> service.get_embedding_dimension()
            384
        """
        return self.embedding_dim

    def validate_model(self) -> bool:
        """
        Validate model loading and basic functionality.

        Returns:
            True if model is operational and generates correct embeddings

        Example:
            >>> service.validate_model()
            True
        """
        try:
            # Test with simple Korean text
            test_text = "테스트"
            test_embedding = self.embed_text(test_text)

            # Validate dimension
            is_correct_dim = len(test_embedding) == self.embedding_dim

            # Validate L2 normalization
            magnitude = np.linalg.norm(test_embedding)
            is_normalized = abs(magnitude - 1.0) < 1e-6

            logger.info(
                f"Model validation: dimension_ok={is_correct_dim}, "
                f"normalized={is_normalized}, magnitude={magnitude:.6f}"
            )

            return is_correct_dim and is_normalized

        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False

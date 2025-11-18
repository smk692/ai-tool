"""Mock HuggingFace SentenceTransformer for deterministic testing.

This mock provides consistent, reproducible embeddings without requiring
the actual sentence-transformers library or model downloads.
"""

import hashlib
import numpy as np
from typing import List, Optional, Union


class MockSentenceTransformer:
    """Mock SentenceTransformer for testing without HuggingFace dependency.

    Generates deterministic embeddings based on text hash to ensure
    reproducible test results across runs.

    Attributes:
        model_name: Identifier for the mock model (default: "mock-model")
        dimension: Embedding vector dimension (default: 384, MiniLM-L12 size)
        device: Compute device (cpu/cuda/mps, default: cpu)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cpu"
    ) -> None:
        """Initialize mock embedding model.

        Args:
            model_name: Optional model identifier (default: "mock-model")
            device: Compute device - cpu, cuda, or mps (default: cpu)
        """
        self.model_name = model_name or "mock-model"
        self.dimension = 384  # MiniLM-L12-v2 embedding dimension
        self._device = device

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False
    ) -> np.ndarray:
        """Generate deterministic mock embeddings for input texts.

        Creates embeddings by:
        1. Hashing text to get deterministic seed
        2. Using seed to generate consistent random vector
        3. Optionally normalizing to unit length

        Args:
            texts: Single string or list of strings to embed
            batch_size: Batch processing size (ignored in mock)
            show_progress_bar: Show progress (ignored in mock)
            convert_to_numpy: Return numpy array vs list (default: True)
            normalize_embeddings: Normalize to unit length (default: False)

        Returns:
            np.ndarray: Shape (len(texts), 384), dtype float32

        Raises:
            ValueError: If texts is empty list or contains empty strings
        """
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]

        # Validation
        if not texts:
            raise ValueError("Empty input: texts list cannot be empty")

        if any(not text or not text.strip() for text in texts):
            raise ValueError("Empty input: all texts must be non-empty strings")

        # Generate deterministic embeddings
        embeddings = []
        for text in texts:
            # Use text hash for deterministic seeding
            text_hash = hashlib.md5(text.encode()).hexdigest()
            seed = int(text_hash[:8], 16) % (2**32)

            # Generate consistent embedding
            np.random.seed(seed)
            embedding = np.random.randn(self.dimension).astype(np.float32)

            # Optional normalization
            if normalize_embeddings:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

            embeddings.append(embedding)

        result = np.array(embeddings, dtype=np.float32)

        # Ensure shape is always (n_texts, dimension)
        if result.ndim == 1:
            result = result.reshape(1, -1)

        return result

    @property
    def device(self) -> str:
        """Get current compute device.

        Returns:
            str: Device identifier (cpu, cuda, mps)
        """
        return self._device

    def to(self, device: str) -> "MockSentenceTransformer":
        """Move model to specified device (mock operation).

        Args:
            device: Target device (cpu, cuda, mps)

        Returns:
            self: For method chaining
        """
        self._device = device
        return self

    def get_sentence_embedding_dimension(self) -> int:
        """Get embedding vector dimension.

        Returns:
            int: Embedding dimension (384)
        """
        return self.dimension

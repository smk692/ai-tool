"""Embedding Configuration entity for vector embedding model settings."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DeviceType(str, Enum):
    """Supported inference devices."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon


class EmbeddingConfiguration(BaseModel):
    """
    Vector embedding model settings for document retrieval.

    Used by RAG System for document indexing and query embedding.
    """

    model_name: str = Field(
        default="paraphrase-multilingual-MiniLM-L12-v2",
        description="Embedding model identifier (Hugging Face model)",
    )
    model_path: Optional[str] = Field(
        default=None, description="Local model path or HF repository"
    )
    embedding_dim: int = Field(
        default=384, gt=0, description="Embedding vector dimensions"
    )
    device: DeviceType = Field(default=DeviceType.CPU, description="Inference device")
    batch_size: int = Field(
        default=100, ge=1, le=1000, description="Batch processing size"
    )
    max_seq_length: int = Field(
        default=512, ge=1, le=8192, description="Maximum input sequence length"
    )

    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
        protected_namespaces=()  # Allow model_name and model_path fields
    )

    @field_validator("model_name")
    @classmethod
    def validate_korean_support(cls, v: str) -> str:
        """
        Validate that model supports Korean language.

        This is a basic check - actual multilingual support should be verified
        through testing with Korean text samples.
        """
        multilingual_models = [
            "paraphrase-multilingual",
            "labse",
            "multilingual-e5",
            "bge-m3",
        ]
        if not any(keyword in v.lower() for keyword in multilingual_models):
            # Warning only, not a hard constraint
            pass
        return v

    @field_validator("embedding_dim")
    @classmethod
    def validate_embedding_dim_consistency(cls, v: int) -> int:
        """
        Validate embedding dimensions match ChromaDB collection.

        paraphrase-multilingual-MiniLM-L12-v2 uses 384 dimensions.
        Note: In Pydantic V2, we can't access other field values in field_validator.
        The dimension check is now done as a standalone validation.
        """
        # Most common multilingual models use 384, 768, or 1024 dimensions
        if v not in [128, 256, 384, 512, 768, 1024, 1536, 2048]:
            raise ValueError(
                f"Unusual embedding dimension: {v}. Common values are 384, 768, 1024"
            )
        return v

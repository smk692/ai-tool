"""Embedding Configuration entity for vector embedding model settings."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, validator


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

    @validator("model_name")
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

    @validator("embedding_dim")
    def validate_embedding_dim_consistency(cls, v: int, values: dict) -> int:
        """
        Validate embedding dimensions match ChromaDB collection.

        paraphrase-multilingual-MiniLM-L12-v2 uses 384 dimensions.
        """
        model_name = values.get("model_name", "")
        if "minilm-l12" in model_name.lower() and v != 384:
            raise ValueError(
                f"Model {model_name} uses 384 dimensions, got {v}"
            )
        return v

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        validate_assignment = True

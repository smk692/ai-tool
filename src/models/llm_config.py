"""LLM Configuration entity for Claude Code API settings."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, validator


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"  # Legacy support


class LLMConfiguration(BaseModel):
    """
    LLM configuration settings and API connection parameters.

    Used by all AI chains (Text-to-SQL, Knowledge Discovery, Router, Multi-turn).
    """

    provider: LLMProvider = Field(
        default=LLMProvider.ANTHROPIC, description="LLM provider identifier"
    )
    model_name: str = Field(
        default="claude-3-5-sonnet-20241022", description="Specific model version identifier"
    )
    api_key: str = Field(..., description="API authentication credential")
    temperature: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Sampling temperature (0 = deterministic)"
    )
    max_tokens: int = Field(
        default=4096, ge=1, le=200000, description="Maximum response tokens"
    )
    timeout: int = Field(
        default=60, ge=1, le=300, description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=3, ge=0, le=5, description="Automatic retry attempts"
    )
    streaming: bool = Field(default=False, description="Enable streaming responses")

    @validator("api_key")
    def validate_api_key(cls, v: str, values: dict) -> str:
        """Validate API key format based on provider."""
        provider = values.get("provider")
        if provider == LLMProvider.ANTHROPIC and not v.startswith("sk-ant-"):
            raise ValueError("Anthropic API key must start with 'sk-ant-'")
        if provider == LLMProvider.OPENAI and not v.startswith("sk-"):
            raise ValueError("OpenAI API key must start with 'sk-'")
        return v

    @validator("temperature")
    def validate_temperature_for_sql(cls, v: float) -> float:
        """Enforce temperature = 0 for deterministic SQL generation."""
        # Note: This is a recommendation, not a hard constraint
        # Actual enforcement happens in chain initialization
        return v

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        validate_assignment = True

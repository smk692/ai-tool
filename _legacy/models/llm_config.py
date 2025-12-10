"""LLM Configuration entity for Claude Code API settings."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    ANTHROPIC = "anthropic"


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

    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True
    )

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate API key format for Anthropic Claude."""
        # Note: In Pydantic V2, we can't access other field values in field_validator
        # This validation assumes Anthropic provider (which is the only one we support)
        if not v.startswith("sk-ant-"):
            raise ValueError("Anthropic API key must start with 'sk-ant-'")
        return v

    @field_validator("temperature")
    @classmethod
    def validate_temperature_for_sql(cls, v: float) -> float:
        """Enforce temperature = 0 for deterministic SQL generation."""
        # Note: This is a recommendation, not a hard constraint
        # Actual enforcement happens in chain initialization
        return v

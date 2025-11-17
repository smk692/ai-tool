"""Query Request entity for user queries."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QueryType(str, Enum):
    """Query type classification."""

    TEXT_TO_SQL = "text_to_sql"
    KNOWLEDGE = "knowledge"
    ASSISTANT = "assistant"


class QueryRequest(BaseModel):
    """
    User query submitted to the AI assistant.

    Processed by Router Chain for intent classification.
    """

    query_id: UUID = Field(default_factory=uuid4, description="Unique query identifier")
    user_id: str = Field(..., description="User identifier (Slack user ID)")
    query_text: str = Field(..., min_length=1, max_length=10000, description="User's natural language query")
    query_language: str = Field(default="ko", description="Query language (ISO 639-1 code)")
    query_type: Optional[QueryType] = Field(default=None, description="Query classification (computed)")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Query submission time")
    session_id: Optional[UUID] = Field(default=None, description="Conversation session ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context")

    model_config = ConfigDict(
        use_enum_values=True
    )

    @field_validator("query_text")
    @classmethod
    def validate_query_text(cls, v: str) -> str:
        """Validate query text is not empty or whitespace-only."""
        if not v.strip():
            raise ValueError("Query text cannot be empty or whitespace-only")
        return v.strip()

    @field_validator("query_language")
    @classmethod
    def validate_language_code(cls, v: str) -> str:
        """Validate ISO 639-1 language code."""
        supported_languages = ["ko", "en"]
        if v not in supported_languages:
            raise ValueError(f"Unsupported language: {v}. Supported: {supported_languages}")
        return v

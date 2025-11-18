"""Query Response entity for AI assistant responses."""

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, ValidationInfo


class ResponseType(str, Enum):
    """Response type classification."""

    SQL_QUERY = "sql_query"
    DOCUMENT_ANSWER = "document_answer"
    ASSISTANT_MESSAGE = "assistant_message"
    ERROR = "error"


class TokenUsage(BaseModel):
    """LLM API token usage."""

    input_tokens: int = Field(..., description="Input tokens consumed")
    output_tokens: int = Field(..., description="Output tokens generated")

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens


class SourceDocument(BaseModel):
    """Retrieved source document reference."""

    doc_id: str = Field(..., description="Document identifier")
    title: str = Field(..., description="Document title")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    content: Optional[str] = Field(default=None, description="Document content snippet")


class QueryResponse(BaseModel):
    """
    AI assistant's response to a user query.

    References the original QueryRequest (1:1 relationship).
    """

    response_id: UUID = Field(default_factory=uuid4, description="Unique response identifier")
    query_id: UUID = Field(..., description="Reference to original query")
    response_text: str = Field(..., description="Generated response content")
    response_type: ResponseType = Field(..., description="Response classification")

    # Type-specific fields
    sql_query: Optional[str] = Field(default=None, description="Generated SQL (if type=sql_query)")
    source_documents: List[SourceDocument] = Field(
        default_factory=list,
        description="Retrieved documents (if type=document_answer)"
    )

    # Quality metrics
    confidence_score: Decimal = Field(
        ...,
        ge=Decimal("0.0"),
        le=Decimal("1.0"),
        description="Response confidence"
    )
    execution_time: float = Field(..., gt=0.0, description="Response generation time (seconds)")
    token_usage: TokenUsage = Field(..., description="LLM API token usage")

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Response generation time")
    error_message: Optional[str] = Field(default=None, description="Error details (if type=error)")

    model_config = ConfigDict(
        use_enum_values=True,
        json_encoders={
            UUID: str,
            datetime: lambda v: v.isoformat(),
            Decimal: float,
        }
    )

    @field_validator("response_text", mode="before")
    @classmethod
    def validate_response_text(cls, v: str) -> str:
        """Validate response text is not empty."""
        if not v.strip():
            raise ValueError("Response text cannot be empty")
        return v

    @field_validator("sql_query", mode="before")
    @classmethod
    def validate_sql_query(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        """Validate SQL query is present for sql_query type."""
        response_type = info.data.get("response_type")
        if response_type == ResponseType.SQL_QUERY and not v:
            raise ValueError("SQL query is required for sql_query response type")
        return v

    @field_validator("source_documents", mode="before")
    @classmethod
    def validate_source_documents(cls, v: List[SourceDocument], info: ValidationInfo) -> List[SourceDocument]:
        """Validate source documents are present for document_answer type."""
        response_type = info.data.get("response_type")
        if response_type == ResponseType.DOCUMENT_ANSWER and not v:
            raise ValueError("Source documents are required for document_answer response type")
        return v

    @field_validator("execution_time", mode="before")
    @classmethod
    def validate_execution_time_sla(cls, v: float, info: ValidationInfo) -> float:
        """Log warning if execution time exceeds SLA."""
        response_type = info.data.get("response_type")

        # SLA thresholds
        sla_limits = {
            ResponseType.SQL_QUERY: 60.0,  # Text-to-SQL: 60 seconds
            ResponseType.DOCUMENT_ANSWER: 3.0,  # Knowledge Discovery: 3 seconds
            ResponseType.ASSISTANT_MESSAGE: 10.0,  # General assistant: 10 seconds
        }

        limit = sla_limits.get(response_type, 60.0)
        if v > limit:
            # Warning only, not a hard constraint
            import warnings
            warnings.warn(
                f"Execution time {v:.2f}s exceeds SLA limit {limit}s for {response_type}",
                UserWarning
            )

        return v

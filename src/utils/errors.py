"""Centralized error handling and custom exceptions."""

from typing import Any, Dict, Optional


class AIAssistantError(Exception):
    """Base exception for all AI Assistant errors."""

    def __init__(
        self,
        message: str,
        error_code: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class LLMAPIError(AIAssistantError):
    """Error from LLM API (Claude)."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="LLM_API_ERROR",
            details=details,
        )


class AuthenticationError(AIAssistantError):
    """Invalid API key or authentication failure."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            details=details,
        )


class TimeoutError(AIAssistantError):
    """Request exceeded timeout threshold."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="TIMEOUT_ERROR",
            details=details,
        )


class ValidationError(AIAssistantError):
    """Invalid input or configuration."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=details,
        )


class DatabaseError(AIAssistantError):
    """Database connection or query error."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            details=details,
        )


class VectorStoreError(AIAssistantError):
    """ChromaDB or embedding error."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="VECTOR_STORE_ERROR",
            details=details,
        )


class RateLimitError(AIAssistantError):
    """Rate limit exceeded."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_ERROR",
            details=details,
        )

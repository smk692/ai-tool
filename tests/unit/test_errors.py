"""Unit tests for custom error classes."""

import pytest
from typing import Dict, Any

from src.utils.errors import (
    AIAssistantError,
    LLMAPIError,
    AuthenticationError,
    TimeoutError,
    ValidationError,
    DatabaseError,
    VectorStoreError,
    RateLimitError,
)


class TestAIAssistantError:
    """Test suite for base AIAssistantError class."""

    def test_initialization_with_all_parameters(self):
        """Test AIAssistantError initialization with all parameters."""
        message = "Test error message"
        error_code = "TEST_ERROR"
        details = {"key": "value", "count": 42}

        error = AIAssistantError(
            message=message,
            error_code=error_code,
            details=details,
        )

        assert error.message == message
        assert error.error_code == error_code
        assert error.details == details
        assert str(error) == message

    def test_initialization_without_details(self):
        """Test AIAssistantError initialization without details."""
        message = "Test error without details"
        error_code = "TEST_ERROR"

        error = AIAssistantError(message=message, error_code=error_code)

        assert error.message == message
        assert error.error_code == error_code
        assert error.details == {}

    def test_initialization_with_empty_details(self):
        """Test AIAssistantError initialization with empty details dict."""
        message = "Test error with empty details"
        error_code = "TEST_ERROR"

        error = AIAssistantError(
            message=message,
            error_code=error_code,
            details={},
        )

        assert error.details == {}

    def test_exception_inheritance(self):
        """Test that AIAssistantError inherits from Exception."""
        error = AIAssistantError(
            message="Test",
            error_code="TEST",
        )

        assert isinstance(error, Exception)
        assert isinstance(error, AIAssistantError)

    def test_exception_can_be_raised(self):
        """Test that AIAssistantError can be raised and caught."""
        with pytest.raises(AIAssistantError) as exc_info:
            raise AIAssistantError(
                message="Test raise",
                error_code="TEST_RAISE",
            )

        assert exc_info.value.message == "Test raise"
        assert exc_info.value.error_code == "TEST_RAISE"

    def test_exception_with_special_characters(self):
        """Test exception with special characters in message."""
        message = "Error: 'quotes' and \"double quotes\" with\nnewlines\tand\ttabs"
        error = AIAssistantError(message=message, error_code="SPECIAL_CHARS")

        assert error.message == message
        assert str(error) == message

    def test_exception_with_empty_message(self):
        """Test exception with empty message."""
        error = AIAssistantError(message="", error_code="EMPTY_MESSAGE")

        assert error.message == ""
        assert str(error) == ""


class TestLLMAPIError:
    """Test suite for LLMAPIError class."""

    def test_initialization_with_details(self):
        """Test LLMAPIError initialization with details."""
        message = "LLM API call failed"
        details = {"status_code": 500, "response": "Internal Server Error"}

        error = LLMAPIError(message=message, details=details)

        assert error.message == message
        assert error.error_code == "LLM_API_ERROR"
        assert error.details == details

    def test_initialization_without_details(self):
        """Test LLMAPIError initialization without details."""
        message = "LLM API call failed"

        error = LLMAPIError(message=message)

        assert error.message == message
        assert error.error_code == "LLM_API_ERROR"
        assert error.details == {}

    def test_inheritance_from_base_error(self):
        """Test LLMAPIError inherits from AIAssistantError."""
        error = LLMAPIError(message="Test")

        assert isinstance(error, AIAssistantError)
        assert isinstance(error, LLMAPIError)


class TestAuthenticationError:
    """Test suite for AuthenticationError class."""

    def test_initialization_with_details(self):
        """Test AuthenticationError initialization with details."""
        message = "Invalid API key"
        details = {"api_key": "sk-ant-***", "provider": "anthropic"}

        error = AuthenticationError(message=message, details=details)

        assert error.message == message
        assert error.error_code == "AUTHENTICATION_ERROR"
        assert error.details == details

    def test_initialization_without_details(self):
        """Test AuthenticationError initialization without details."""
        message = "Authentication failed"

        error = AuthenticationError(message=message)

        assert error.message == message
        assert error.error_code == "AUTHENTICATION_ERROR"
        assert error.details == {}

    def test_inheritance_from_base_error(self):
        """Test AuthenticationError inherits from AIAssistantError."""
        error = AuthenticationError(message="Test")

        assert isinstance(error, AIAssistantError)
        assert isinstance(error, AuthenticationError)


class TestTimeoutError:
    """Test suite for TimeoutError class."""

    def test_initialization_with_details(self):
        """Test TimeoutError initialization with details."""
        message = "Request timed out"
        details = {"timeout": 60, "elapsed": 65.3}

        error = TimeoutError(message=message, details=details)

        assert error.message == message
        assert error.error_code == "TIMEOUT_ERROR"
        assert error.details == details

    def test_initialization_without_details(self):
        """Test TimeoutError initialization without details."""
        message = "Timeout occurred"

        error = TimeoutError(message=message)

        assert error.message == message
        assert error.error_code == "TIMEOUT_ERROR"
        assert error.details == {}

    def test_inheritance_from_base_error(self):
        """Test TimeoutError inherits from AIAssistantError."""
        error = TimeoutError(message="Test")

        assert isinstance(error, AIAssistantError)
        assert isinstance(error, TimeoutError)


class TestValidationError:
    """Test suite for ValidationError class."""

    def test_initialization_with_details(self):
        """Test ValidationError initialization with details."""
        message = "Invalid input"
        details = {"field": "email", "value": "invalid_email"}

        error = ValidationError(message=message, details=details)

        assert error.message == message
        assert error.error_code == "VALIDATION_ERROR"
        assert error.details == details

    def test_initialization_without_details(self):
        """Test ValidationError initialization without details."""
        message = "Validation failed"

        error = ValidationError(message=message)

        assert error.message == message
        assert error.error_code == "VALIDATION_ERROR"
        assert error.details == {}

    def test_inheritance_from_base_error(self):
        """Test ValidationError inherits from AIAssistantError."""
        error = ValidationError(message="Test")

        assert isinstance(error, AIAssistantError)
        assert isinstance(error, ValidationError)


class TestDatabaseError:
    """Test suite for DatabaseError class."""

    def test_initialization_with_details(self):
        """Test DatabaseError initialization with details."""
        message = "Database connection failed"
        details = {"host": "localhost", "port": 5432, "error": "Connection refused"}

        error = DatabaseError(message=message, details=details)

        assert error.message == message
        assert error.error_code == "DATABASE_ERROR"
        assert error.details == details

    def test_initialization_without_details(self):
        """Test DatabaseError initialization without details."""
        message = "Database error"

        error = DatabaseError(message=message)

        assert error.message == message
        assert error.error_code == "DATABASE_ERROR"
        assert error.details == {}

    def test_inheritance_from_base_error(self):
        """Test DatabaseError inherits from AIAssistantError."""
        error = DatabaseError(message="Test")

        assert isinstance(error, AIAssistantError)
        assert isinstance(error, DatabaseError)


class TestVectorStoreError:
    """Test suite for VectorStoreError class."""

    def test_initialization_with_details(self):
        """Test VectorStoreError initialization with details."""
        message = "ChromaDB operation failed"
        details = {"operation": "add", "collection": "documents", "count": 100}

        error = VectorStoreError(message=message, details=details)

        assert error.message == message
        assert error.error_code == "VECTOR_STORE_ERROR"
        assert error.details == details

    def test_initialization_without_details(self):
        """Test VectorStoreError initialization without details."""
        message = "Vector store error"

        error = VectorStoreError(message=message)

        assert error.message == message
        assert error.error_code == "VECTOR_STORE_ERROR"
        assert error.details == {}

    def test_inheritance_from_base_error(self):
        """Test VectorStoreError inherits from AIAssistantError."""
        error = VectorStoreError(message="Test")

        assert isinstance(error, AIAssistantError)
        assert isinstance(error, VectorStoreError)


class TestRateLimitError:
    """Test suite for RateLimitError class."""

    def test_initialization_with_details(self):
        """Test RateLimitError initialization with details."""
        message = "Rate limit exceeded"
        details = {"limit": 100, "used": 105, "reset_time": "2024-01-01T00:00:00Z"}

        error = RateLimitError(message=message, details=details)

        assert error.message == message
        assert error.error_code == "RATE_LIMIT_ERROR"
        assert error.details == details

    def test_initialization_without_details(self):
        """Test RateLimitError initialization without details."""
        message = "Rate limit error"

        error = RateLimitError(message=message)

        assert error.message == message
        assert error.error_code == "RATE_LIMIT_ERROR"
        assert error.details == {}

    def test_inheritance_from_base_error(self):
        """Test RateLimitError inherits from AIAssistantError."""
        error = RateLimitError(message="Test")

        assert isinstance(error, AIAssistantError)
        assert isinstance(error, RateLimitError)


class TestExceptionChaining:
    """Test suite for exception chaining and context preservation."""

    def test_exception_chaining_with_from_clause(self):
        """Test exception chaining using 'from' clause."""
        original_error = ValueError("Original error")

        with pytest.raises(LLMAPIError) as exc_info:
            try:
                raise original_error
            except ValueError as e:
                raise LLMAPIError(
                    message="Wrapped error",
                    details={"original": str(e)},
                ) from e

        # Check the raised exception
        assert exc_info.value.message == "Wrapped error"
        assert exc_info.value.details["original"] == "Original error"

        # Check exception chain
        assert exc_info.value.__cause__ is original_error

    def test_exception_context_preservation(self):
        """Test automatic exception context preservation."""
        with pytest.raises(AuthenticationError) as exc_info:
            try:
                raise RuntimeError("Internal error")
            except RuntimeError:
                raise AuthenticationError(message="Auth failed")

        # Check that context is preserved (implicitly)
        assert exc_info.value.__context__ is not None
        assert isinstance(exc_info.value.__context__, RuntimeError)

    def test_multiple_exception_types_in_hierarchy(self):
        """Test catching exceptions at different hierarchy levels."""
        # Test catching specific exception
        with pytest.raises(ValidationError):
            raise ValidationError(message="Validation failed")

        # Test catching base exception
        with pytest.raises(AIAssistantError):
            raise ValidationError(message="Validation failed")

        # Test catching Exception (top level)
        with pytest.raises(Exception):
            raise ValidationError(message="Validation failed")

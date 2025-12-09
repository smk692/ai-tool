"""Unit tests for LLM Client."""

import pytest
from unittest.mock import Mock, patch
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.models.llm_config import LLMConfiguration, LLMProvider
from src.models.query_response import TokenUsage
from src.services.llm_client import LLMClient
from src.utils.errors import AuthenticationError, LLMAPIError, TimeoutError


class TestLLMClient:
    """Test suite for LLMClient class."""

    def test_default_config_creation(self):
        """Test default configuration creation from settings."""
        client = LLMClient()

        assert client.config is not None
        assert client.config.provider == LLMProvider.ANTHROPIC
        assert client.config.model_name == "claude-3-5-sonnet-20241022"
        assert client.config.temperature == 0.0

    def test_custom_config(self):
        """Test initialization with custom configuration."""
        custom_config = LLMConfiguration(
            provider=LLMProvider.ANTHROPIC,
            model_name="claude-3-opus-20240229",
            api_key="sk-ant-test_api_key_12345",
            temperature=0.5,
            max_tokens=2000,
        )

        client = LLMClient(config=custom_config)

        assert client.config == custom_config
        assert client.config.model_name == "claude-3-opus-20240229"
        assert client.config.temperature == 0.5

    @patch("src.services.llm_client.ChatAnthropic")
    def test_client_initialization(self, mock_chat_anthropic):
        """Test lazy initialization of ChatAnthropic client."""
        client = LLMClient()

        # Client should not be initialized until accessed
        assert client._client is None

        # Access client property
        _ = client.client

        # Now client should be initialized
        assert mock_chat_anthropic.called
        assert client._client is not None

    @patch("src.services.llm_client.ChatAnthropic")
    def test_authentication_error(self, mock_chat_anthropic):
        """Test authentication error handling."""
        from anthropic import AuthenticationError as AnthropicAuthError
        from httpx import Response, Request

        # Create mock request and response for Anthropic exception
        mock_request = Request("POST", "https://api.anthropic.com/v1/messages")
        mock_response = Response(status_code=401, request=mock_request)

        mock_chat_anthropic.side_effect = AnthropicAuthError(
            "Invalid API key",
            response=mock_response,
            body={"error": {"message": "Invalid API key"}}
        )

        client = LLMClient()

        with pytest.raises(AuthenticationError) as exc_info:
            _ = client.client

        assert "Invalid Anthropic API key" in str(exc_info.value)

    def test_invoke_success(self):
        """Test successful LLM invocation."""
        # Create mock response
        mock_response = AIMessage(
            content="안녕하세요! 무엇을 도와드릴까요?",
            response_metadata={
                "usage": {
                    "input_tokens": 20,
                    "output_tokens": 15,
                }
            },
        )

        # Patch the client
        with patch.object(LLMClient, "client") as mock_client:
            mock_client.invoke.return_value = mock_response

            client = LLMClient()
            messages = [
                SystemMessage(content="You are helpful"),
                HumanMessage(content="안녕하세요"),
            ]

            result = client.invoke(messages)

            # Verify result structure
            assert "content" in result
            assert "response" in result
            assert "token_usage" in result
            assert "execution_time" in result

            # Verify content
            assert result["content"] == "안녕하세요! 무엇을 도와드릴까요?"

            # Verify token usage
            assert isinstance(result["token_usage"], TokenUsage)
            assert result["token_usage"].input_tokens == 20
            assert result["token_usage"].output_tokens == 15
            assert result["token_usage"].total_tokens == 35

            # Verify execution time
            assert result["execution_time"] > 0

    def test_invoke_timeout_error(self):
        """Test timeout error handling."""
        from anthropic import APIError
        from httpx import Response, Request

        with patch.object(LLMClient, "client") as mock_client:
            # Create mock request for APIError
            mock_request = Request("POST", "https://api.anthropic.com/v1/messages")

            mock_client.invoke.side_effect = APIError(
                "Request timeout",
                request=mock_request,
                body={"error": {"message": "Request timeout"}}
            )

            # Set short timeout
            config = LLMConfiguration(
                api_key="sk-ant-test_api_key_12345",
                timeout=1,
            )
            client = LLMClient(config=config)

            messages = [HumanMessage(content="test")]

            # Should raise timeout error
            with pytest.raises((TimeoutError, LLMAPIError)):
                client.invoke(messages)

    def test_token_usage_extraction(self):
        """Test token usage extraction from response."""
        mock_response = AIMessage(
            content="Test response",
            response_metadata={
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                }
            },
        )

        client = LLMClient()
        token_usage = client._extract_token_usage(mock_response)

        assert isinstance(token_usage, TokenUsage)
        assert token_usage.input_tokens == 100
        assert token_usage.output_tokens == 50
        assert token_usage.total_tokens == 150

    def test_token_usage_extraction_missing_data(self):
        """Test token usage extraction with missing data."""
        mock_response = AIMessage(
            content="Test response",
            response_metadata={},  # No usage data
        )

        client = LLMClient()
        token_usage = client._extract_token_usage(mock_response)

        # Should default to 0
        assert token_usage.input_tokens == 0
        assert token_usage.output_tokens == 0
        assert token_usage.total_tokens == 0

    def test_format_messages_simple(self):
        """Test message formatting without conversation history."""
        client = LLMClient()

        messages = client.format_messages(
            system_prompt="You are helpful",
            user_prompt="안녕하세요",
        )

        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert messages[0].content == "You are helpful"
        assert isinstance(messages[1], HumanMessage)
        assert messages[1].content == "안녕하세요"

    def test_format_messages_with_history(self):
        """Test message formatting with conversation history."""
        client = LLMClient()

        conversation_history = [
            {"role": "user", "content": "이전 질문"},
            {"role": "assistant", "content": "이전 답변"},
        ]

        messages = client.format_messages(
            system_prompt="You are helpful",
            user_prompt="새로운 질문",
            conversation_history=conversation_history,
        )

        assert len(messages) == 4
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
        assert messages[1].content == "이전 질문"
        assert isinstance(messages[2], AIMessage)
        assert messages[2].content == "이전 답변"
        assert isinstance(messages[3], HumanMessage)
        assert messages[3].content == "새로운 질문"

    def test_format_messages_invalid_history_role(self):
        """Test message formatting with invalid history role."""
        client = LLMClient()

        conversation_history = [
            {"role": "invalid", "content": "test"},
        ]

        messages = client.format_messages(
            system_prompt="You are helpful",
            user_prompt="test",
            conversation_history=conversation_history,
        )

        # Should skip invalid role
        assert len(messages) == 2  # Only system and current user message

    @patch.object(LLMClient, "invoke")
    def test_connection_test_success(self, mock_invoke):
        """Test successful connection test."""
        mock_invoke.return_value = {
            "content": "안녕하세요",
            "token_usage": TokenUsage(input_tokens=10, output_tokens=5),
        }

        client = LLMClient()
        result = client.test_connection()

        assert result is True
        assert mock_invoke.called

    @patch.object(LLMClient, "invoke")
    def test_connection_test_failure(self, mock_invoke):
        """Test failed connection test."""
        mock_invoke.side_effect = Exception("Connection failed")

        client = LLMClient()
        result = client.test_connection()

        assert result is False

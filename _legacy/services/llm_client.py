"""Claude API client wrapper for LLM operations."""

import time
from typing import Any, Dict, List, Optional

from anthropic import Anthropic, APIError, AuthenticationError as AnthropicAuthError
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from config.settings import settings
from src.models.llm_config import LLMConfiguration, LLMProvider
from src.models.query_response import TokenUsage
from src.utils.errors import AuthenticationError, LLMAPIError, TimeoutError
from src.utils.logging import logger


class LLMClient:
    """
    Claude API client wrapper for LangChain integration.

    Provides unified interface for LLM operations with error handling,
    retry logic, and token usage tracking.
    """

    def __init__(self, config: Optional[LLMConfiguration] = None):
        """
        Initialize LLM client.

        Args:
            config: LLM configuration (defaults to settings)
        """
        self.config = config or self._default_config()
        self._client: Optional[ChatAnthropic] = None
        self._anthropic_client: Optional[Anthropic] = None

    def _default_config(self) -> LLMConfiguration:
        """Create default configuration from settings."""
        return LLMConfiguration(
            provider=LLMProvider.ANTHROPIC,
            model_name=settings.claude_model,
            api_key=settings.anthropic_api_key or "test_key",
            temperature=settings.claude_temperature,
            max_tokens=settings.claude_max_tokens,
            timeout=settings.claude_timeout,
            max_retries=settings.claude_max_retries,
        )

    @property
    def client(self) -> ChatAnthropic:
        """Get or create ChatAnthropic client."""
        if self._client is None:
            try:
                self._client = ChatAnthropic(
                    model=self.config.model_name,
                    anthropic_api_key=self.config.api_key,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout,
                    max_retries=self.config.max_retries,
                )
                logger.info(f"ChatAnthropic client initialized: {self.config.model_name}")
            except AnthropicAuthError as e:
                logger.error(f"Authentication failed: {e}")
                raise AuthenticationError(
                    message="Invalid Anthropic API key",
                    details={"error": str(e)},
                )
            except Exception as e:
                logger.error(f"Failed to initialize LLM client: {e}")
                raise LLMAPIError(
                    message="Failed to initialize LLM client",
                    details={"error": str(e)},
                )
        return self._client

    @property
    def anthropic_client(self) -> Anthropic:
        """Get or create direct Anthropic client (for advanced features)."""
        if self._anthropic_client is None:
            try:
                self._anthropic_client = Anthropic(
                    api_key=self.config.api_key,
                    timeout=self.config.timeout,
                    max_retries=self.config.max_retries,
                )
                logger.info("Anthropic client initialized")
            except AnthropicAuthError as e:
                logger.error(f"Authentication failed: {e}")
                raise AuthenticationError(
                    message="Invalid Anthropic API key",
                    details={"error": str(e)},
                )
        return self._anthropic_client

    def invoke(
        self,
        messages: List[BaseMessage],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Invoke LLM with messages and track token usage.

        Args:
            messages: List of LangChain messages
            **kwargs: Additional arguments for LLM

        Returns:
            Dictionary with response content and token usage

        Raises:
            LLMAPIError: If LLM invocation fails
            TimeoutError: If request exceeds timeout
        """
        start_time = time.time()

        try:
            # Invoke LLM
            response: AIMessage = self.client.invoke(messages, **kwargs)

            # Calculate execution time
            execution_time = time.time() - start_time

            # Extract token usage from response
            token_usage = self._extract_token_usage(response)

            logger.info(
                f"LLM invocation successful: {token_usage.total_tokens} tokens, "
                f"{execution_time:.2f}s"
            )

            return {
                "content": response.content,
                "response": response,
                "token_usage": token_usage,
                "execution_time": execution_time,
            }

        except APIError as e:
            execution_time = time.time() - start_time
            logger.error(f"LLM API error after {execution_time:.2f}s: {e}")

            # Check if timeout
            if execution_time >= self.config.timeout:
                raise TimeoutError(
                    message=f"LLM request exceeded timeout ({self.config.timeout}s)",
                    details={"execution_time": execution_time, "error": str(e)},
                )

            raise LLMAPIError(
                message="LLM API request failed",
                details={"error": str(e), "execution_time": execution_time},
            )

    def _extract_token_usage(self, response: AIMessage) -> TokenUsage:
        """
        Extract token usage from LLM response.

        Args:
            response: LangChain AIMessage response

        Returns:
            TokenUsage object
        """
        # LangChain response includes usage_metadata
        usage = response.response_metadata.get("usage", {})

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def format_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> List[BaseMessage]:
        """
        Format messages for LLM input.

        Args:
            system_prompt: System instruction
            user_prompt: User query
            conversation_history: Optional conversation history

        Returns:
            List of formatted LangChain messages
        """
        messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]

        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history:
                role = msg.get("role")
                content = msg.get("content", "")

                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))

        # Add current user query
        messages.append(HumanMessage(content=user_prompt))

        return messages

    def test_connection(self) -> bool:
        """
        Test LLM API connection.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            messages = [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="안녕하세요"),
            ]

            result = self.invoke(messages)
            logger.info("✅ LLM connection test passed")
            return True

        except Exception as e:
            logger.error(f"❌ LLM connection test failed: {e}")
            return False

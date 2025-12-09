"""Unit tests for retry utilities."""

import pytest
from unittest.mock import MagicMock, patch
import httpx

from src.utils.retry import (
    RetryConfig,
    create_retry_decorator,
    with_retry,
    notion_retry,
    http_retry,
    qdrant_retry,
    is_retryable_http_error,
    is_retryable_notion_error,
    NOTION_CONFIG,
    HTTP_CONFIG,
    QDRANT_CONFIG,
    NETWORK_EXCEPTIONS,
)


class TestRetryConfig:
    """Tests for RetryConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.min_wait == 1.0
        assert config.max_wait == 60.0
        assert config.exponential_base == 2.0
        assert config.max_delay is None
        assert config.jitter is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = RetryConfig(
            max_attempts=5,
            min_wait=0.5,
            max_wait=30.0,
            exponential_base=1.5,
            max_delay=120.0,
            jitter=False,
        )
        assert config.max_attempts == 5
        assert config.min_wait == 0.5
        assert config.max_wait == 30.0
        assert config.exponential_base == 1.5
        assert config.max_delay == 120.0
        assert config.jitter is False

    def test_to_tenacity_kwargs_with_jitter(self):
        """Test conversion to tenacity kwargs with jitter."""
        config = RetryConfig(
            max_attempts=5,
            min_wait=1.0,
            max_wait=30.0,
            jitter=True,
        )
        kwargs = config.to_tenacity_kwargs()

        assert "stop" in kwargs
        assert "wait" in kwargs

    def test_to_tenacity_kwargs_without_jitter(self):
        """Test conversion to tenacity kwargs without jitter."""
        config = RetryConfig(
            max_attempts=3,
            min_wait=0.5,
            max_wait=10.0,
            jitter=False,
        )
        kwargs = config.to_tenacity_kwargs()

        assert "stop" in kwargs
        assert "wait" in kwargs

    def test_to_tenacity_kwargs_with_max_delay(self):
        """Test conversion to tenacity kwargs with max_delay."""
        config = RetryConfig(
            max_attempts=5,
            max_delay=60.0,
        )
        kwargs = config.to_tenacity_kwargs()

        # Should have combined stop condition
        assert "stop" in kwargs


class TestPreconfiguredConfigs:
    """Tests for pre-configured retry settings."""

    def test_notion_config(self):
        """Test Notion retry configuration."""
        assert NOTION_CONFIG.max_attempts == 5
        assert NOTION_CONFIG.min_wait == 1.0
        assert NOTION_CONFIG.max_wait == 30.0
        assert NOTION_CONFIG.jitter is True

    def test_http_config(self):
        """Test HTTP retry configuration."""
        assert HTTP_CONFIG.max_attempts == 3
        assert HTTP_CONFIG.min_wait == 0.5
        assert HTTP_CONFIG.max_wait == 10.0
        assert HTTP_CONFIG.jitter is True

    def test_qdrant_config(self):
        """Test Qdrant retry configuration."""
        assert QDRANT_CONFIG.max_attempts == 3
        assert QDRANT_CONFIG.min_wait == 0.5
        assert QDRANT_CONFIG.max_wait == 15.0
        assert QDRANT_CONFIG.jitter is True


class TestNetworkExceptions:
    """Tests for NETWORK_EXCEPTIONS tuple."""

    def test_contains_expected_exceptions(self):
        """Test NETWORK_EXCEPTIONS contains expected types."""
        assert httpx.ConnectError in NETWORK_EXCEPTIONS
        assert httpx.ConnectTimeout in NETWORK_EXCEPTIONS
        assert httpx.ReadTimeout in NETWORK_EXCEPTIONS
        assert httpx.WriteTimeout in NETWORK_EXCEPTIONS
        assert httpx.PoolTimeout in NETWORK_EXCEPTIONS
        assert ConnectionError in NETWORK_EXCEPTIONS
        assert TimeoutError in NETWORK_EXCEPTIONS


class TestIsRetryableHttpError:
    """Tests for is_retryable_http_error function."""

    def test_rate_limit_error(self):
        """Test 429 error is retryable."""
        response = MagicMock()
        response.status_code = 429
        error = httpx.HTTPStatusError("Rate limited", request=MagicMock(), response=response)

        assert is_retryable_http_error(error) is True

    def test_server_errors(self):
        """Test 5xx errors are retryable."""
        for status in [500, 502, 503, 504]:
            response = MagicMock()
            response.status_code = status
            error = httpx.HTTPStatusError("Server error", request=MagicMock(), response=response)

            assert is_retryable_http_error(error) is True

    def test_client_errors_not_retryable(self):
        """Test 4xx errors (except 429) are not retryable."""
        for status in [400, 401, 403, 404]:
            response = MagicMock()
            response.status_code = status
            error = httpx.HTTPStatusError("Client error", request=MagicMock(), response=response)

            assert is_retryable_http_error(error) is False

    def test_network_errors_retryable(self):
        """Test network errors are retryable."""
        for exc_class in [httpx.ConnectError, httpx.ReadTimeout, ConnectionError]:
            error = exc_class("Network error")
            assert is_retryable_http_error(error) is True

    def test_other_exceptions_not_retryable(self):
        """Test other exceptions are not retryable."""
        error = ValueError("Some error")
        assert is_retryable_http_error(error) is False


class TestIsRetryableNotionError:
    """Tests for is_retryable_notion_error function."""

    def test_rate_limited_code(self):
        """Test rate_limited error code is retryable."""
        error = MagicMock()
        error.code = "rate_limited"

        assert is_retryable_notion_error(error) is True

    def test_internal_server_error_code(self):
        """Test internal_server_error code is retryable."""
        error = MagicMock()
        error.code = "internal_server_error"

        assert is_retryable_notion_error(error) is True

    def test_service_unavailable_code(self):
        """Test service_unavailable code is retryable."""
        error = MagicMock()
        error.code = "service_unavailable"

        assert is_retryable_notion_error(error) is True

    def test_other_codes_not_retryable(self):
        """Test other error codes are not retryable."""
        error = MagicMock()
        error.code = "validation_error"

        assert is_retryable_notion_error(error) is False

    def test_http_errors_delegated(self):
        """Test HTTP errors are delegated to is_retryable_http_error."""
        response = MagicMock()
        response.status_code = 429
        error = httpx.HTTPStatusError("Rate limited", request=MagicMock(), response=response)

        assert is_retryable_notion_error(error) is True


class TestCreateRetryDecorator:
    """Tests for create_retry_decorator function."""

    def test_creates_decorator(self):
        """Test decorator can be created."""
        config = RetryConfig(max_attempts=2)
        decorator = create_retry_decorator(config)

        assert callable(decorator)

    def test_decorator_wraps_function(self):
        """Test decorator wraps a function."""
        config = RetryConfig(max_attempts=2)
        decorator = create_retry_decorator(config)

        @decorator
        def my_func():
            return "success"

        assert my_func() == "success"

    def test_decorator_retries_on_exception(self):
        """Test decorator retries on specified exception."""
        config = RetryConfig(max_attempts=3, min_wait=0.01, max_wait=0.01)
        call_count = 0

        @create_retry_decorator(config, retry_on=ValueError)
        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Fail")
            return "success"

        result = failing_func()
        assert result == "success"
        assert call_count == 3

    def test_decorator_with_retry_if(self):
        """Test decorator with custom retry_if function."""
        config = RetryConfig(max_attempts=3, min_wait=0.01, max_wait=0.01)

        def should_retry(exc):
            return isinstance(exc, ValueError) and "retry" in str(exc)

        call_count = 0

        @create_retry_decorator(config, retry_if=should_retry)
        def func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("retry this")
            return "done"

        result = func()
        assert result == "done"
        assert call_count == 2


class TestWithRetry:
    """Tests for with_retry decorator."""

    def test_with_retry_no_args(self):
        """Test with_retry without arguments."""
        @with_retry
        def my_func():
            return "result"

        assert my_func() == "result"

    def test_with_retry_with_config(self):
        """Test with_retry with custom config."""
        config = RetryConfig(max_attempts=2)

        @with_retry(config=config)
        def my_func():
            return "result"

        assert my_func() == "result"


class TestNotionRetry:
    """Tests for notion_retry decorator."""

    def test_notion_retry_no_args(self):
        """Test notion_retry without arguments."""
        @notion_retry
        def my_func():
            return "result"

        assert my_func() == "result"

    def test_notion_retry_with_parens(self):
        """Test notion_retry with parentheses."""
        @notion_retry()
        def my_func():
            return "result"

        assert my_func() == "result"


class TestHttpRetry:
    """Tests for http_retry decorator."""

    def test_http_retry_no_args(self):
        """Test http_retry without arguments."""
        @http_retry
        def my_func():
            return "result"

        assert my_func() == "result"

    def test_http_retry_with_parens(self):
        """Test http_retry with parentheses."""
        @http_retry()
        def my_func():
            return "result"

        assert my_func() == "result"


class TestQdrantRetry:
    """Tests for qdrant_retry decorator."""

    def test_qdrant_retry_no_args(self):
        """Test qdrant_retry without arguments."""
        @qdrant_retry
        def my_func():
            return "result"

        assert my_func() == "result"

    def test_qdrant_retry_with_parens(self):
        """Test qdrant_retry with parentheses."""
        @qdrant_retry()
        def my_func():
            return "result"

        assert my_func() == "result"


class TestRetryContext:
    """Tests for RetryContext context manager."""

    def test_init_default_config(self):
        """Test RetryContext initialization with default config."""
        from src.utils.retry import RetryContext, DEFAULT_CONFIG

        ctx = RetryContext()
        assert ctx.config == DEFAULT_CONFIG
        assert ctx.retry_on is None
        assert ctx.retry_if is None
        assert ctx._retrying is None

    def test_init_with_custom_config(self):
        """Test RetryContext initialization with custom config."""
        from src.utils.retry import RetryContext

        config = RetryConfig(max_attempts=5, min_wait=0.5)
        ctx = RetryContext(config=config)
        assert ctx.config == config

    def test_init_with_retry_on(self):
        """Test RetryContext initialization with retry_on."""
        from src.utils.retry import RetryContext

        ctx = RetryContext(retry_on=ValueError)
        assert ctx.retry_on == ValueError
        assert ctx.retry_if is None

    def test_init_with_retry_if(self):
        """Test RetryContext initialization with retry_if callable."""
        from src.utils.retry import RetryContext

        def should_retry(exc):
            return isinstance(exc, ValueError)

        ctx = RetryContext(retry_if=should_retry)
        assert ctx.retry_if == should_retry
        assert ctx.retry_on is None

    def test_sync_context_manager_basic(self):
        """Test RetryContext as sync context manager."""
        from src.utils.retry import RetryContext
        from tenacity import Retrying

        with RetryContext() as retrying:
            assert isinstance(retrying, Retrying)

    def test_sync_context_manager_with_retry_on(self):
        """Test RetryContext with retry_on parameter."""
        from src.utils.retry import RetryContext
        from tenacity import Retrying

        with RetryContext(retry_on=ValueError) as retrying:
            assert isinstance(retrying, Retrying)

    def test_sync_context_manager_retry_on_tuple(self):
        """Test RetryContext with retry_on as tuple of exceptions."""
        from src.utils.retry import RetryContext
        from tenacity import Retrying

        with RetryContext(retry_on=(ValueError, TypeError)) as retrying:
            assert isinstance(retrying, Retrying)

    def test_sync_context_manager_exit_clears_retrying(self):
        """Test that exiting context clears _retrying."""
        from src.utils.retry import RetryContext

        ctx = RetryContext()
        with ctx:
            assert ctx._retrying is not None
        assert ctx._retrying is None

    @pytest.mark.asyncio
    async def test_async_context_manager_basic(self):
        """Test RetryContext as async context manager."""
        from src.utils.retry import RetryContext
        from tenacity import AsyncRetrying

        async with RetryContext() as retrying:
            assert isinstance(retrying, AsyncRetrying)

    @pytest.mark.asyncio
    async def test_async_context_manager_with_retry_on(self):
        """Test async RetryContext with retry_on parameter."""
        from src.utils.retry import RetryContext
        from tenacity import AsyncRetrying

        async with RetryContext(retry_on=ValueError) as retrying:
            assert isinstance(retrying, AsyncRetrying)

    @pytest.mark.asyncio
    async def test_async_context_manager_retry_on_tuple(self):
        """Test async RetryContext with retry_on as tuple of exceptions."""
        from src.utils.retry import RetryContext
        from tenacity import AsyncRetrying

        async with RetryContext(retry_on=(ValueError, TypeError)) as retrying:
            assert isinstance(retrying, AsyncRetrying)

    @pytest.mark.asyncio
    async def test_async_context_manager_exit_clears_retrying(self):
        """Test that exiting async context clears _retrying."""
        from src.utils.retry import RetryContext

        ctx = RetryContext()
        async with ctx:
            assert ctx._retrying is not None
        assert ctx._retrying is None

    def test_exit_returns_false(self):
        """Test that __exit__ returns False (doesn't suppress exceptions)."""
        from src.utils.retry import RetryContext

        ctx = RetryContext()
        ctx.__enter__()
        result = ctx.__exit__(None, None, None)
        assert result is False

    @pytest.mark.asyncio
    async def test_aexit_returns_false(self):
        """Test that __aexit__ returns False (doesn't suppress exceptions)."""
        from src.utils.retry import RetryContext

        ctx = RetryContext()
        await ctx.__aenter__()
        result = await ctx.__aexit__(None, None, None)
        assert result is False

    def test_sync_context_manager_with_retry_if_callable(self):
        """Test RetryContext with retry_if callable function."""
        from src.utils.retry import RetryContext
        from tenacity import Retrying

        def should_retry(exc):
            return isinstance(exc, ValueError)

        with RetryContext(retry_if=should_retry) as retrying:
            assert isinstance(retrying, Retrying)

    @pytest.mark.asyncio
    async def test_async_context_manager_with_retry_if_callable(self):
        """Test async RetryContext with retry_if callable function."""
        from src.utils.retry import RetryContext
        from tenacity import AsyncRetrying

        def should_retry(exc):
            return isinstance(exc, ValueError)

        async with RetryContext(retry_if=should_retry) as retrying:
            assert isinstance(retrying, AsyncRetrying)
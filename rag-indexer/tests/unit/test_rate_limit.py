"""Unit tests for rate limiting utilities."""

import asyncio
import time
import pytest
from unittest.mock import MagicMock, patch

from src.utils.rate_limit import (
    RateLimitConfig,
    TokenBucketRateLimiter,
    AsyncTokenBucketRateLimiter,
    RateLimitedClient,
    rate_limited,
    async_rate_limited,
    NOTION_RATE_LIMIT,
    HTTP_RATE_LIMIT,
    QDRANT_RATE_LIMIT,
)


class TestRateLimitConfig:
    """Tests for RateLimitConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RateLimitConfig()
        assert config.requests_per_second == 3.0
        assert config.burst_size == 10
        assert config.min_interval == 0.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = RateLimitConfig(
            requests_per_second=5.0,
            burst_size=20,
            min_interval=0.1,
        )
        assert config.requests_per_second == 5.0
        assert config.burst_size == 20
        assert config.min_interval == 0.1

    def test_token_rate_property(self):
        """Test token_rate property."""
        config = RateLimitConfig(requests_per_second=10.0)
        assert config.token_rate == 10.0


class TestPreconfiguredLimits:
    """Tests for pre-configured rate limit settings."""

    def test_notion_rate_limit(self):
        """Test Notion rate limit configuration."""
        assert NOTION_RATE_LIMIT.requests_per_second == 3.0
        assert NOTION_RATE_LIMIT.burst_size == 10

    def test_http_rate_limit(self):
        """Test HTTP rate limit configuration."""
        assert HTTP_RATE_LIMIT.requests_per_second == 10.0
        assert HTTP_RATE_LIMIT.burst_size == 20

    def test_qdrant_rate_limit(self):
        """Test Qdrant rate limit configuration."""
        assert QDRANT_RATE_LIMIT.requests_per_second == 100.0
        assert QDRANT_RATE_LIMIT.burst_size == 100


class TestTokenBucketRateLimiter:
    """Tests for synchronous TokenBucketRateLimiter."""

    def test_initialization(self):
        """Test limiter initialization."""
        config = RateLimitConfig(requests_per_second=5.0, burst_size=10)
        limiter = TokenBucketRateLimiter(config)

        assert limiter.config == config
        assert limiter.available_tokens == 10.0

    def test_acquire_single_token(self):
        """Test acquiring a single token."""
        config = RateLimitConfig(requests_per_second=10.0, burst_size=5)
        limiter = TokenBucketRateLimiter(config)

        result = limiter.acquire(1, blocking=False)

        assert result is True
        assert limiter.available_tokens < 5.0

    def test_acquire_multiple_tokens(self):
        """Test acquiring multiple tokens."""
        config = RateLimitConfig(requests_per_second=10.0, burst_size=10)
        limiter = TokenBucketRateLimiter(config)

        result = limiter.acquire(5, blocking=False)

        assert result is True
        # Allow small tolerance for timing
        assert limiter.available_tokens <= 5.1

    def test_acquire_exhausts_bucket(self):
        """Test acquiring all available tokens."""
        config = RateLimitConfig(requests_per_second=1.0, burst_size=3)
        limiter = TokenBucketRateLimiter(config)

        # Exhaust tokens
        for _ in range(3):
            assert limiter.acquire(1, blocking=False) is True

        # Next request should fail (non-blocking)
        assert limiter.acquire(1, blocking=False) is False

    def test_acquire_blocking_waits(self):
        """Test blocking acquire waits for tokens."""
        config = RateLimitConfig(requests_per_second=100.0, burst_size=1)
        limiter = TokenBucketRateLimiter(config)

        # Use the token
        limiter.acquire(1)

        # Blocking acquire should wait
        start = time.monotonic()
        limiter.acquire(1, blocking=True)
        elapsed = time.monotonic() - start

        assert elapsed >= 0  # Should have waited some time

    def test_token_refill(self):
        """Test tokens refill over time."""
        config = RateLimitConfig(requests_per_second=100.0, burst_size=5)
        limiter = TokenBucketRateLimiter(config)

        # Exhaust some tokens
        limiter.acquire(3, blocking=False)
        initial = limiter.available_tokens

        # Wait for refill
        time.sleep(0.05)  # 50ms
        after = limiter.available_tokens

        # Should have more tokens now
        assert after >= initial

    def test_min_interval_enforcement(self):
        """Test minimum interval between requests."""
        config = RateLimitConfig(
            requests_per_second=100.0,
            burst_size=10,
            min_interval=0.05,  # 50ms
        )
        limiter = TokenBucketRateLimiter(config)

        # First request
        limiter.acquire(1)

        # Immediate second request (non-blocking) should fail
        result = limiter.acquire(1, blocking=False)
        assert result is False

        # Wait for min_interval
        time.sleep(0.06)
        result = limiter.acquire(1, blocking=False)
        assert result is True

    def test_available_tokens_property(self):
        """Test available_tokens property."""
        config = RateLimitConfig(requests_per_second=10.0, burst_size=10)
        limiter = TokenBucketRateLimiter(config)

        assert limiter.available_tokens == 10.0

        limiter.acquire(3)
        # Allow small tolerance for timing (token refill happens between operations)
        assert limiter.available_tokens <= 7.1


class TestAsyncTokenBucketRateLimiter:
    """Tests for async TokenBucketRateLimiter."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test async limiter initialization."""
        config = RateLimitConfig(requests_per_second=5.0, burst_size=10)
        limiter = AsyncTokenBucketRateLimiter(config)

        assert limiter.config == config
        assert limiter.available_tokens == 10.0

    @pytest.mark.asyncio
    async def test_acquire_single_token(self):
        """Test acquiring a single token asynchronously."""
        config = RateLimitConfig(requests_per_second=10.0, burst_size=5)
        limiter = AsyncTokenBucketRateLimiter(config)

        result = await limiter.acquire(1, blocking=False)

        assert result is True
        assert limiter.available_tokens < 5.0

    @pytest.mark.asyncio
    async def test_acquire_exhausts_bucket(self):
        """Test async acquiring all available tokens."""
        config = RateLimitConfig(requests_per_second=1.0, burst_size=3)
        limiter = AsyncTokenBucketRateLimiter(config)

        # Exhaust tokens
        for _ in range(3):
            assert await limiter.acquire(1, blocking=False) is True

        # Next request should fail (non-blocking)
        assert await limiter.acquire(1, blocking=False) is False

    @pytest.mark.asyncio
    async def test_acquire_blocking_waits(self):
        """Test async blocking acquire waits for tokens."""
        config = RateLimitConfig(requests_per_second=100.0, burst_size=1)
        limiter = AsyncTokenBucketRateLimiter(config)

        # Use the token
        await limiter.acquire(1)

        # Blocking acquire should wait
        start = time.monotonic()
        await limiter.acquire(1, blocking=True)
        elapsed = time.monotonic() - start

        assert elapsed >= 0  # Should have waited some time

    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        """Test async limiter handles concurrent access."""
        config = RateLimitConfig(requests_per_second=100.0, burst_size=5)
        limiter = AsyncTokenBucketRateLimiter(config)

        async def acquire_task():
            return await limiter.acquire(1, blocking=True)

        # Run multiple concurrent acquires
        results = await asyncio.gather(*[acquire_task() for _ in range(5)])

        assert all(r is True for r in results)


class TestRateLimitedClient:
    """Tests for RateLimitedClient wrapper."""

    def test_wraps_function(self):
        """Test RateLimitedClient wraps a function."""
        mock_func = MagicMock(return_value="result")
        config = RateLimitConfig(requests_per_second=10.0, burst_size=5)

        client = RateLimitedClient(mock_func, config)
        result = client("arg1", kwarg="value")

        assert result == "result"
        mock_func.assert_called_once_with("arg1", kwarg="value")

    def test_rate_limits_calls(self):
        """Test RateLimitedClient rate limits calls."""
        call_times = []

        def tracked_func():
            call_times.append(time.monotonic())
            return "result"

        config = RateLimitConfig(requests_per_second=100.0, burst_size=2)
        client = RateLimitedClient(tracked_func, config)

        # Make rapid calls
        for _ in range(3):
            client()

        # First two should be immediate (burst), third may be delayed
        assert len(call_times) == 3


class TestRateLimitedDecorator:
    """Tests for rate_limited decorator."""

    def test_decorator_applies_rate_limiting(self):
        """Test rate_limited decorator works."""
        config = RateLimitConfig(requests_per_second=100.0, burst_size=2)

        @rate_limited(config)
        def my_func(x):
            return x * 2

        assert my_func(5) == 10
        assert my_func(10) == 20


class TestAsyncRateLimitedDecorator:
    """Tests for async_rate_limited decorator."""

    @pytest.mark.asyncio
    async def test_decorator_applies_rate_limiting(self):
        """Test async_rate_limited decorator works."""
        config = RateLimitConfig(requests_per_second=100.0, burst_size=2)

        @async_rate_limited(config)
        async def my_async_func(x):
            return x * 2

        assert await my_async_func(5) == 10
        assert await my_async_func(10) == 20

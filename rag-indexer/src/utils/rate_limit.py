"""Rate limiting 유틸리티.

API 호출에 대한 토큰 버킷 Rate Limiter를 제공합니다.

주요 기능:
    - 토큰 버킷 알고리즘 기반 Rate Limiting
    - 버스트 트래픽 허용 (burst_size까지)
    - 동기/비동기 모두 지원
    - 데코레이터 및 클라이언트 래퍼 제공
"""

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RateLimitConfig:
    """Rate limiting 설정.

    토큰 버킷 알고리즘의 파라미터를 정의합니다.

    Attributes:
        requests_per_second: 초당 허용 요청 수.
        burst_size: 버스트 허용 크기 (최대 토큰 수).
        min_interval: 요청 간 최소 간격 (초).
    """

    requests_per_second: float = 3.0
    burst_size: int = 10
    min_interval: float = 0.0

    @property
    def token_rate(self) -> float:
        """초당 추가되는 토큰 수.

        Returns:
            초당 토큰 보충률.
        """
        return self.requests_per_second


class TokenBucketRateLimiter:
    """스레드 안전한 토큰 버킷 Rate Limiter.

    burst_size까지 버스트 트래픽을 허용하고,
    이후에는 requests_per_second로 제한합니다.

    토큰 버킷 알고리즘 동작:
        1. 버킷은 최대 burst_size개의 토큰을 보유
        2. 매초 requests_per_second개의 토큰이 추가됨
        3. 요청 시 토큰 1개를 소비
        4. 토큰이 없으면 대기

    Attributes:
        config: Rate limit 설정.
    """

    def __init__(self, config: RateLimitConfig):
        """Rate Limiter를 초기화합니다.

        Args:
            config: Rate limit 설정.
        """
        self.config = config
        self._tokens = float(config.burst_size)
        self._last_update = time.monotonic()
        self._last_request = 0.0
        self._lock = threading.Lock()

    def acquire(self, tokens: int = 1, blocking: bool = True) -> bool:
        """버킷에서 토큰을 획득합니다.

        토큰이 없으면 blocking=True일 때 사용 가능할 때까지 대기합니다.

        Args:
            tokens: 획득할 토큰 수.
            blocking: True면 토큰 사용 가능할 때까지 대기.

        Returns:
            토큰 획득 성공 시 True.
            blocking=False이고 토큰이 없으면 False.
        """
        with self._lock:
            while True:
                self._refill()

                # 최소 간격 확인
                if self.config.min_interval > 0:
                    elapsed = time.monotonic() - self._last_request
                    if elapsed < self.config.min_interval:
                        if blocking:
                            wait_time = self.config.min_interval - elapsed
                            self._lock.release()
                            time.sleep(wait_time)
                            self._lock.acquire()
                            continue
                        return False

                # 토큰 가용성 확인
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    self._last_request = time.monotonic()
                    return True

                if not blocking:
                    return False

                # 대기 시간 계산
                tokens_needed = tokens - self._tokens
                wait_time = tokens_needed / self.config.token_rate
                self._lock.release()
                time.sleep(wait_time)
                self._lock.acquire()

    def _refill(self) -> None:
        """경과 시간에 따라 토큰을 보충합니다.

        마지막 업데이트 이후 경과 시간만큼 토큰을 추가합니다.
        단, burst_size를 초과하지 않습니다.
        """
        now = time.monotonic()
        elapsed = now - self._last_update
        self._tokens = min(
            self.config.burst_size,
            self._tokens + elapsed * self.config.token_rate,
        )
        self._last_update = now

    @property
    def available_tokens(self) -> float:
        """현재 사용 가능한 토큰 수를 반환합니다.

        Returns:
            사용 가능한 토큰 수.
        """
        with self._lock:
            self._refill()
            return self._tokens


class AsyncTokenBucketRateLimiter:
    """비동기 토큰 버킷 Rate Limiter.

    burst_size까지 버스트 트래픽을 허용하고,
    이후에는 requests_per_second로 제한합니다.

    asyncio와 함께 사용할 수 있는 비동기 버전입니다.

    Attributes:
        config: Rate limit 설정.
    """

    def __init__(self, config: RateLimitConfig):
        """비동기 Rate Limiter를 초기화합니다.

        Args:
            config: Rate limit 설정.
        """
        self.config = config
        self._tokens = float(config.burst_size)
        self._last_update = time.monotonic()
        self._last_request = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1, blocking: bool = True) -> bool:
        """버킷에서 토큰을 획득합니다 (비동기).

        토큰이 없으면 blocking=True일 때 사용 가능할 때까지 대기합니다.

        Args:
            tokens: 획득할 토큰 수.
            blocking: True면 토큰 사용 가능할 때까지 대기.

        Returns:
            토큰 획득 성공 시 True.
            blocking=False이고 토큰이 없으면 False.
        """
        async with self._lock:
            while True:
                self._refill()

                # 최소 간격 확인
                if self.config.min_interval > 0:
                    elapsed = time.monotonic() - self._last_request
                    if elapsed < self.config.min_interval:
                        if blocking:
                            wait_time = self.config.min_interval - elapsed
                            await asyncio.sleep(wait_time)
                            continue
                        return False

                # 토큰 가용성 확인
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    self._last_request = time.monotonic()
                    return True

                if not blocking:
                    return False

                # 대기 시간 계산
                tokens_needed = tokens - self._tokens
                wait_time = tokens_needed / self.config.token_rate
                await asyncio.sleep(wait_time)

    def _refill(self) -> None:
        """경과 시간에 따라 토큰을 보충합니다.

        마지막 업데이트 이후 경과 시간만큼 토큰을 추가합니다.
        단, burst_size를 초과하지 않습니다.
        """
        now = time.monotonic()
        elapsed = now - self._last_update
        self._tokens = min(
            self.config.burst_size,
            self._tokens + elapsed * self.config.token_rate,
        )
        self._last_update = now

    @property
    def available_tokens(self) -> float:
        """현재 사용 가능한 토큰 수를 반환합니다.

        주의: 이 속성은 비동기적으로 안전하지 않습니다.
        정확한 값이 필요하면 acquire()를 사용하세요.

        Returns:
            사용 가능한 토큰 수 (근사값).
        """
        self._refill()
        return self._tokens


# 일반적인 API를 위한 사전 구성된 Rate Limiter
NOTION_RATE_LIMIT = RateLimitConfig(
    requests_per_second=3.0,  # Notion 제한은 3 req/s
    burst_size=10,
    min_interval=0.0,
)

HTTP_RATE_LIMIT = RateLimitConfig(
    requests_per_second=10.0,
    burst_size=20,
    min_interval=0.0,
)

QDRANT_RATE_LIMIT = RateLimitConfig(
    requests_per_second=100.0,  # Qdrant는 상당히 빠름
    burst_size=100,
    min_interval=0.0,
)


class RateLimitedClient:
    """모든 콜러블에 Rate limiting을 추가하는 래퍼.

    기존 함수를 감싸서 Rate limiting을 적용합니다.

    Attributes:
        _func: 래핑된 함수.
        _limiter: 토큰 버킷 Rate Limiter.

    Example:
        >>> limiter = RateLimitedClient(api.call, NOTION_RATE_LIMIT)
        >>> result = limiter("arg1", kwarg="value")
    """

    def __init__(
        self,
        func,
        config: RateLimitConfig,
    ):
        """Rate limited 클라이언트를 초기화합니다.

        Args:
            func: 래핑할 함수.
            config: Rate limit 설정.
        """
        self._func = func
        self._limiter = TokenBucketRateLimiter(config)

    def __call__(self, *args, **kwargs):
        """Rate limiting이 적용된 래핑된 함수를 호출합니다.

        토큰을 획득한 후 원본 함수를 호출합니다.

        Args:
            *args: 함수에 전달할 위치 인자.
            **kwargs: 함수에 전달할 키워드 인자.

        Returns:
            래핑된 함수의 반환값.
        """
        self._limiter.acquire()
        return self._func(*args, **kwargs)


def rate_limited(config: RateLimitConfig):
    """함수에 Rate limiting을 추가하는 데코레이터.

    동기 함수용 데코레이터입니다.

    Args:
        config: Rate limit 설정.

    Returns:
        데코레이트된 함수.

    Example:
        >>> @rate_limited(NOTION_RATE_LIMIT)
        ... def api_call():
        ...     pass
    """
    limiter = TokenBucketRateLimiter(config)

    def decorator(func):
        def wrapper(*args, **kwargs):
            limiter.acquire()
            return func(*args, **kwargs)
        return wrapper

    return decorator


def async_rate_limited(config: RateLimitConfig):
    """비동기 함수에 Rate limiting을 추가하는 데코레이터.

    비동기 함수용 데코레이터입니다.

    Args:
        config: Rate limit 설정.

    Returns:
        데코레이트된 비동기 함수.

    Example:
        >>> @async_rate_limited(NOTION_RATE_LIMIT)
        ... async def api_call():
        ...     pass
    """
    limiter = AsyncTokenBucketRateLimiter(config)

    def decorator(func):
        async def wrapper(*args, **kwargs):
            await limiter.acquire()
            return await func(*args, **kwargs)
        return wrapper

    return decorator

"""RAG Indexer 유틸리티 모듈.

API 호출 및 외부 서비스 통신에 필요한 유틸리티를 제공합니다.

유틸리티 구성:
    Rate Limiting:
        - RateLimitConfig: Rate limit 설정
        - TokenBucketRateLimiter: 동기 토큰 버킷 Rate Limiter
        - AsyncTokenBucketRateLimiter: 비동기 토큰 버킷 Rate Limiter
        - RateLimitedClient: Rate limiting이 적용된 클라이언트 래퍼
        - rate_limited: 동기 함수용 Rate limit 데코레이터
        - async_rate_limited: 비동기 함수용 Rate limit 데코레이터
        - NOTION_RATE_LIMIT: Notion API용 기본 설정 (3 req/s)
        - HTTP_RATE_LIMIT: 일반 HTTP용 기본 설정 (10 req/s)
        - QDRANT_RATE_LIMIT: Qdrant용 기본 설정 (100 req/s)

    Retry:
        - RetryConfig: 재시도 설정
        - create_retry_decorator: 커스텀 재시도 데코레이터 생성
        - with_retry: 재시도 데코레이터
        - notion_retry: Notion API 전용 재시도 데코레이터
        - http_retry: HTTP 호출 전용 재시도 데코레이터
        - qdrant_retry: Qdrant 전용 재시도 데코레이터
"""

from .rate_limit import (
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
from .retry import (
    RetryConfig,
    create_retry_decorator,
    with_retry,
    notion_retry,
    http_retry,
    qdrant_retry,
)

__all__ = [
    # Rate Limiting (속도 제한)
    "RateLimitConfig",
    "TokenBucketRateLimiter",
    "AsyncTokenBucketRateLimiter",
    "RateLimitedClient",
    "rate_limited",
    "async_rate_limited",
    "NOTION_RATE_LIMIT",
    "HTTP_RATE_LIMIT",
    "QDRANT_RATE_LIMIT",
    # Retry (재시도)
    "RetryConfig",
    "create_retry_decorator",
    "with_retry",
    "notion_retry",
    "http_retry",
    "qdrant_retry",
]

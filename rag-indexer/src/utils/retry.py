"""재시도 유틸리티 (tenacity 기반).

API 호출 및 외부 서비스에 대한 설정 가능한 재시도 데코레이터를 제공합니다.

주요 기능:
    - 지수 백오프를 사용한 자동 재시도
    - 서비스별 사전 설정된 재시도 구성
    - 조건부 재시도 (특정 예외/상태 코드만)
    - 재시도 시도 로깅
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Type, Union

import httpx
import structlog
from tenacity import (
    AsyncRetrying,
    RetryError,
    Retrying,
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
    wait_random_exponential,
    before_sleep_log,
    after_log,
)

logger = structlog.get_logger(__name__)


@dataclass
class RetryConfig:
    """재시도 동작 설정.

    지수 백오프와 지터(jitter)를 사용한 재시도 전략을 정의합니다.

    Attributes:
        max_attempts: 최대 재시도 횟수.
        min_wait: 최소 대기 시간 (초).
        max_wait: 최대 대기 시간 (초).
        exponential_base: 지수 백오프 베이스.
        max_delay: 전체 최대 지연 시간 (초).
            설정 시 max_attempts와 함께 적용됩니다.
        jitter: 무작위 지터 사용 여부.
            True면 wait_random_exponential 사용.
    """

    max_attempts: int = 3
    min_wait: float = 1.0
    max_wait: float = 60.0
    exponential_base: float = 2.0
    max_delay: Optional[float] = None
    jitter: bool = True

    def to_tenacity_kwargs(self) -> dict[str, Any]:
        """tenacity 재시도 kwargs로 변환합니다.

        설정을 tenacity 라이브러리가 이해하는
        키워드 인자로 변환합니다.

        Returns:
            tenacity.retry에 전달할 kwargs 딕셔너리.
        """
        kwargs: dict[str, Any] = {
            "stop": stop_after_attempt(self.max_attempts),
        }

        # max_delay가 설정되면 시간 제한도 추가
        if self.max_delay:
            kwargs["stop"] = kwargs["stop"] | stop_after_delay(self.max_delay)

        # 지터 사용 여부에 따라 대기 전략 선택
        if self.jitter:
            kwargs["wait"] = wait_random_exponential(
                multiplier=self.min_wait,
                max=self.max_wait,
            )
        else:
            kwargs["wait"] = wait_exponential(
                multiplier=self.min_wait,
                max=self.max_wait,
                exp_base=self.exponential_base,
            )

        return kwargs


# 일반적인 사용 사례를 위한 사전 구성된 재시도 설정
DEFAULT_CONFIG = RetryConfig()

NOTION_CONFIG = RetryConfig(
    max_attempts=5,
    min_wait=1.0,
    max_wait=30.0,
    jitter=True,
)

HTTP_CONFIG = RetryConfig(
    max_attempts=3,
    min_wait=0.5,
    max_wait=10.0,
    jitter=True,
)

QDRANT_CONFIG = RetryConfig(
    max_attempts=3,
    min_wait=0.5,
    max_wait=15.0,
    jitter=True,
)


# 재시도 가능한 일반적인 예외들
NETWORK_EXCEPTIONS: tuple[Type[Exception], ...] = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    ConnectionError,
    TimeoutError,
)

HTTP_RETRYABLE_EXCEPTIONS: tuple[Type[Exception], ...] = (
    httpx.HTTPStatusError,  # retry_if 콜백에서 필터링됨
    *NETWORK_EXCEPTIONS,
)


def is_retryable_http_error(exception: BaseException) -> bool:
    """HTTP 오류가 재시도 가능한지 확인합니다.

    다음 경우에 재시도합니다:
        - 429 (Rate Limited)
        - 500, 502, 503, 504 (서버 오류)
        - 네트워크/타임아웃 오류

    Args:
        exception: 확인할 예외.

    Returns:
        재시도해야 하면 True.
    """
    if isinstance(exception, httpx.HTTPStatusError):
        status = exception.response.status_code
        return status == 429 or status >= 500
    return isinstance(exception, NETWORK_EXCEPTIONS)


def is_retryable_notion_error(exception: BaseException) -> bool:
    """Notion API 오류가 재시도 가능한지 확인합니다.

    Rate limit과 서버 오류에 대한 Notion 특화 재시도 로직입니다.

    Args:
        exception: 확인할 예외.

    Returns:
        재시도해야 하면 True.
    """
    # Notion 클라이언트 오류 처리 (notion_client.errors.APIResponseError)
    error_code = getattr(exception, "code", None)
    if error_code:
        return error_code in ("rate_limited", "internal_server_error", "service_unavailable")

    # httpx 오류 처리
    return is_retryable_http_error(exception)


def create_retry_decorator(
    config: RetryConfig = DEFAULT_CONFIG,
    retry_on: Optional[Union[Type[Exception], tuple[Type[Exception], ...]]] = None,
    retry_if: Optional[Callable[[BaseException], bool]] = None,
    log_retries: bool = True,
) -> Callable:
    """커스텀 설정으로 재시도 데코레이터를 생성합니다.

    Args:
        config: 재시도 파라미터가 담긴 RetryConfig 인스턴스.
        retry_on: 재시도할 예외 타입.
        retry_if: 예외 재시도 여부를 결정하는 콜러블.
        log_retries: 재시도 시도를 로깅할지 여부.

    Returns:
        설정된 재시도 데코레이터.

    Example:
        >>> decorator = create_retry_decorator(
        ...     config=NOTION_CONFIG,
        ...     retry_if=is_retryable_notion_error,
        ... )
        >>> @decorator
        ... def my_api_call():
        ...     pass
    """
    kwargs = config.to_tenacity_kwargs()

    # 재시도 조건 설정
    if retry_if:
        kwargs["retry"] = retry_if_exception(retry_if)
    elif retry_on:
        kwargs["retry"] = retry_if_exception_type(retry_on)

    # 로깅 추가
    if log_retries:
        kwargs["before_sleep"] = before_sleep_log(logger, log_level=20)  # INFO
        kwargs["after"] = after_log(logger, log_level=20)

    return retry(**kwargs)


def with_retry(
    func: Optional[Callable] = None,
    *,
    config: RetryConfig = DEFAULT_CONFIG,
    retry_on: Optional[Union[Type[Exception], tuple[Type[Exception], ...]]] = None,
    retry_if: Optional[Callable[[BaseException], bool]] = None,
    log_retries: bool = True,
) -> Callable:
    """함수에 재시도 로직을 추가하는 데코레이터.

    괄호 있이/없이 모두 사용 가능합니다.

    Args:
        func: 래핑할 함수 (괄호 없이 사용할 때).
        config: RetryConfig 인스턴스.
        retry_on: 재시도할 예외 타입.
        retry_if: 예외 재시도 여부를 결정하는 콜러블.
        log_retries: 재시도 시도를 로깅할지 여부.

    Returns:
        데코레이트된 함수 또는 데코레이터.

    Examples:
        괄호 없이 사용:
        >>> @with_retry
        ... def my_func():
        ...     pass

        괄호와 함께 사용:
        >>> @with_retry(config=NOTION_CONFIG)
        ... def my_func():
        ...     pass
    """
    decorator = create_retry_decorator(
        config=config,
        retry_on=retry_on,
        retry_if=retry_if,
        log_retries=log_retries,
    )

    if func is not None:
        return decorator(func)
    return decorator


# 일반적인 사용 사례를 위한 사전 구성된 데코레이터
def notion_retry(func: Optional[Callable] = None) -> Callable:
    """Notion API 호출용 재시도 데코레이터.

    Rate limit과 서버 오류에 대해 적절한 백오프로 재시도합니다.

    Args:
        func: 데코레이트할 함수 (선택적).

    Returns:
        데코레이트된 함수 또는 데코레이터.

    Example:
        >>> @notion_retry
        ... def fetch_page(page_id):
        ...     return notion.pages.retrieve(page_id)
    """
    decorator = create_retry_decorator(
        config=NOTION_CONFIG,
        retry_if=is_retryable_notion_error,
    )

    if func is not None:
        return decorator(func)
    return decorator


def http_retry(func: Optional[Callable] = None) -> Callable:
    """일반 HTTP 호출용 재시도 데코레이터.

    네트워크 오류와 5xx/429 상태 코드에 대해 재시도합니다.

    Args:
        func: 데코레이트할 함수 (선택적).

    Returns:
        데코레이트된 함수 또는 데코레이터.

    Example:
        >>> @http_retry
        ... def fetch_data(url):
        ...     return httpx.get(url)
    """
    decorator = create_retry_decorator(
        config=HTTP_CONFIG,
        retry_if=is_retryable_http_error,
    )

    if func is not None:
        return decorator(func)
    return decorator


def qdrant_retry(func: Optional[Callable] = None) -> Callable:
    """Qdrant 작업용 재시도 데코레이터.

    네트워크 오류와 연결 문제에 대해 재시도합니다.

    Args:
        func: 데코레이트할 함수 (선택적).

    Returns:
        데코레이트된 함수 또는 데코레이터.

    Example:
        >>> @qdrant_retry
        ... def search_vectors(query):
        ...     return qdrant.search(query)
    """
    decorator = create_retry_decorator(
        config=QDRANT_CONFIG,
        retry_on=NETWORK_EXCEPTIONS,
    )

    if func is not None:
        return decorator(func)
    return decorator


class RetryContext:
    """수동 제어가 가능한 재시도 컨텍스트 매니저.

    반복문과 함께 사용하여 세밀한 재시도 제어가 가능합니다.

    Attributes:
        config: 재시도 설정.
        retry_on: 재시도할 예외 타입.
        retry_if: 예외 재시도 여부를 결정하는 콜러블.

    Example:
        비동기 사용:
        >>> async with RetryContext(config=NOTION_CONFIG) as ctx:
        ...     for attempt in ctx:
        ...         with attempt:
        ...             result = await api_call()

        동기 사용:
        >>> with RetryContext(config=HTTP_CONFIG) as ctx:
        ...     for attempt in ctx:
        ...         with attempt:
        ...             result = api_call()
    """

    def __init__(
        self,
        config: RetryConfig = DEFAULT_CONFIG,
        retry_on: Optional[Union[Type[Exception], tuple[Type[Exception], ...]]] = None,
        retry_if: Optional[Callable[[BaseException], bool]] = None,
    ):
        """재시도 컨텍스트를 초기화합니다.

        Args:
            config: 재시도 설정.
            retry_on: 재시도할 예외 타입.
            retry_if: 예외 재시도 여부를 결정하는 콜러블.
        """
        self.config = config
        self.retry_on = retry_on
        self.retry_if = retry_if
        self._retrying: Optional[Retrying] = None

    def __enter__(self) -> "Retrying":
        """동기 컨텍스트 진입.

        Returns:
            설정된 Retrying 인스턴스.
        """
        kwargs = self.config.to_tenacity_kwargs()
        if self.retry_if:
            kwargs["retry"] = retry_if_exception(self.retry_if)
        elif self.retry_on:
            kwargs["retry"] = retry_if_exception_type(self.retry_on)

        self._retrying = Retrying(**kwargs)
        return self._retrying

    def __exit__(self, exc_type, exc_val, exc_tb):
        """동기 컨텍스트 종료."""
        self._retrying = None
        return False

    async def __aenter__(self) -> "AsyncRetrying":
        """비동기 컨텍스트 진입.

        Returns:
            설정된 AsyncRetrying 인스턴스.
        """
        kwargs = self.config.to_tenacity_kwargs()
        if self.retry_if:
            kwargs["retry"] = retry_if_exception(self.retry_if)
        elif self.retry_on:
            kwargs["retry"] = retry_if_exception_type(self.retry_on)

        self._retrying = AsyncRetrying(**kwargs)
        return self._retrying

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 종료."""
        self._retrying = None
        return False

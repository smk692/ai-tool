"""structlog 기반 로깅 설정.

프로덕션용 구조화된 JSON 로깅과
개발용 가독성 높은 콘솔 로깅을 제공합니다.

주요 기능:
    - JSON 포맷 (프로덕션) / 컬러 콘솔 (개발)
    - ISO 타임스탬프 자동 추가
    - 로그 레벨 및 로거 이름 포함
    - 스택 트레이스 및 예외 정보 렌더링
    - 파일 로깅 지원

사용 예:
    >>> from .logging_config import configure_logging, get_logger
    >>> configure_logging(level="DEBUG", json_format=False)
    >>> logger = get_logger(__name__)
    >>> logger.info("작업 완료", count=10, duration=1.5)
"""

import logging
import sys
from typing import Optional

import structlog


def configure_logging(
    level: str = "INFO",
    json_format: bool = False,
    log_file: Optional[str] = None,
) -> None:
    """structlog을 애플리케이션에 맞게 설정합니다.

    개발 환경에서는 컬러 콘솔 출력을,
    프로덕션에서는 JSON 포맷을 사용합니다.

    Args:
        level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR).
            기본값 "INFO".
        json_format: JSON 포맷 사용 여부 (프로덕션용).
            기본값 False (개발용 콘솔 출력).
        log_file: 로그 파일 경로. 선택적.
            지정 시 파일에도 로그를 기록합니다.
    """
    # 문자열 레벨을 logging 상수로 변환
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # 표준 라이브러리 로깅 기본 설정
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
    )

    # 파일 핸들러 추가 (지정된 경우)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        logging.getLogger().addHandler(file_handler)

    # 모든 설정에 공통으로 사용되는 프로세서
    shared_processors = [
        structlog.stdlib.add_log_level,  # 로그 레벨 추가
        structlog.stdlib.add_logger_name,  # 로거 이름 추가
        structlog.processors.TimeStamper(fmt="iso"),  # ISO 타임스탬프
        structlog.processors.StackInfoRenderer(),  # 스택 정보
        structlog.processors.UnicodeDecoder(),  # 유니코드 디코딩
    ]

    if json_format:
        # 프로덕션: JSON 출력
        processors = shared_processors + [
            structlog.processors.format_exc_info,  # 예외 정보 포맷팅
            structlog.processors.JSONRenderer(),  # JSON 렌더러
        ]
    else:
        # 개발: 컬러 콘솔 출력
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(
                colors=True,  # ANSI 컬러 활성화
                exception_formatter=structlog.dev.plain_traceback,
            ),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,  # 로거 캐싱으로 성능 향상
    )


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """구조화된 로거 인스턴스를 반환합니다.

    Args:
        name: 로거 이름. 보통 __name__ 사용.

    Returns:
        설정된 structlog 로거.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("이벤트 발생", user_id="123", action="login")
    """
    return structlog.get_logger(name)


class Loggers:
    """애플리케이션 모듈별 사전 구성된 로거.

    각 모듈에서 일관된 로거 이름을 사용하기 위한
    팩토리 메서드를 제공합니다.

    Usage:
        >>> from .logging_config import Loggers
        >>> logger = Loggers.indexer()
        >>> logger.info("인덱싱 시작", document_count=10)
    """

    @staticmethod
    def indexer() -> structlog.stdlib.BoundLogger:
        """인덱서 작업용 로거.

        Returns:
            "rag_indexer.indexer" 이름의 로거.
        """
        return get_logger("rag_indexer.indexer")

    @staticmethod
    def scheduler() -> structlog.stdlib.BoundLogger:
        """스케줄러 작업용 로거.

        Returns:
            "rag_indexer.scheduler" 이름의 로거.
        """
        return get_logger("rag_indexer.scheduler")

    @staticmethod
    def notion() -> structlog.stdlib.BoundLogger:
        """Notion 커넥터용 로거.

        Returns:
            "rag_indexer.notion" 이름의 로거.
        """
        return get_logger("rag_indexer.notion")

    @staticmethod
    def swagger() -> structlog.stdlib.BoundLogger:
        """Swagger 커넥터용 로거.

        Returns:
            "rag_indexer.swagger" 이름의 로거.
        """
        return get_logger("rag_indexer.swagger")

    @staticmethod
    def cli() -> structlog.stdlib.BoundLogger:
        """CLI 작업용 로거.

        Returns:
            "rag_indexer.cli" 이름의 로거.
        """
        return get_logger("rag_indexer.cli")


# 모듈 임포트 시 기본 설정으로 초기화
configure_logging()

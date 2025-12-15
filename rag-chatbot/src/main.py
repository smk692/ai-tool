"""Slack RAG 챗봇 메인 엔트리포인트.

Slack Bolt 앱을 초기화하고 Socket Mode로 실행합니다.
"""

import logging
import sys

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from .config import get_settings
from .handlers import (
    register_dm_handlers,
    register_feedback_handlers,
    register_mention_handlers,
)
from .services import get_rag_service

# 로거 설정
logger = logging.getLogger(__name__)


def create_app() -> App:
    """Slack Bolt 앱 생성.

    Returns:
        설정된 Slack Bolt App 인스턴스
    """
    settings = get_settings()

    app = App(
        token=settings.slack_bot_token,
        signing_secret=settings.slack_signing_secret,
    )

    # 이벤트 핸들러 등록
    _register_handlers(app)

    logger.info("Slack Bolt 앱 초기화 완료")
    return app


def _register_handlers(app: App) -> None:
    """이벤트 핸들러 등록.

    Args:
        app: Slack Bolt App 인스턴스
    """
    # 앱 멘션 핸들러 등록
    register_mention_handlers(app)

    # DM 메시지 핸들러 등록
    register_dm_handlers(app)

    # 피드백 리액션 핸들러 등록
    register_feedback_handlers(app)

    logger.debug("이벤트 핸들러 등록 완료")


def setup_logging() -> None:
    """로깅 설정."""
    settings = get_settings()

    log_level = getattr(logging, settings.log_level)

    if settings.log_format == "json":
        log_format = (
            '{"time": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "message": "%(message)s"}'
        )
    else:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def preload_services() -> None:
    """서비스 사전 로드.

    임베딩 모델과 벡터 DB 연결을 앱 시작 시 미리 초기화합니다.
    """
    import time

    total_start = time.time()
    logger.info("=" * 50)
    logger.info("서비스 사전 로드 시작...")
    logger.info("=" * 50)

    # RAG 서비스 초기화
    logger.info("[1/3] RAG 서비스 초기화 중...")
    start = time.time()
    rag_service = get_rag_service()
    logger.info(f"[1/3] RAG 서비스 초기화 완료 ({time.time() - start:.2f}초)")

    # 임베딩 모델 워밍업 (모델 로드 - 가장 오래 걸림)
    logger.info("[2/3] 임베딩 모델 로드 중...")
    logger.info("      ⏳ 모델: intfloat/multilingual-e5-large-instruct (2.2GB)")
    logger.info("      💡 팁: 'make download-model'로 미리 다운로드하면 빨라집니다")
    start = time.time()
    _ = rag_service._embedding_model.embed_query("warmup query")
    logger.info(f"[2/3] 임베딩 모델 로드 완료 ({time.time() - start:.2f}초)")

    # 벡터 DB 연결 확인
    logger.info("[3/3] 벡터 DB 연결 확인 중...")
    start = time.time()
    try:
        _ = rag_service._vector_store.client.get_collections()
        logger.info(f"[3/3] 벡터 DB 연결 성공 ({time.time() - start:.2f}초)")
    except Exception as e:
        logger.warning(f"[3/3] 벡터 DB 연결 실패 (계속 진행): {e}")

    logger.info("=" * 50)
    logger.info(f"서비스 사전 로드 완료 (총 {time.time() - total_start:.2f}초)")
    logger.info("=" * 50)


def main() -> None:
    """메인 실행 함수."""
    setup_logging()

    logger.info("Slack RAG 챗봇 시작 중...")

    # 서비스 사전 로드 (임베딩 모델, 벡터 DB)
    preload_services()

    settings = get_settings()
    app = create_app()

    # Socket Mode 핸들러 생성 및 실행
    handler = SocketModeHandler(app, settings.slack_app_token)

    logger.info("Socket Mode 연결 시작...")
    handler.start()


if __name__ == "__main__":
    main()

"""애플리케이션 설정 모듈.

환경 변수 기반 설정 관리를 제공합니다.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """애플리케이션 설정.

    환경 변수 또는 .env 파일에서 설정을 로드합니다.
    """

    # Slack 설정
    slack_bot_token: str = Field(..., description="Slack Bot 토큰 (xoxb-)")
    slack_app_token: str = Field(..., description="Slack App 토큰 (xapp-)")
    slack_signing_secret: str = Field(..., description="Slack Signing Secret")
    slack_bot_user_id: str | None = Field(
        default=None, description="Slack Bot 사용자 ID (피드백 필터용)"
    )

    # Claude Agent SDK 설정 (OAuth 인증 시 선택적)
    anthropic_api_key: str | None = Field(
        default=None, description="Anthropic API 키 (CLI OAuth 인증 시 불필요)"
    )

    # Qdrant 설정
    qdrant_host: str = Field(default="localhost", description="Qdrant 호스트")
    qdrant_port: int = Field(default=6333, description="Qdrant 포트")
    qdrant_collection: str = Field(default="rag_documents", description="Qdrant 컬렉션명")

    # Redis 설정
    redis_host: str = Field(default="localhost", description="Redis 호스트")
    redis_port: int = Field(default=6379, description="Redis 포트")
    redis_db: int = Field(default=0, description="Redis DB 번호")

    # RAG 설정
    rag_top_k: int = Field(default=5, ge=1, le=20, description="검색 결과 개수")
    rag_score_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="유사도 임계값"
    )
    rag_max_context_tokens: int = Field(
        default=4000, ge=100, description="컨텍스트 최대 토큰 수"
    )

    # 대화 컨텍스트 설정
    conversation_ttl_seconds: int = Field(
        default=3600, ge=60, description="대화 TTL (초)"
    )
    conversation_max_messages: int = Field(
        default=10, ge=1, le=50, description="대화 최대 메시지 수"
    )

    # 로깅 설정
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="로그 레벨"
    )
    log_format: Literal["json", "text"] = Field(
        default="json", description="로그 형식"
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",  # .env 파일의 추가 변수 무시
    }

    @property
    def redis_url(self) -> str:
        """Redis 연결 URL 반환."""
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


@lru_cache
def get_settings() -> Settings:
    """설정 싱글톤 인스턴스 반환.

    Returns:
        Settings 인스턴스 (캐시됨)
    """
    return Settings()

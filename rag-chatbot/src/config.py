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
        default=10800, ge=60, description="대화 TTL (초) - 기본 3시간"
    )
    conversation_max_messages: int = Field(
        default=10, ge=1, le=50, description="대화 최대 메시지 수"
    )

    # DM 대화 컨텍스트 설정
    dm_conversation_max_messages: int = Field(
        default=3, ge=1, le=10, description="DM 대화 최대 메시지 수"
    )

    # 리액션 설정
    reaction_processing: str = Field(
        default="eyes", description="처리 중 리액션 (👀)"
    )
    reaction_done: str = Field(
        default="white_check_mark", description="완료 리액션 (✅)"
    )

    # 이미지 처리 설정
    image_processing_enabled: bool = Field(
        default=True, description="이미지 처리 활성화 여부"
    )
    image_max_size_mb: int = Field(
        default=20, ge=1, le=50, description="최대 이미지 크기 (MB)"
    )
    image_max_count: int = Field(
        default=5, ge=1, le=10, description="요청당 최대 이미지 수"
    )
    image_download_timeout: int = Field(
        default=30, ge=5, le=120, description="이미지 다운로드 타임아웃 (초)"
    )

    # 로깅 설정
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="로그 레벨"
    )
    log_format: Literal["json", "text"] = Field(
        default="json", description="로그 형식"
    )

    # MCP 서버 설정 - Grafana (필수: URL, 토큰)
    grafana_url: str | None = Field(default=None, description="Grafana URL")
    grafana_service_account_token: str | None = Field(
        default=None, description="Grafana Service Account 토큰"
    )

    # MCP 서버 설정 - Sentry (필수: 토큰, 선택: 호스트)
    sentry_access_token: str | None = Field(default=None, description="Sentry Access 토큰")
    sentry_host: str | None = Field(default=None, description="Sentry 호스트 (Self-hosted)")

    # MCP 서버 설정 - AWS (필수: 프로필, 리전)
    aws_profile: str = Field(default="default", description="AWS 프로필")
    aws_region: str = Field(default="ap-northeast-2", description="AWS 리전")

    # MCP 서버 설정 - Swagger (필수: JAR 경로)
    swagger_mcp_jar_path: str | None = Field(default=None, description="Swagger MCP JAR 경로")

    # MCP 서버 설정 - Jira (필수: 사이트명, 이메일, API 토큰)
    atlassian_site_name: str | None = Field(
        default=None, description="Atlassian 사이트명 (예: mycompany.atlassian.net의 mycompany)"
    )
    atlassian_user_email: str | None = Field(
        default=None, description="Atlassian 사용자 이메일"
    )
    atlassian_api_token: str | None = Field(
        default=None, description="Atlassian API 토큰"
    )

    # MCP 서버 설정 - Notion (OAuth 인증 사용)
    notion_mcp_enabled: bool = Field(
        default=False, description="Notion MCP 활성화 여부 (OAuth 인증 필요)"
    )

    # MCP 서버 설정 - Slack (채널/메시지/사용자 조회)
    slack_mcp_enabled: bool = Field(
        default=False, description="Slack MCP 활성화 여부"
    )
    slack_team_id: str | None = Field(
        default=None, description="Slack Team ID (Slack MCP용)"
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

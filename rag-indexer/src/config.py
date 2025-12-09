"""RAG Indexer 설정 관리.

환경 변수에서 설정을 로드하고 적절한 기본값을 제공합니다.

설정 그룹:
    QdrantSettings: Qdrant 벡터 데이터베이스 연결 설정
    NotionSettings: Notion API 연결 및 Rate Limit 설정
    EmbeddingSettings: 임베딩 모델 설정
    ChunkingSettings: 텍스트 청킹 파라미터
    SchedulerSettings: 자동 동기화 스케줄러 설정
    AISettings: AI 메타데이터 추출 설정
    Settings: 메인 애플리케이션 설정 (모든 하위 설정 포함)

환경 변수:
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, QDRANT_API_KEY
    NOTION_API_KEY, NOTION_RATE_LIMIT_DELAY, NOTION_MAX_RETRIES
    EMBEDDING_MODEL, EMBEDDING_DIMENSION, EMBEDDING_BATCH_SIZE
    CHUNK_SIZE, CHUNK_OVERLAP
    SCHEDULER_ENABLED, SCHEDULER_CRON, SCHEDULER_TIMEZONE
    AI_EXTRACTION_ENABLED, ANTHROPIC_API_KEY, AI_MODEL, AI_MAX_TOKENS
    DEBUG, LOG_LEVEL, DATA_DIR
"""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class QdrantSettings(BaseSettings):
    """Qdrant 벡터 데이터베이스 설정.

    Qdrant 서버 연결에 필요한 설정을 관리합니다.

    Attributes:
        host: Qdrant 서버 호스트. 기본값 "localhost".
        port: Qdrant 서버 포트. 기본값 6333.
        collection_name: 사용할 컬렉션 이름. 기본값 "rag_documents".
        api_key: Qdrant Cloud 사용 시 API 키. 선택적.
    """

    host: str = Field(default="localhost", alias="QDRANT_HOST")
    port: int = Field(default=6333, alias="QDRANT_PORT")
    collection_name: str = Field(default="rag_documents", alias="QDRANT_COLLECTION")
    api_key: Optional[str] = Field(default=None, alias="QDRANT_API_KEY")

    model_config = {"env_prefix": "", "extra": "ignore"}

    @property
    def url(self) -> str:
        """Qdrant URL을 반환합니다.

        Returns:
            http://{host}:{port} 형식의 URL.
        """
        return f"http://{self.host}:{self.port}"


class NotionSettings(BaseSettings):
    """Notion API 설정.

    Notion API 연결 및 요청 제한에 관한 설정입니다.

    Attributes:
        api_key: Notion Integration API 키. 필수.
        rate_limit_delay: 요청 간 지연 시간 (초). 기본값 0.35.
            Notion의 Rate Limit (3 req/s)을 준수하기 위함.
        max_retries: 실패 시 최대 재시도 횟수. 기본값 5.
    """

    api_key: str = Field(default="", alias="NOTION_API_KEY")
    rate_limit_delay: float = Field(default=0.35, alias="NOTION_RATE_LIMIT_DELAY")
    max_retries: int = Field(default=5, alias="NOTION_MAX_RETRIES")

    model_config = {"env_prefix": "", "extra": "ignore"}


class EmbeddingSettings(BaseSettings):
    """임베딩 모델 설정.

    텍스트를 벡터로 변환하는 임베딩 모델의 설정입니다.

    Attributes:
        model_name: HuggingFace 모델 이름.
            기본값 "intfloat/multilingual-e5-large-instruct".
            다국어 지원을 위해 multilingual 모델 사용.
        dimension: 임베딩 벡터 차원. 기본값 1024.
            모델에 따라 다름 (e5-large는 1024).
        batch_size: 배치 처리 크기. 기본값 32.
            메모리와 속도 간의 균형을 위한 설정.
    """

    model_name: str = Field(
        default="intfloat/multilingual-e5-large-instruct",
        alias="EMBEDDING_MODEL",
    )
    dimension: int = Field(default=1024, alias="EMBEDDING_DIMENSION")
    batch_size: int = Field(default=32, alias="EMBEDDING_BATCH_SIZE")

    model_config = {"env_prefix": "", "extra": "ignore"}


class ChunkingSettings(BaseSettings):
    """텍스트 청킹 설정.

    문서를 작은 청크로 분할하는 파라미터입니다.

    Attributes:
        chunk_size: 청크 최대 크기 (문자 수). 기본값 1000.
            검색 정밀도와 컨텍스트 길이 간의 균형.
        chunk_overlap: 청크 간 오버랩 크기. 기본값 200.
            문맥 연속성 유지를 위한 중복 영역.
    """

    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")

    model_config = {"env_prefix": "", "extra": "ignore"}


class SchedulerSettings(BaseSettings):
    """스케줄러 설정.

    자동 동기화 스케줄러의 동작을 제어합니다.

    Attributes:
        enabled: 스케줄러 활성화 여부. 기본값 False.
        cron_expression: Cron 표현식. 기본값 "0 6 * * *" (매일 오전 6시).
        timezone: 시간대. 기본값 "Asia/Seoul".
        max_workers: 최대 동시 작업 수. 기본값 2.
    """

    enabled: bool = Field(default=False, alias="SCHEDULER_ENABLED")
    cron_expression: str = Field(default="0 6 * * *", alias="SCHEDULER_CRON")
    timezone: str = Field(default="Asia/Seoul", alias="SCHEDULER_TIMEZONE")
    max_workers: int = Field(default=2, alias="SCHEDULER_MAX_WORKERS")

    model_config = {"env_prefix": "", "extra": "ignore"}


class AISettings(BaseSettings):
    """AI 메타데이터 추출 설정.

    Claude를 사용한 메타데이터 자동 추출 기능의 설정입니다.
    Phase 1.2 기능으로, 검색 품질 향상에 사용됩니다.

    Attributes:
        enabled: AI 추출 기능 활성화 여부. 기본값 False.
        api_key: Anthropic API 키. ANTHROPIC_API_KEY 환경 변수.
        model: 사용할 Claude 모델. 기본값 "claude-3-haiku-20240307".
            비용 효율성을 위해 Haiku 모델 기본 사용.
        max_tokens: 응답 최대 토큰 수. 기본값 1024.
        timeout: 요청 타임아웃 (초). 기본값 30.0.
        cache_enabled: 추출 결과 캐싱 여부. 기본값 True.
            동일 콘텐츠에 대한 중복 API 호출 방지.
    """

    enabled: bool = Field(default=False, alias="AI_EXTRACTION_ENABLED")
    api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    model: str = Field(default="claude-3-haiku-20240307", alias="AI_MODEL")
    max_tokens: int = Field(default=1024, alias="AI_MAX_TOKENS")
    timeout: float = Field(default=30.0, alias="AI_TIMEOUT")
    cache_enabled: bool = Field(default=True, alias="AI_CACHE_ENABLED")

    model_config = {"env_prefix": "", "extra": "ignore"}


class Settings(BaseSettings):
    """메인 애플리케이션 설정.

    모든 하위 설정 그룹을 포함하는 최상위 설정 클래스입니다.

    Attributes:
        app_name: 애플리케이션 이름. 기본값 "rag-indexer".
        debug: 디버그 모드 활성화 여부. 기본값 False.
        log_level: 로그 레벨. 기본값 "INFO".
        data_dir: 데이터 디렉토리 경로. 기본값 "data".
        qdrant: Qdrant 벡터 DB 설정.
        notion: Notion API 설정.
        embedding: 임베딩 모델 설정.
        chunking: 텍스트 청킹 설정.
        scheduler: 스케줄러 설정.
        ai: AI 메타데이터 추출 설정.
    """

    # 애플리케이션 기본 설정
    app_name: str = Field(default="rag-indexer")
    debug: bool = Field(default=False, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # 데이터 디렉토리
    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")

    # 하위 설정 그룹
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    notion: NotionSettings = Field(default_factory=NotionSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)
    ai: AISettings = Field(default_factory=AISettings)

    model_config = {"env_prefix": "", "extra": "ignore"}

    def ensure_data_dir(self) -> Path:
        """데이터 디렉토리가 존재하는지 확인하고 경로를 반환합니다.

        디렉토리가 없으면 생성합니다.

        Returns:
            데이터 디렉토리 Path 객체.
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir


def get_settings() -> Settings:
    """애플리케이션 설정을 반환합니다.

    매 호출마다 새 인스턴스를 생성합니다.
    캐싱이 필요한 경우 별도로 관리해야 합니다.

    Returns:
        Settings 인스턴스.
    """
    return Settings()


# 전역 설정 인스턴스
settings = get_settings()

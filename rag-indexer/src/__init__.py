"""RAG Indexer - 문서 인덱싱 서비스.

Notion과 Swagger 소스에서 문서를 인덱싱하는 파이프라인을 제공합니다.

주요 기능:
    - Notion 페이지 및 데이터베이스 인덱싱
    - Swagger/OpenAPI 스펙 파싱 및 인덱싱
    - 벡터 데이터베이스 (Qdrant) 저장
    - 스케줄러를 통한 자동 동기화
    - CLI 인터페이스

모듈 구성:
    config: 환경 변수 기반 설정 관리
    connectors: 외부 데이터 소스 커넥터 (Notion, Swagger)
    models: 데이터 모델 (Source, Document, Chunk, SyncJob)
    services: 핵심 서비스 (Chunker, Embedder, Indexer)
    scheduler: APScheduler 기반 자동 동기화
    storage: JSON 파일 기반 상태 저장소
    cli: Typer 기반 CLI 인터페이스
"""

from .cli import cli
from .config import Settings, get_settings
from .connectors import NotionConnector, SwaggerConnector
from .models import (
    Chunk,
    Document,
    NotionSourceConfig,
    Source,
    SourceType,
    SwaggerSourceConfig,
    SyncError,
    SyncJob,
    SyncJobStatus,
    SyncJobTrigger,
)
from .scheduler import SyncScheduler, get_scheduler
from .services import Chunker, Embedder, Indexer, get_chunker, get_embedder, get_indexer
from .storage import Storage, get_storage

__version__ = "0.1.0"

__all__ = [
    # CLI
    "cli",
    # 설정
    "Settings",
    "get_settings",
    # 커넥터
    "NotionConnector",
    "SwaggerConnector",
    # 모델
    "Source",
    "SourceType",
    "NotionSourceConfig",
    "SwaggerSourceConfig",
    "Document",
    "Chunk",
    "SyncJob",
    "SyncJobStatus",
    "SyncJobTrigger",
    "SyncError",
    # 서비스
    "Chunker",
    "get_chunker",
    "Embedder",
    "get_embedder",
    "Indexer",
    "get_indexer",
    # 스케줄러
    "SyncScheduler",
    "get_scheduler",
    # 저장소
    "Storage",
    "get_storage",
]

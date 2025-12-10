"""RAG Indexer 데이터 모델.

모든 모델 클래스를 외부에서 사용할 수 있도록 내보냅니다.

모델 구성:
    - Source: 데이터 소스 정의 (Notion, Swagger)
    - Document: 소스에서 추출한 문서
    - Chunk: 벡터 인덱싱을 위한 텍스트 청크
    - SyncJob: 동기화 작업 실행 기록
"""

from .chunk import Chunk, ChunkType
from .document import Document
from .source import NotionSourceConfig, Source, SourceType, SwaggerSourceConfig
from .sync_job import SyncError, SyncJob, SyncJobStatus, SyncJobTrigger

__all__ = [
    # 소스 관련
    "Source",
    "SourceType",
    "NotionSourceConfig",
    "SwaggerSourceConfig",
    # 문서
    "Document",
    # 청크
    "Chunk",
    "ChunkType",
    # 동기화 작업
    "SyncJob",
    "SyncJobStatus",
    "SyncJobTrigger",
    "SyncError",
]

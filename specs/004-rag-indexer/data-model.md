# 데이터 모델: RAG Document Indexer

**작성일**: 2025-12-05
**버전**: 1.0

## 개요

RAG Document Indexer의 핵심 데이터 엔티티와 상태 관리를 정의합니다.

---

## 핵심 엔티티

### 1. Source (데이터 소스)

데이터를 수집할 외부 소스 정의입니다.

```python
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional

class SourceType(str, Enum):
    NOTION = "notion"
    SWAGGER = "swagger"

class NotionSourceConfig(BaseModel):
    """Notion 소스 설정"""
    page_ids: list[str] = Field(default_factory=list, description="동기화할 페이지 ID 목록")
    database_ids: list[str] = Field(default_factory=list, description="동기화할 데이터베이스 ID 목록")
    include_children: bool = Field(default=True, description="하위 페이지 포함 여부")

class SwaggerSourceConfig(BaseModel):
    """Swagger 소스 설정"""
    url: str = Field(..., description="Swagger JSON URL")
    auth_header: Optional[str] = Field(None, description="인증 헤더 (선택)")

class Source(BaseModel):
    """데이터 소스 정의"""
    id: str = Field(..., description="소스 고유 식별자 (UUID)")
    name: str = Field(..., description="소스 이름")
    source_type: SourceType = Field(..., description="소스 타입")
    config: NotionSourceConfig | SwaggerSourceConfig = Field(..., description="타입별 설정")
    enabled: bool = Field(default=True, description="활성화 상태")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_synced_at: Optional[datetime] = Field(None, description="마지막 동기화 시간")
```

---

### 2. Document (문서)

소스에서 추출한 개별 문서 단위입니다.

```python
class Document(BaseModel):
    """원본 문서"""
    id: str = Field(..., description="문서 고유 식별자 (UUID)")
    source_id: str = Field(..., description="소속 소스 ID")
    external_id: str = Field(..., description="외부 시스템 ID (Notion page_id 등)")
    title: str = Field(..., description="문서 제목")
    url: Optional[str] = Field(None, description="원본 URL")
    content_hash: str = Field(..., description="콘텐츠 SHA256 해시")
    metadata: dict = Field(default_factory=dict, description="추가 메타데이터")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    indexed_at: Optional[datetime] = Field(None, description="인덱싱 완료 시간")
```

**메타데이터 예시**:
```python
# Notion 페이지
{
    "notion_type": "page",
    "parent_id": "abc123",
    "last_edited_by": "user@example.com",
    "icon": "📄"
}

# Swagger 엔드포인트
{
    "swagger_version": "3.0.0",
    "method": "POST",
    "path": "/api/users",
    "tags": ["users"]
}
```

---

### 3. Chunk (청크)

문서를 분할한 검색 가능한 텍스트 조각입니다.

```python
class Chunk(BaseModel):
    """텍스트 청크"""
    id: str = Field(..., description="청크 고유 식별자 (UUID)")
    document_id: str = Field(..., description="소속 문서 ID")
    chunk_index: int = Field(..., description="문서 내 청크 순서 (0부터 시작)")
    text: str = Field(..., description="청크 텍스트")
    token_count: int = Field(..., description="토큰 수 (근사치)")
    embedding: Optional[list[float]] = Field(None, description="임베딩 벡터 (1024차원)")
    metadata: dict = Field(default_factory=dict, description="청크별 메타데이터")
    created_at: datetime = Field(default_factory=datetime.utcnow)
```

**Qdrant 페이로드 구조**:
```python
{
    "chunk_id": "uuid",
    "document_id": "uuid",
    "source_id": "uuid",
    "source_type": "notion",
    "title": "문서 제목",
    "url": "https://notion.so/...",
    "chunk_index": 0,
    "text": "청크 텍스트 내용..."
}
```

---

### 4. SyncJob (동기화 작업)

동기화 작업의 실행 기록입니다.

```python
class SyncJobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # 일부 성공

class SyncJob(BaseModel):
    """동기화 작업"""
    id: str = Field(..., description="작업 고유 식별자 (UUID)")
    source_id: Optional[str] = Field(None, description="대상 소스 ID (None이면 전체)")
    trigger: str = Field(..., description="트리거 타입: manual, scheduled")
    status: SyncJobStatus = Field(default=SyncJobStatus.PENDING)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)

    # 통계
    documents_processed: int = Field(default=0, description="처리된 문서 수")
    documents_created: int = Field(default=0, description="신규 생성 문서 수")
    documents_updated: int = Field(default=0, description="업데이트 문서 수")
    documents_deleted: int = Field(default=0, description="삭제된 문서 수")
    documents_skipped: int = Field(default=0, description="스킵된 문서 수 (변경 없음)")
    chunks_created: int = Field(default=0, description="생성된 청크 수")

    # 에러 추적
    errors: list[dict] = Field(default_factory=list, description="에러 목록")
    error_message: Optional[str] = Field(None, description="주요 에러 메시지")
```

**에러 구조 예시**:
```python
{
    "document_id": "abc123",
    "error_type": "NotionAPIError",
    "message": "Rate limit exceeded",
    "timestamp": "2025-12-05T06:00:00Z",
    "retryable": True
}
```

---

## 상태 다이어그램

### SyncJob 상태 전이

```
                    ┌─────────┐
                    │ PENDING │
                    └────┬────┘
                         │ start()
                         ▼
                    ┌─────────┐
              ┌────►│ RUNNING │◄────┐
              │     └────┬────┘     │
              │          │          │
              │    ┌─────┴─────┐    │
              │    │           │    │
              │    ▼           ▼    │
         retry│ ┌──────┐  ┌────────┐│ partial success
              │ │FAILED│  │PARTIAL ││
              │ └──────┘  └────────┘│
              │                     │
              └─────────────────────┘
                         │
                         │ all success
                         ▼
                   ┌───────────┐
                   │ COMPLETED │
                   └───────────┘
```

### Document 인덱싱 플로우

```
 [Source]
    │
    │ fetch()
    ▼
 [Raw Content]
    │
    │ parse()
    ▼
 [Document]──────────┐
    │                │
    │ hash match?    │ yes
    │                ▼
    │ no         [SKIP]
    ▼
 [Chunking]
    │
    │ split()
    ▼
 [Chunks]
    │
    │ embed()
    ▼
 [Embeddings]
    │
    │ upsert()
    ▼
 [Qdrant]
```

---

## 저장소 설계

### 로컬 상태 저장 (JSON 파일)

MVP에서는 SQLite 대신 JSON 파일로 상태 관리합니다.

```
rag-indexer/
├── data/
│   ├── sources.json      # Source 목록
│   ├── documents.json    # Document 목록 (해시 포함)
│   └── sync_history.json # SyncJob 기록 (최근 100개)
```

**sources.json 구조**:
```json
{
  "sources": [
    {
      "id": "uuid-1",
      "name": "팀 위키",
      "source_type": "notion",
      "config": {
        "page_ids": ["abc123"],
        "database_ids": [],
        "include_children": true
      },
      "enabled": true,
      "last_synced_at": "2025-12-05T06:00:00Z"
    }
  ]
}
```

### Qdrant 벡터 저장소

**컬렉션 설정**:
```python
from qdrant_client.models import Distance, VectorParams

COLLECTION_CONFIG = {
    "collection_name": "rag_documents",
    "vectors_config": VectorParams(
        size=1024,  # intfloat/multilingual-e5-large-instruct
        distance=Distance.COSINE
    )
}
```

**인덱스 전략**:
- Point ID: chunk_id (UUID)
- 벡터: 768차원 float 배열
- 페이로드: 검색/필터링용 메타데이터

---

## 데이터 무결성 규칙

### 제약 조건

1. **Source 유일성**: `name`은 중복 불가
2. **Document 유일성**: 동일 소스 내 `external_id` 중복 불가
3. **Chunk 순서**: 동일 문서 내 `chunk_index`는 0부터 연속
4. **해시 검증**: `content_hash`는 SHA256 hex 문자열 (64자)

### 삭제 정책

1. **Source 삭제 시**: 연관된 모든 Document, Chunk, Qdrant 포인트 삭제
2. **Document 삭제 시**: 연관된 모든 Chunk, Qdrant 포인트 삭제
3. **재동기화 시**: 원본에 없는 Document는 soft delete 후 정리

### 데이터 정합성

```python
class DataIntegrityChecker:
    """데이터 정합성 검증"""

    async def verify_qdrant_sync(self) -> list[str]:
        """Qdrant와 로컬 상태 정합성 검증"""
        issues = []

        # 1. 로컬에 있는데 Qdrant에 없는 청크
        for chunk in local_chunks:
            if not await qdrant.point_exists(chunk.id):
                issues.append(f"Missing in Qdrant: {chunk.id}")

        # 2. Qdrant에 있는데 로컬에 없는 포인트
        orphan_points = await qdrant.find_orphans()
        for point_id in orphan_points:
            issues.append(f"Orphan in Qdrant: {point_id}")

        return issues
```

---

## 마이그레이션 전략

### 버전 관리

```python
DATA_VERSION = "1.0.0"

class DataMigration:
    """데이터 스키마 마이그레이션"""

    MIGRATIONS = {
        "1.0.0": None,  # 초기 버전
        # "1.1.0": migrate_1_0_to_1_1,  # 향후 마이그레이션
    }
```

### 백업 정책

- 동기화 전 자동 백업: `data/backup/YYYYMMDD_HHMMSS/`
- 최근 7일 백업 보관
- 복구: `rag-indexer restore --backup <path>`

---

## 성능 고려사항

### 벌크 작업

```python
# 청크 일괄 upsert (100개 단위)
BATCH_SIZE = 100

async def bulk_upsert_chunks(chunks: list[Chunk]):
    for batch in chunked(chunks, BATCH_SIZE):
        points = [chunk_to_point(c) for c in batch]
        await qdrant.upsert(points=points)
```

### 메모리 관리

- 대용량 문서: 스트리밍 처리로 메모리 제한
- 임베딩 배치: GPU 메모리 고려하여 32개씩 처리
- 캐시: Document 해시 메모리 캐시 (LRU, 최대 1000개)

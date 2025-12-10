# RAG Document Indexer

RAG 문서 인덱싱 파이프라인 - Notion 및 Swagger 문서를 벡터 데이터베이스에 색인화합니다.

## Features

- **Notion 연동**: 페이지 및 데이터베이스 자동 동기화
- **Swagger/OpenAPI 지원**: API 문서 자동 파싱 및 색인화
- **스마트 청킹**: 다국어 지원 텍스트 분할 (한국어/영어)
- **벡터 임베딩**: HuggingFace `intfloat/multilingual-e5-large-instruct` (1024차원)
- **스케줄러**: cron 기반 자동 동기화
- **CLI 도구**: 수동 관리 및 검색 인터페이스
- **Rate Limiting**: Token bucket 기반 API 호출 제한
- **에러 복구**: 자동 재시도 및 exponential backoff

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│  Notion API     │     │  Swagger JSON   │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│           Document Connectors           │
│   (NotionConnector, SwaggerConnector)   │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│              Chunker                    │
│   (LangChain RecursiveTextSplitter)     │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│              Embedder                   │
│   (multilingual-e5-large-instruct)      │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│            Vector Store                 │
│              (Qdrant)                   │
└─────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.10+
- Qdrant (벡터 데이터베이스)
- Notion API 키 (Notion 동기화 시)

### Setup

```bash
# 1. shared 모듈 설치
cd shared && pip install -e .

# 2. rag-indexer 설치
cd ../rag-indexer && pip install -e .

# 3. 개발 의존성 포함 설치
pip install -e ".[dev]"
```

## Configuration

환경 변수를 `.env` 파일 또는 시스템 환경에 설정합니다:

```bash
# Vector DB (필수)
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Notion API (Notion 소스 사용 시)
NOTION_API_KEY=your_notion_api_key

# Embedding Model (선택, 기본값 사용 가능)
EMBEDDING_MODEL=intfloat/multilingual-e5-large-instruct
```

## Usage

### CLI Commands

```bash
# 시스템 상태 확인
rag-indexer status

# 소스 관리
rag-indexer source list
rag-indexer source add --name "API Docs" --type swagger --url https://api.example.com/swagger.json
rag-indexer source add --name "Wiki" --type notion --page-ids "page-id-1,page-id-2"
rag-indexer source show <source-name>
rag-indexer source remove <source-name>

# 동기화
rag-indexer sync run                    # 전체 소스 동기화
rag-indexer sync run --source <name>    # 특정 소스만 동기화
rag-indexer sync history                # 동기화 기록

# 스케줄러
rag-indexer scheduler start             # 스케줄러 시작
rag-indexer scheduler stop              # 스케줄러 중지
rag-indexer scheduler status            # 스케줄러 상태

# 검색 (테스트용)
rag-indexer search "검색 쿼리"
rag-indexer search "query" --limit 5
```

### Python API

```python
from src.services import Indexer, get_indexer
from src.models import Source, SwaggerSourceConfig

# 인덱서 초기화
indexer = get_indexer()

# 소스 생성
source = Source(
    id="api-docs",
    name="API Documentation",
    source_type="swagger",
    config=SwaggerSourceConfig(url="https://api.example.com/swagger.json"),
)

# 문서 색인화
from src.connectors import SwaggerConnector
connector = SwaggerConnector()
new_docs, updated_docs, deleted_ids = connector.fetch_documents(source, [])
result = indexer.index_documents(new_docs, source)
print(f"Indexed {result['chunks_created']} chunks")
```

## Project Structure

```
rag-indexer/
├── src/
│   ├── cli.py                 # CLI 인터페이스 (Typer)
│   ├── config.py              # 설정 관리
│   ├── models/                # Pydantic 모델
│   │   ├── chunk.py           # 청크 모델
│   │   ├── document.py        # 문서 모델
│   │   └── source.py          # 소스 설정 모델
│   ├── connectors/            # 데이터 소스 커넥터
│   │   ├── base.py            # 기본 커넥터 인터페이스
│   │   ├── notion.py          # Notion API 커넥터
│   │   └── swagger.py         # Swagger/OpenAPI 커넥터
│   ├── services/              # 핵심 서비스
│   │   ├── chunker.py         # 텍스트 청킹
│   │   ├── embedder.py        # 임베딩 생성
│   │   ├── indexer.py         # 벡터 DB 색인화
│   │   └── scheduler.py       # 스케줄러
│   ├── storage/               # 메타데이터 저장소
│   │   └── sqlite.py          # SQLite 스토리지
│   └── utils/                 # 유틸리티
│       ├── errors.py          # 에러 정의
│       ├── rate_limit.py      # Rate limiting
│       └── retry.py           # 재시도 로직
├── tests/
│   ├── unit/                  # 단위 테스트 (92 tests)
│   └── integration/           # 통합 테스트 (50 tests)
└── pyproject.toml
```

## Development

### Testing

```bash
# 전체 테스트 실행
pytest tests/ -v

# 단위 테스트만
pytest tests/unit -v

# 통합 테스트만
pytest tests/integration -v

# 커버리지 포함
pytest tests/ --cov=src --cov-report=html
```

### Linting

```bash
# 코드 스타일 검사
ruff check src/ tests/

# 자동 수정
ruff check src/ tests/ --fix

# 포맷팅
ruff format src/ tests/
```

### Type Checking

```bash
mypy src/
```

## Source Types

### Notion

Notion 페이지 및 데이터베이스를 동기화합니다.

```bash
rag-indexer source add \
  --name "Product Wiki" \
  --type notion \
  --page-ids "page-id-1,page-id-2" \
  --database-ids "db-id-1"
```

**설정 옵션**:
- `page_ids`: 동기화할 페이지 ID 목록
- `database_ids`: 동기화할 데이터베이스 ID 목록
- `include_children`: 하위 페이지 포함 여부 (기본: true)
- `max_depth`: 최대 탐색 깊이 (기본: 5)

### Swagger/OpenAPI

Swagger 또는 OpenAPI 스펙을 파싱하여 API 엔드포인트별로 문서를 생성합니다.

```bash
rag-indexer source add \
  --name "Backend API" \
  --type swagger \
  --url "https://api.example.com/swagger.json"
```

**지원 버전**:
- Swagger 2.0
- OpenAPI 3.0.x

## Rate Limiting

API 호출에 대한 rate limiting이 적용됩니다:

| Service | Requests/sec | Burst |
|---------|-------------|-------|
| Notion API | 3 | 10 |
| HTTP (Swagger) | 10 | 20 |
| Qdrant | 100 | 100 |

## Error Handling

자동 재시도 및 복구 기능:

- **재시도 횟수**: 최대 3회
- **백오프**: Exponential (1s, 2s, 4s)
- **재시도 대상**: Rate limit 오류, 일시적 네트워크 오류
- **재시도 제외**: 인증 오류, 잘못된 요청

## License

MIT

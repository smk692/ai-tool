# Implementation Plan: RAG Document Indexer

**Branch**: `004-rag-indexer` | **Date**: 2025-12-09 | **Spec**: [spec.md](./spec.md)
**Input**: RAG 문서 등록 파이프라인 - Notion 데이터와 Swagger.json을 벡터DB에 등록

## Summary

문서 인덱싱 서비스로, Notion 페이지/데이터베이스와 Swagger/OpenAPI 문서를 벡터DB(Qdrant)에 동기화합니다. 스케줄러 자동 동기화 및 CLI 수동 트리거를 지원하며, 증분 업데이트로 변경된 문서만 재처리합니다.

## Technical Context

**Language/Version**: Python 3.10+ (타입 힌팅 필수)
**Primary Dependencies**:
- `notion-client>=2.0.0` (Notion API)
- `httpx>=0.25.0` (HTTP 클라이언트)
- `pydantic>=2.0.0` (데이터 검증)
- `langchain-text-splitters>=0.0.1` (텍스트 청킹)
- `apscheduler>=3.10.0` (스케줄러)
- `typer>=0.9.0` (CLI)
- `tenacity>=8.2.0` (재시도 로직)
- `anthropic>=0.40.0` (AI 메타데이터 추출)
- `rag-shared` (Vector Store, Embedder - 공유 모듈)

**Storage**:
- Qdrant (Vector DB) - 임베딩 저장 및 검색
- JSON 파일 (sources.json, documents.json, sync_history.json) - 상태 저장

**Testing**: pytest, pytest-asyncio, pytest-cov
**Target Platform**: Linux server (Docker 환경)
**Project Type**: Single project
**Performance Goals**:
- 100개 Notion 페이지 동기화 5분 이내 (SC-002)
- 증분 업데이트 시 80% 이상 시간 단축 (SC-004)

**Constraints**:
- Notion API 요청 한도 (3 req/sec) → Rate Limiting
- Swagger JSON 파싱 에러 핸들링
- 대용량 페이지 청킹 (최대 4000자 청크)

**Scale/Scope**:
- 초기 100-1000개 페이지 지원
- 월 $50 이하 운영 비용 (Constitution)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. 비용 효율성 우선 | ✅ PASS | 로컬 임베딩(sentence-transformers), Qdrant 오픈소스 사용 |
| II. 응답 시간 보장 | ✅ PASS | 인덱싱 서비스는 배치 작업, 응답 시간 SLA 해당 없음 |
| III. 정확도 우선 | ✅ PASS | 청킹/임베딩 품질 테스트 포함 |
| IV. 오픈소스 기반 | ✅ PASS | 모든 컴포넌트 오픈소스 (Qdrant, sentence-transformers) |
| V. 단순성 유지 | ✅ PASS | JSON 파일 기반 상태 저장, 최소한의 추상화 |
| VI. 보안 우선 | ✅ PASS | API 키 환경변수 관리, 읽기 전용 데이터 접근 |

**Quality Standards**:
- 단위 테스트 커버리지 80% 이상 ✅
- 통합 테스트 주요 기능별 최소 1개 ✅
- flake8/ruff 린팅 통과 ✅
- docstring Google 스타일 ✅

## Project Structure

### Documentation (this feature)

```text
specs/004-rag-indexer/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── search-api.yaml  # Search API OpenAPI spec
└── tasks.md             # Phase 2 output (by /speckit.tasks)
```

### Source Code (repository root)

```text
rag-indexer/
├── src/
│   ├── __init__.py
│   ├── cli.py                    # CLI entry point (typer)
│   ├── config.py                 # Settings (pydantic-settings)
│   ├── storage.py                # JSON file state storage
│   ├── logging_config.py         # Structured logging
│   ├── models/
│   │   ├── __init__.py
│   │   ├── source.py             # Source entity
│   │   ├── document.py           # Document entity
│   │   ├── chunk.py              # Chunk entity
│   │   └── sync_job.py           # SyncJob entity
│   ├── connectors/
│   │   ├── __init__.py
│   │   ├── notion.py             # Notion API connector
│   │   └── swagger.py            # Swagger/OpenAPI parser
│   ├── services/
│   │   ├── __init__.py
│   │   ├── chunker.py            # Text chunker
│   │   ├── embedder.py           # Embedding generator
│   │   ├── indexer.py            # Main indexing orchestrator
│   │   └── ai_extractor.py       # AI metadata extraction
│   ├── scheduler/
│   │   ├── __init__.py
│   │   └── sync_scheduler.py     # APScheduler wrapper
│   └── utils/
│       ├── __init__.py
│       ├── rate_limit.py         # Rate limiter
│       └── retry.py              # Retry decorator
├── tests/
│   ├── unit/
│   │   ├── test_chunker.py
│   │   ├── test_embedder.py
│   │   ├── test_indexer.py
│   │   ├── test_notion_connector.py
│   │   └── test_swagger_connector.py
│   └── integration/
│       ├── test_sync_pipeline.py
│       └── test_cli.py
└── pyproject.toml

shared/
├── shared/
│   ├── __init__.py
│   ├── embedding.py              # HuggingFace embedder
│   └── vector_store.py           # Qdrant client wrapper
└── tests/
```

**Structure Decision**: Single project 구조 선택. rag-indexer는 독립적인 CLI 서비스로, shared 모듈에서 임베딩/벡터DB 클라이언트를 공유합니다.

## Complexity Tracking

> **No violations - all Constitution checks PASS**

| Component | Complexity | Justification |
|-----------|------------|---------------|
| Hierarchical Chunking | Medium | Parent/Child 청크로 검색 정확도 향상. Constitution III 준수 |
| AI Metadata Extraction | Low | 선택적 기능, 비활성화 가능. 비용 최소화 |
| JSON File Storage | Low | 단순성 원칙 (V) 준수. SQLite 대신 선택 |

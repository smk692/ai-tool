# Tasks: RAG Document Indexer

**Input**: Design documents from `/specs/004-rag-indexer/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅, quickstart.md ✅

**Tests**: Test tasks included as specified in plan.md quality standards (80%+ unit test coverage)

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions

- **Project**: `rag-indexer/` (single project with shared module)
- **Source**: `rag-indexer/src/`
- **Tests**: `rag-indexer/tests/`
- **Shared Module**: `shared/shared/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 Create project structure per plan.md in rag-indexer/
- [x] T002 [P] Initialize Python project with pyproject.toml including all dependencies (notion-client, httpx, pydantic, langchain-text-splitters, apscheduler, typer, tenacity, anthropic)
- [x] T003 [P] Create rag-indexer/src/__init__.py with version info
- [x] T004 [P] Configure Ruff linting in rag-indexer/pyproject.toml
- [x] T005 [P] Create shared/shared/__init__.py module structure
- [x] T006 [P] Create test directory structure: rag-indexer/tests/unit/, rag-indexer/tests/integration/

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

### Core Configuration

- [x] T007 Implement Settings class with pydantic-settings in rag-indexer/src/config.py (Notion API key, Qdrant host/port, embedding model, chunk size/overlap)
- [x] T008 [P] Implement structured logging configuration in rag-indexer/src/logging_config.py

### Data Models (Shared by All Stories)

- [x] T009 [P] Create SourceType enum and Source model in rag-indexer/src/models/source.py (NotionSourceConfig, SwaggerSourceConfig)
- [x] T010 [P] Create Document model in rag-indexer/src/models/document.py
- [x] T011 [P] Create Chunk model in rag-indexer/src/models/chunk.py
- [x] T012 [P] Create SyncJobStatus enum and SyncJob model in rag-indexer/src/models/sync_job.py
- [x] T013 Create rag-indexer/src/models/__init__.py exporting all models

### Storage Layer

- [x] T014 Implement JSONFileStorage class in rag-indexer/src/storage.py (sources.json, documents.json, sync_history.json management)

### Shared Module - Embedding & Vector Store

- [x] T015 [P] Implement Embedder class using sentence-transformers in shared/shared/embedding.py (intfloat/multilingual-e5-large-instruct model, batch encoding)
- [x] T016 [P] Implement VectorStoreClient class for Qdrant in shared/shared/vector_store.py (collection management, upsert, search, delete)

### Utility Functions

- [x] T017 [P] Implement RateLimiter class in rag-indexer/src/utils/rate_limit.py (for Notion API 3 req/sec limit)
- [x] T018 [P] Implement retry decorator with tenacity in rag-indexer/src/utils/retry.py
- [x] T019 Create rag-indexer/src/utils/__init__.py exporting utilities

### Core Services

- [x] T020 Implement TextChunker class using RecursiveCharacterTextSplitter in rag-indexer/src/services/chunker.py (chunk_size=1000, chunk_overlap=200)
- [x] T021 Implement EmbeddingService wrapper in rag-indexer/src/services/embedder.py (wraps shared Embedder with passage/query prefix handling)
- [x] T022 Create rag-indexer/src/services/__init__.py and rag-indexer/src/connectors/__init__.py

### Unit Tests for Foundational Components

- [x] T023 [P] Write unit tests for chunker in rag-indexer/tests/unit/test_chunker.py
- [x] T024 [P] Write unit tests for embedder in rag-indexer/tests/unit/test_embedder.py
- [x] T025 [P] Write unit tests for storage in rag-indexer/tests/unit/test_storage.py

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Notion 문서 동기화 (Priority: P1) 🎯 MVP

**Goal**: 운영자가 Notion 워크스페이스의 페이지와 데이터베이스를 벡터DB에 동기화하여, 나중에 Slack 챗봇이 해당 정보를 검색할 수 있게 합니다.

**Independent Test**: Notion API 키와 페이지 ID만 있으면 독립적으로 테스트 가능. 동기화 후 벡터DB에서 직접 검색하여 결과 확인 가능.

**Acceptance Criteria**:
1. Notion API 키와 대상 페이지 ID로 동기화 명령 실행 시 해당 페이지의 모든 텍스트 콘텐츠가 벡터DB에 저장됨
2. Notion 데이터베이스 ID로 동기화 명령 실행 시 모든 항목이 개별 문서로 벡터DB에 저장됨
3. 이미 동기화된 페이지가 변경된 후 재동기화 시 변경된 내용만 업데이트됨

### Implementation for User Story 1

- [x] T026 [US1] Implement NotionConnector class in rag-indexer/src/connectors/notion.py (API client, page retrieval, block extraction for P1+P2 block types)
- [x] T027 [US1] Implement rich_text extraction functions for supported Notion block types in rag-indexer/src/connectors/notion.py
- [x] T028 [US1] Add recursive children block fetching to NotionConnector in rag-indexer/src/connectors/notion.py
- [x] T029 [US1] Implement database item retrieval in NotionConnector in rag-indexer/src/connectors/notion.py
- [x] T030 [US1] Add content hash computation (SHA256) for change detection in rag-indexer/src/connectors/notion.py
- [x] T031 [US1] Implement IndexerService class in rag-indexer/src/services/indexer.py (orchestrates chunking, embedding, vector storage)
- [x] T032 [US1] Add Notion source sync method to IndexerService in rag-indexer/src/services/indexer.py
- [x] T033 [US1] Implement incremental update logic (hash comparison, skip unchanged) in rag-indexer/src/services/indexer.py

### Tests for User Story 1

- [x] T034 [P] [US1] Write unit tests for NotionConnector in rag-indexer/tests/unit/test_notion_connector.py (mock Notion API responses)
- [x] T035 [P] [US1] Write unit tests for IndexerService in rag-indexer/tests/unit/test_indexer.py

### Integration Test for User Story 1

- [x] T036 [US1] Write integration test for Notion sync pipeline in rag-indexer/tests/integration/test_sync_pipeline.py

**Checkpoint**: User Story 1 (Notion 동기화) is fully functional and testable independently

---

## Phase 4: User Story 2 - Swagger API 문서 등록 (Priority: P2)

**Goal**: 운영자가 외부 URL에서 Swagger/OpenAPI 문서를 가져와 벡터DB에 등록하여, API 관련 질문에 답변할 수 있게 합니다.

**Independent Test**: Swagger URL만 있으면 독립적으로 테스트 가능. 등록 후 API 엔드포인트 검색으로 결과 확인.

**Acceptance Criteria**:
1. 유효한 Swagger JSON URL로 등록 명령 실행 시 모든 API 엔드포인트가 벡터DB에 저장됨
2. OpenAPI 3.0 형식 문서 등록 시 각 엔드포인트의 경로, 메소드, 파라미터, 응답 스키마가 검색 가능한 형태로 저장됨
3. 이미 등록된 Swagger 문서를 새 버전 URL로 재등록 시 기존 데이터가 새 버전으로 교체됨

### Implementation for User Story 2

- [x] T037 [P] [US2] Implement SwaggerConnector class in rag-indexer/src/connectors/swagger.py (URL fetch with httpx, JSON parsing)
- [x] T038 [US2] Add OpenAPI 2.0 (Swagger) parsing support in rag-indexer/src/connectors/swagger.py
- [x] T039 [US2] Add OpenAPI 3.0/3.1 parsing support in rag-indexer/src/connectors/swagger.py
- [x] T040 [US2] Implement endpoint extraction and text formatting in rag-indexer/src/connectors/swagger.py (path, method, params, response schema)
- [x] T041 [US2] Add Swagger source sync method to IndexerService in rag-indexer/src/services/indexer.py
- [x] T042 [US2] Implement Swagger document replacement logic (delete old → insert new) in rag-indexer/src/services/indexer.py

### Tests for User Story 2

- [x] T043 [P] [US2] Write unit tests for SwaggerConnector in rag-indexer/tests/unit/test_swagger_connector.py (mock HTTP responses, test both OpenAPI 2.0 and 3.0)

**Checkpoint**: User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - 스케줄러 자동 동기화 (Priority: P3)

**Goal**: 운영자가 주기적인 자동 동기화를 설정하여, 수동 개입 없이 최신 문서가 벡터DB에 반영되도록 합니다.

**Independent Test**: 스케줄러 설정 후 지정된 시간에 동기화가 실행되는지 로그로 확인.

**Acceptance Criteria**:
1. 동기화 스케줄 "매일 오전 6시" 설정 시 해당 시간에 모든 등록된 소스가 자동 동기화됨
2. 스케줄러 동기화 중 특정 소스에서 에러 발생 시 다른 소스는 계속 처리되고 에러 내용이 기록됨

### Implementation for User Story 3

- [x] T044 [US3] Implement SyncScheduler class using APScheduler in rag-indexer/src/scheduler/sync_scheduler.py (AsyncIOScheduler, CronTrigger)
- [x] T045 [US3] Add cron schedule configuration support in rag-indexer/src/scheduler/sync_scheduler.py
- [x] T046 [US3] Implement error isolation (continue processing other sources on error) in rag-indexer/src/scheduler/sync_scheduler.py
- [x] T047 [US3] Add SyncJob logging and history recording in rag-indexer/src/scheduler/sync_scheduler.py
- [x] T048 [US3] Create rag-indexer/src/scheduler/__init__.py exporting SyncScheduler

**Checkpoint**: Scheduler auto-sync is functional and testable

---

## Phase 6: User Story 4 - 수동 트리거 CLI (Priority: P4)

**Goal**: 운영자가 CLI 명령어로 특정 소스만 선택적으로 동기화하거나 전체 동기화를 실행합니다.

**Independent Test**: CLI 명령어 실행으로 독립 테스트 가능.

**Acceptance Criteria**:
1. `rag-indexer sync --source notion` 명령 실행 시 Notion 소스만 동기화됨
2. `rag-indexer sync --all` 명령 실행 시 모든 소스가 동기화됨
3. `rag-indexer status` 명령 실행 시 각 소스의 마지막 동기화 시간과 문서 수가 표시됨

### Implementation for User Story 4

- [x] T049 [US4] Implement CLI entry point using typer in rag-indexer/src/cli.py (main app setup)
- [x] T050 [US4] Implement `source add` command in rag-indexer/src/cli.py (--name, --type, --page-id, --database-id, --url options)
- [x] T051 [US4] Implement `source list` command in rag-indexer/src/cli.py (table/json format output)
- [x] T052 [US4] Implement `source remove` command in rag-indexer/src/cli.py (--name, --id, --force options)
- [x] T053 [US4] Implement `sync` command in rag-indexer/src/cli.py (--all, --source, --source-id, --force, --dry-run, --verbose options)
- [x] T054 [US4] Implement `status` command in rag-indexer/src/cli.py (source status table, vector DB stats)
- [x] T055 [US4] Implement `status history` command in rag-indexer/src/cli.py (--limit, --source options)
- [x] T056 [US4] Implement `scheduler start/stop/status` commands in rag-indexer/src/cli.py
- [x] T057 [US4] Add global options (--config, --log-level, --quiet, --version) in rag-indexer/src/cli.py

### Tests for User Story 4

- [x] T058 [US4] Write integration tests for CLI commands in rag-indexer/tests/integration/test_cli.py

**Checkpoint**: All user stories should now be independently functional

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [x] T059 [P] Add AI metadata extraction service (optional) in rag-indexer/src/services/ai_extractor.py (content type, topics, difficulty classification using Anthropic API)
- [x] T060 [P] Add search method to IndexerService in rag-indexer/src/services/indexer.py (implementing search-api.yaml contract)
- [x] T061 [P] Add hierarchical search with parent context in rag-indexer/src/services/indexer.py
- [x] T062 [P] Add collection stats method in rag-indexer/src/services/indexer.py
- [x] T063 Code cleanup and docstring completion (Google style) across all modules
- [ ] T064 Run quickstart.md validation end-to-end
- [ ] T065 Ensure test coverage ≥80% with pytest-cov

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-6)**: All depend on Foundational phase completion
  - User stories can then proceed in priority order (P1 → P2 → P3 → P4)
  - US2, US3, US4 can be worked on in parallel after US1 if team allows
- **Polish (Phase 7)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Shares IndexerService with US1
- **User Story 3 (P3)**: Depends on IndexerService from US1 - Uses sync functionality
- **User Story 4 (P4)**: Depends on all previous stories - CLI wraps all functionality

### Within Each User Story

- Models before services
- Connectors before indexer integration
- Core implementation before tests
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational model tasks (T009-T012) marked [P] can run in parallel
- Shared module tasks (T015-T016) can run in parallel
- Utility tasks (T017-T018) can run in parallel
- Foundational unit tests (T023-T025) can run in parallel
- US2 SwaggerConnector (T037) can start in parallel with US1 after Foundational
- Polish phase tasks (T059-T062) marked [P] can run in parallel

---

## Parallel Example: Foundational Phase

```bash
# Launch all model tasks together:
Task: "Create SourceType enum and Source model in rag-indexer/src/models/source.py"
Task: "Create Document model in rag-indexer/src/models/document.py"
Task: "Create Chunk model in rag-indexer/src/models/chunk.py"
Task: "Create SyncJobStatus enum and SyncJob model in rag-indexer/src/models/sync_job.py"

# Launch shared module tasks together:
Task: "Implement Embedder class using sentence-transformers in shared/shared/embedding.py"
Task: "Implement VectorStoreClient class for Qdrant in shared/shared/vector_store.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1 (Notion 동기화)
4. **STOP and VALIDATE**: Test Notion sync independently
5. Deploy/demo if ready - This is the MVP!

### Incremental Delivery

1. Setup + Foundational → Foundation ready
2. Add User Story 1 (Notion) → Test independently → **MVP Delivery!**
3. Add User Story 2 (Swagger) → Test independently → Expanded sources
4. Add User Story 3 (Scheduler) → Test independently → Automation
5. Add User Story 4 (CLI) → Test independently → Full operational control
6. Polish → Production-ready

### Suggested MVP Scope

**MVP = Phase 1 + Phase 2 + Phase 3 (User Story 1)**
- Total Tasks for MVP: 36 tasks (T001-T036)
- Delivers: Notion document sync to vector DB with incremental updates
- Verifiable: Sync Notion pages → Search in Qdrant → Confirm results

---

## Summary

| Phase | Task Count | Description |
|-------|------------|-------------|
| Phase 1: Setup | 6 | Project initialization |
| Phase 2: Foundational | 19 | Core infrastructure (blocking) |
| Phase 3: US1 - Notion | 11 | MVP - Notion sync |
| Phase 4: US2 - Swagger | 7 | API document support |
| Phase 5: US3 - Scheduler | 5 | Auto-sync |
| Phase 6: US4 - CLI | 10 | Manual control |
| Phase 7: Polish | 7 | Production-readiness |
| **Total** | **65** | |

### Task Count by User Story

| User Story | Task Count | Independent Test |
|------------|------------|------------------|
| US1 - Notion | 11 | ✅ Notion API + Qdrant |
| US2 - Swagger | 7 | ✅ Swagger URL + Qdrant |
| US3 - Scheduler | 5 | ✅ Cron logs |
| US4 - CLI | 10 | ✅ CLI commands |

### Parallel Opportunities

- Phase 1: 5 tasks can run in parallel
- Phase 2: 12 tasks can run in parallel (models, shared, utilities, tests)
- Phase 3-6: Tests can run in parallel within each phase
- Phase 7: 4 tasks can run in parallel

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence

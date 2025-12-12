# Tasks: Slack RAG Chatbot

**Input**: Design documents from `/specs/005-rag-chatbot/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/slack-events.md ✅, quickstart.md ✅

**Tests**: 명시적으로 요청되지 않음 - 테스트 태스크 제외

**Organization**: 태스크는 유저 스토리별로 그룹화되어 독립적인 구현 및 테스트가 가능합니다.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: 병렬 실행 가능 (다른 파일, 종속성 없음)
- **[Story]**: 해당 유저 스토리 (예: US1, US2, US3, US4)
- 설명에 정확한 파일 경로 포함

## Path Conventions

- **프로젝트 구조**: `rag-chatbot/src/`, `rag-chatbot/tests/`
- shared 모듈 의존: `shared/` (임베딩, 벡터 스토어)

---

## Phase 1: Setup (프로젝트 초기화)

**Purpose**: 프로젝트 구조 생성 및 기본 설정

- [ ] T001 Create rag-chatbot project structure per plan.md (rag-chatbot/src/, rag-chatbot/tests/)
- [ ] T002 Initialize pyproject.toml with dependencies: slack-bolt>=1.18.0, anthropic>=0.40.0, redis>=5.0.0, pydantic>=2.0.0, tenacity
- [ ] T003 [P] Create .env.example with all required environment variables per quickstart.md (rag-chatbot/.env.example)
- [ ] T004 [P] Configure ruff linting and formatting in pyproject.toml (rag-chatbot/pyproject.toml)
- [ ] T005 [P] Create __init__.py files for all package directories

---

## Phase 2: Foundational (핵심 인프라)

**Purpose**: 모든 유저 스토리 구현 전에 완료되어야 하는 핵심 인프라

**⚠️ CRITICAL**: 이 페이즈가 완료되기 전까지 유저 스토리 작업 불가

- [ ] T006 Create config.py with environment settings in rag-chatbot/src/config.py
- [ ] T007 [P] Create Query model with Pydantic validation in rag-chatbot/src/models/query.py
- [ ] T008 [P] Create SearchResult model in rag-chatbot/src/models/search_result.py
- [ ] T009 [P] Create SourceReference and Response models in rag-chatbot/src/models/response.py
- [ ] T010 [P] Create ConversationMessage and Conversation models in rag-chatbot/src/models/conversation.py
- [ ] T011 [P] Create Feedback model with reaction mapping in rag-chatbot/src/models/feedback.py
- [ ] T012 Create models/__init__.py exporting all models in rag-chatbot/src/models/__init__.py
- [ ] T013 Implement ClaudeClient with streaming support in rag-chatbot/src/llm/claude_client.py
- [ ] T014 [P] Create prompt templates for RAG responses in rag-chatbot/src/llm/prompts.py
- [ ] T015 Create llm/__init__.py exporting ClaudeClient in rag-chatbot/src/llm/__init__.py
- [ ] T016 Implement sensitive info detection guardrails in rag-chatbot/src/guardrails/sensitive.py
- [ ] T017 Create guardrails/__init__.py in rag-chatbot/src/guardrails/__init__.py
- [ ] T018 Initialize Slack Bolt app structure in rag-chatbot/src/main.py

**Checkpoint**: 기본 인프라 준비 완료 - 유저 스토리 구현 시작 가능

---

## Phase 3: User Story 1 - Slack 메시지 질문 응답 (Priority: P1) 🎯 MVP

**Goal**: 사용자가 Slack에서 질문하면 벡터DB 검색 + Claude LLM으로 답변 생성

**Independent Test**: Slack 봇 토큰과 Claude API 키가 있으면 멘션 또는 DM으로 질문하여 답변 수신 확인

### Implementation for User Story 1

- [ ] T019 [US1] Implement RAGService with vector search integration (threshold=0.7) in rag-chatbot/src/services/rag_service.py
- [ ] T020 [US1] Create services/__init__.py exporting RAGService in rag-chatbot/src/services/__init__.py
- [ ] T021 [US1] Implement mention event handler per slack-events.md contract in rag-chatbot/src/handlers/mention.py
- [ ] T022 [US1] Implement DM event handler per slack-events.md contract in rag-chatbot/src/handlers/dm.py
- [ ] T023 [US1] Create handlers/__init__.py exporting all handlers in rag-chatbot/src/handlers/__init__.py
- [ ] T024 [US1] Integrate handlers with main.py Slack Bolt app in rag-chatbot/src/main.py
- [ ] T025 [US1] Add error handling for Slack/Claude API failures with retry logic in handlers
- [ ] T026 [US1] Add fallback response for no search results scenario
- [ ] T027 [US1] Add guardrails integration to detect sensitive info in questions
- [ ] T028 [US1] Add logging for question/answer flow

**Checkpoint**: User Story 1 완료 - Slack에서 질문 → 답변 받기 가능

---

## Phase 4: User Story 2 - 컨텍스트 기반 대화 (Priority: P2)

**Goal**: 동일 스레드에서 이전 대화 맥락을 유지하여 연속 대화 가능

**Independent Test**: 동일 스레드에서 후속 질문 시 이전 대화 컨텍스트 반영 확인

### Implementation for User Story 2

- [ ] T029 [US2] Implement ConversationService with Redis storage in rag-chatbot/src/services/conversation.py
- [ ] T030 [US2] Add Redis connection management to config.py
- [ ] T031 [US2] Integrate ConversationService with RAGService for context-aware responses
- [ ] T032 [US2] Update mention handler to load/save conversation context
- [ ] T033 [US2] Update DM handler to load/save conversation context
- [ ] T034 [US2] Implement message limit (max 5) and TTL (1 hour) management
- [ ] T035 [US2] Add conversation context to Claude prompt template

**Checkpoint**: User Story 2 완료 - 스레드 내 연속 대화 가능

---

## Phase 5: User Story 3 - 검색 결과 출처 표시 (Priority: P3)

**Goal**: 답변에 참조 문서의 출처를 함께 표시

**Independent Test**: 답변 메시지에 "📚 참조 문서:" 섹션과 문서 링크 표시 확인

### Implementation for User Story 3

- [ ] T036 [US3] Enhance Response.format_for_slack() to include source references with links and handle 4000 char limit (auto-split)
- [ ] T037 [US3] Update SearchResult to include source_url from vector DB metadata
- [ ] T038 [US3] Update RAGService to collect and deduplicate source references
- [ ] T039 [US3] Format Notion URLs as clickable Slack links (<url|title>)
- [ ] T040 [US3] Handle cases with multiple sources (list format)

**Checkpoint**: User Story 3 완료 - 답변에 출처 표시

---

## Phase 6: User Story 4 - 답변 품질 피드백 (Priority: P4)

**Goal**: 사용자가 답변에 리액션으로 피드백 제공

**Independent Test**: 봇 답변에 👍/👎 리액션 추가 후 Redis에서 피드백 데이터 확인

### Implementation for User Story 4

- [ ] T041 [US4] Implement FeedbackService with Redis storage in rag-chatbot/src/services/feedback.py
- [ ] T042 [US4] Implement reaction event handler per slack-events.md contract in rag-chatbot/src/handlers/feedback.py
- [ ] T043 [US4] Integrate feedback handler with main.py Slack Bolt app
- [ ] T044 [US4] Store original question with feedback (retrieve from thread)
- [ ] T045 [US4] Add JSON export functionality for feedback data (optional backup)
- [ ] T046 [US4] Update services/__init__.py to export FeedbackService

**Checkpoint**: User Story 4 완료 - 피드백 수집 가능

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: 전체 시스템 완성도 향상

- [ ] T047 [P] Update Makefile with install-chatbot, test-chatbot targets at repo root
- [ ] T048 [P] Add rag-chatbot service to docker-compose.yml in infra/docker/
- [ ] T049 Create rag-chatbot/src/__init__.py with version info
- [ ] T050 Validate full flow per quickstart.md test scenarios
- [ ] T051 [P] Add type hints to all public functions
- [ ] T052 [P] Add docstrings to all public classes and methods
- [ ] T053 Run ruff lint and fix any issues
- [ ] T054 Performance validation: response time under 10 seconds

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: 의존성 없음 - 즉시 시작 가능
- **Foundational (Phase 2)**: Setup 완료 필요 - 모든 유저 스토리 차단
- **User Stories (Phase 3+)**: Foundational 완료 필요
  - 유저 스토리는 우선순위 순서로 순차 진행 (P1 → P2 → P3 → P4)
  - US2는 US1의 핸들러에 컨텍스트 로직 추가
  - US3는 US1의 Response 포맷 확장
  - US4는 US1과 독립적으로 구현 가능
- **Polish (Phase 7)**: 원하는 유저 스토리 완료 후 진행

### User Story Dependencies

- **User Story 1 (P1)**: Foundational 완료 후 시작 - MVP, 다른 스토리 의존 없음
- **User Story 2 (P2)**: US1 완료 권장 - 핸들러 확장
- **User Story 3 (P3)**: US1 완료 권장 - Response 포맷 확장
- **User Story 4 (P4)**: US1 완료 필요 - 봇 응답에 리액션

### Within Each User Story

- 서비스 먼저, 핸들러 다음
- 핵심 기능 먼저, 에러 처리 다음
- 로깅은 마지막

### Parallel Opportunities

- Phase 1: T003, T004, T005 병렬 가능
- Phase 2: 모든 모델 (T007-T011) 병렬 가능, 프롬프트 (T014) 병렬 가능
- Phase 7: 대부분의 태스크 병렬 가능

---

## Parallel Example: Phase 2 Models

```bash
# 모든 모델 동시 생성:
Task: "Create Query model in rag-chatbot/src/models/query.py"
Task: "Create SearchResult model in rag-chatbot/src/models/search_result.py"
Task: "Create Response model in rag-chatbot/src/models/response.py"
Task: "Create Conversation model in rag-chatbot/src/models/conversation.py"
Task: "Create Feedback model in rag-chatbot/src/models/feedback.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Phase 1: Setup 완료
2. Phase 2: Foundational 완료 (CRITICAL)
3. Phase 3: User Story 1 완료
4. **STOP and VALIDATE**: Slack에서 질문 → 답변 테스트
5. 필요시 배포/데모

### Incremental Delivery

1. Setup + Foundational → 기반 준비
2. User Story 1 → 독립 테스트 → 배포/데모 (MVP!)
3. User Story 2 → 독립 테스트 → 배포/데모
4. User Story 3 → 독립 테스트 → 배포/데모
5. User Story 4 → 독립 테스트 → 배포/데모
6. 각 스토리가 이전 기능 유지하며 가치 추가

---

## Notes

- [P] 태스크 = 다른 파일, 종속성 없음
- [Story] 라벨 = 특정 유저 스토리에 매핑
- 각 유저 스토리는 독립적으로 완료 및 테스트 가능
- 논리적 그룹 또는 태스크 완료 후 커밋
- 체크포인트에서 스토리 독립 검증 가능
- shared 모듈(EmbeddingClient, QdrantStore)은 이미 구현됨 - 재사용

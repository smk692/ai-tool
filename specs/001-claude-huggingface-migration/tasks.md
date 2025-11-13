# Tasks: LLM Migration to Claude Code + Hugging Face Embeddings

**Branch**: `001-claude-huggingface-migration`
**Input**: Design documents from `/specs/001-claude-huggingface-migration/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/llm-service.yaml

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `- [ ] [ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure) ✅ COMPLETE

**Purpose**: Project initialization and basic structure

- [x] T001 Create project structure per implementation plan in plan.md
- [x] T002 Initialize Python 3.10+ project with requirements.txt
- [x] T003 [P] Configure pytest testing framework and coverage tools
- [x] T004 [P] Setup pre-commit hooks for linting (ruff, black)
- [x] T005 [P] Create .env.example template with all required environment variables
- [x] T006 [P] Update .gitignore to exclude .env, __pycache__, ChromaDB data
- [x] T007 Create config/settings.py for centralized configuration management

---

## Phase 2: Foundational (Blocking Prerequisites) ✅ COMPLETE

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T008 Setup ChromaDB initialization in src/services/vector_store.py
- [x] T009 [P] Configure PostgreSQL read-only connection in src/db/postgres.py
- [x] T010 [P] Configure SQLite conversation memory in src/db/sqlite.py
- [x] T011 [P] Create base LLM configuration entity in src/models/llm_config.py
- [x] T012 [P] Create embedding configuration entity in src/models/embedding.py
- [x] T013 [P] Implement centralized error handling in src/utils/errors.py
- [x] T014 [P] Setup logging configuration in src/utils/logging.py
- [x] T015 [P] Create prompt template utilities in src/utils/prompts.py
- [x] T016 Download and cache paraphrase-multilingual-MiniLM-L12-v2 model in scripts/download_embedding_model.py
- [x] T017 Initialize ChromaDB collection with 384 dimensions in scripts/init_vector_store.py

**Checkpoint**: ✅ Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Developer Integrates Claude Code API (Priority: P1) 🎯 MVP

**Goal**: Replace OpenAI GPT-4o with Claude Code API for all LLM operations while maintaining response quality and Korean language support

**Independent Test**: Send a Korean natural language query (e.g., "지난달 신규 가입자 수는?") and verify Claude Code generates a valid SQL query response within 30-60 second SLA

### Implementation for User Story 1 (🔄 65% Complete)

- [x] T018 [P] [US1] Install anthropic and langchain-anthropic packages in requirements.txt
- [x] T019 [P] [US1] Remove openai and langchain-openai dependencies from requirements.txt
- [x] T020 [P] [US1] Create Claude API client wrapper in src/services/llm_client.py
- [x] T021 [P] [US1] Create query request entity in src/models/query_request.py
- [x] T022 [P] [US1] Create query response entity in src/models/query_response.py
- [x] T023 [US1] Update Text-to-SQL chain with ChatAnthropic in src/chains/text_to_sql.py
- [x] T024 [US1] Update Knowledge Discovery chain with ChatAnthropic in src/chains/knowledge.py
- [x] T025 [US1] Update Router chain with ChatAnthropic in src/chains/router.py
- [x] T026 [US1] Update Multi-turn conversation chain with ChatAnthropic in src/chains/multi_turn.py
- [x] T027 [US1] Convert prompts to Claude XML format in src/utils/prompts.py
- [x] T028 [US1] Update LLM configuration to use Claude 3.5 Sonnet in config/settings.py
- [x] T029 [US1] Create API connection test script in scripts/test_claude_connection.py
- [x] T030 [US1] Add error handling for Claude API failures in src/services/llm_client.py
- [x] T031 [US1] Update token usage tracking for Anthropic format in src/models/query_response.py
- [ ] T032 [P] [US1] Create unit tests for LLM client in tests/unit/test_llm_client.py
- [ ] T033 [P] [US1] Create unit tests for updated chains in tests/unit/test_chains.py
- [ ] T034 [P] [US1] Create E2E test for Text-to-SQL workflow in tests/integration/test_text_to_sql_e2e.py
- [ ] T035 [P] [US1] Create E2E test for Knowledge Discovery workflow in tests/integration/test_knowledge_e2e.py
- [ ] T036 [US1] Create Korean language test fixtures in tests/fixtures/sample_queries.json
- [ ] T037 [US1] Create mock Claude responses in tests/fixtures/mock_responses.json
- [ ] T038 [US1] Verify all tests pass with ≥80% coverage
- [ ] T039 [US1] Update environment variables in config/.env.example with ANTHROPIC_API_KEY
- [ ] T040 [US1] Remove OPENAI_API_KEY references from all configuration files

**Checkpoint**: At this point, User Story 1 should be fully functional - Claude Code integration complete and testable independently

---

## Phase 4: User Story 2 - System Uses High-Quality Embeddings (Priority: P2)

**Goal**: Maintain high-quality Hugging Face embedding model (paraphrase-multilingual-MiniLM-L12-v2) to ensure document retrieval accuracy meets ≥90% Top-5 target

**Independent Test**: Run benchmark test set of 100 Korean queries against knowledge base and measure Top-5 accuracy achieving ≥90%

### Implementation for User Story 2

- [ ] T041 [P] [US2] Verify paraphrase-multilingual-MiniLM-L12-v2 model configuration in src/models/embedding.py
- [ ] T042 [P] [US2] Implement embedding service in src/services/embeddings.py
- [ ] T043 [US2] Integrate embedding service with ChromaDB in src/services/vector_store.py
- [ ] T044 [US2] Create document indexing utility in scripts/index_documents.py
- [ ] T045 [US2] Test Korean language query embedding in tests/unit/test_embeddings.py
- [ ] T046 [US2] Validate vector search latency ≤0.5 seconds in tests/integration/test_vector_search.py
- [ ] T047 [US2] Create benchmark test suite for Top-5 accuracy in tests/benchmarks/test_embedding_accuracy.py
- [ ] T048 [US2] Validate multilingual support (Korean + English) in tests/unit/test_embeddings.py
- [ ] T049 [US2] Document embedding model specifications in docs/embedding-model.md

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently - embedding quality validated

---

## Phase 5: User Story 3 - Constitutional Budget Approval and Monitoring (Priority: P3)

**Goal**: Update constitutional budget to $100/month and implement monitoring to ensure costs remain sustainable and transparent

**Independent Test**: Run system for one week, verify daily cost reports are generated, and cumulative cost projection stays within approved monthly budget

### Implementation for User Story 3

- [ ] T050 [P] [US3] Create budget tracking entity in src/models/budget.py
- [ ] T051 [P] [US3] Implement cost tracker service in src/services/cost_tracker.py
- [ ] T052 [US3] Add budget tracking to query processing pipeline in src/chains/router.py
- [ ] T053 [US3] Create daily cost monitoring script in scripts/monitor_daily_costs.py
- [ ] T054 [US3] Implement budget alert system in src/utils/monitoring.py
- [ ] T055 [US3] Add cost dashboard to Streamlit UI in src/ui/app.py
- [ ] T056 [US3] Create budget status API endpoint in src/api/budget.py (if using FastAPI backend)
- [ ] T057 [US3] Update constitution.md with version 2.0.0 and $100 budget
- [ ] T058 [P] [US3] Create unit tests for budget tracking in tests/unit/test_cost_tracker.py
- [ ] T059 [P] [US3] Create integration test for budget monitoring in tests/integration/test_budget_monitoring.py
- [ ] T060 [US3] Configure alert thresholds (80%, 90%, 100%) in config/settings.py
- [ ] T061 [US3] Validate alert notifications work correctly
- [ ] T062 [US3] Document budget monitoring in docs/cost-monitoring.md

**Checkpoint**: All user stories should now be independently functional - complete migration with budget monitoring

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T063 [P] Update README.md with Claude Code migration details
- [ ] T064 [P] Create migration guide in docs/migration-guide.md
- [ ] T065 [P] Update quickstart.md validation script
- [ ] T066 [P] Update CLAUDE.md with Python 3.10+ and anthropic package
- [ ] T067 Code cleanup - remove all OpenAI references from codebase
- [ ] T068 Performance optimization - validate all SLA targets (Text-to-SQL ≤60s, Knowledge ≤3s, Vector ≤0.5s)
- [ ] T069 Security hardening - verify API key management and error messages don't expose sensitive data
- [ ] T070 [P] Run full test suite and ensure ≥80% coverage
- [ ] T071 [P] Create rollback plan documentation in docs/rollback-plan.md
- [ ] T072 Run quickstart.md validation with fresh environment
- [ ] T073 Create production deployment checklist in docs/deployment-checklist.md

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-5)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Phase 6)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
  - Core LLM migration - highest priority
  - Delivers immediate value: system operates with Claude Code

- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Independent of US1
  - Embedding quality validation
  - Can be tested with existing embedding model while US1 proceeds
  - May integrate with US1's Knowledge Discovery chain

- **User Story 3 (P3)**: Can start after US1 is complete (needs query processing for tracking)
  - Budget monitoring depends on LLM being operational
  - Should start after US1 basic functionality verified
  - Independent testing through cost tracking reports

### Within Each User Story

**User Story 1 (Claude Integration)**:
1. Package updates (T018-T019)
2. Entity models in parallel (T021-T022)
3. LLM client implementation (T020)
4. Chain updates (T023-T026) - can run in parallel
5. Prompt conversion (T027)
6. Configuration (T028)
7. Testing (T032-T038) - can run in parallel
8. Cleanup (T039-T040)

**User Story 2 (Embedding Quality)**:
1. Model verification and service implementation (T041-T043) - can run in parallel
2. Indexing utility (T044)
3. Testing (T045-T048) - can run in parallel
4. Documentation (T049)

**User Story 3 (Budget Monitoring)**:
1. Entity and service (T050-T051) - can run in parallel
2. Integration (T052)
3. Monitoring infrastructure (T053-T055)
4. API endpoint (T056)
5. Constitution update (T057)
6. Testing (T058-T059) - can run in parallel
7. Configuration and validation (T060-T062)

### Parallel Opportunities

#### Phase 1 (Setup)
- T003, T004, T005, T006 can all run in parallel (different configuration files)

#### Phase 2 (Foundational)
- T009, T010, T011, T012, T013, T014, T015 can all run in parallel (different files)

#### User Story 1 (Claude Integration)
- T018-T019: Package updates in parallel
- T021-T022: Entity models in parallel
- T023-T026: All chain updates in parallel (different files)
- T032-T035: All tests in parallel (different test files)

#### User Story 2 (Embedding Quality)
- T041-T043: Model verification and service in parallel
- T045-T048: All tests in parallel

#### User Story 3 (Budget Monitoring)
- T050-T051: Entity and service in parallel
- T058-T059: Tests in parallel

#### Phase 6 (Polish)
- T063, T064, T065, T066, T070, T071 can all run in parallel (different documentation files)

---

## Parallel Example: User Story 1 (Claude Integration)

```bash
# Launch all chain updates together:
Task T023: "Update Text-to-SQL chain with ChatAnthropic in src/chains/text_to_sql.py"
Task T024: "Update Knowledge Discovery chain with ChatAnthropic in src/chains/knowledge.py"
Task T025: "Update Router chain with ChatAnthropic in src/chains/router.py"
Task T026: "Update Multi-turn conversation chain with ChatAnthropic in src/chains/multi_turn.py"

# Launch all tests together:
Task T032: "Create unit tests for LLM client in tests/unit/test_llm_client.py"
Task T033: "Create unit tests for updated chains in tests/unit/test_chains.py"
Task T034: "Create E2E test for Text-to-SQL workflow in tests/integration/test_text_to_sql_e2e.py"
Task T035: "Create E2E test for Knowledge Discovery workflow in tests/integration/test_knowledge_e2e.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T007)
2. Complete Phase 2: Foundational (T008-T017) - CRITICAL
3. Complete Phase 3: User Story 1 (T018-T040)
4. **STOP and VALIDATE**:
   - Test Korean Text-to-SQL query
   - Test Knowledge Discovery query
   - Verify response times within SLA
   - Verify ≥80% test coverage
5. Deploy/demo if ready

**Estimated Time**: 10-14 hours total
- Setup: 1-2 hours
- Foundational: 2-3 hours
- User Story 1: 6-8 hours
- Validation: 1 hour

### Incremental Delivery

1. **Foundation** (Phases 1-2): Setup + Foundational → Environment ready (3-5 hours)
2. **MVP Release** (Phase 3): User Story 1 → Claude integration complete → Deploy/Demo (6-8 hours)
3. **Enhanced Quality** (Phase 4): User Story 2 → Embedding validation → Deploy/Demo (2-3 hours)
4. **Sustainability** (Phase 5): User Story 3 → Budget monitoring → Deploy/Demo (3-4 hours)
5. **Production Ready** (Phase 6): Polish → Final validation → Production deployment (2-3 hours)

**Total Estimated Time**: 16-23 hours for complete migration

### Parallel Team Strategy

With multiple developers:

1. **All team members together**: Complete Setup + Foundational (Phases 1-2)
2. **Once Foundational is done** (after T017):
   - **Developer A**: User Story 1 (T018-T040) - Claude integration
   - **Developer B**: User Story 2 (T041-T049) - Embedding validation (can start in parallel)
   - **Developer C**: Prepare User Story 3 infrastructure
3. **After US1 completes**:
   - **Developer A**: Help with testing and validation
   - **Developer C**: User Story 3 (T050-T062) - Budget monitoring
4. **All team members**: Phase 6 Polish (T063-T073)

**With parallel execution**: Total time could be reduced to 12-16 hours

---

## Migration Risk Mitigation

### Pre-Migration Checklist

- [ ] Backup current OpenAI configuration
- [ ] Create rollback Git branch (`backup-openai`)
- [ ] Verify Claude API key is valid and active
- [ ] Confirm PostgreSQL read-only access
- [ ] Test ChromaDB initialization in dev environment
- [ ] Review all test fixtures for Korean language queries

### Critical Validation Points

**After Phase 2 (Foundational)**:
- [ ] ChromaDB collection created with 384 dimensions
- [ ] Embedding model downloaded and cached
- [ ] Database connections working (PostgreSQL read-only, SQLite)

**After Phase 3 (User Story 1 - Claude Integration)**:
- [ ] Korean Text-to-SQL query generates valid SQL (≥85% accuracy)
- [ ] Knowledge Discovery returns relevant results (≥90% Top-5)
- [ ] Response times within SLA (Text-to-SQL ≤60s, Knowledge ≤3s)
- [ ] Intent classification accuracy ≥95%
- [ ] No OpenAI references remaining in code
- [ ] Test coverage ≥80%

**After Phase 5 (User Story 3 - Budget Monitoring)**:
- [ ] Daily query count increments correctly
- [ ] Budget alerts trigger at thresholds (80%, 90%, 100%)
- [ ] Cost dashboard displays accurate data

### Rollback Criteria

Trigger rollback if ANY of the following occur in first 100 production queries:

- [ ] Error rate >5%
- [ ] SQL accuracy <85%
- [ ] Document search Top-5 accuracy <90%
- [ ] Response time p95 >60s for Text-to-SQL
- [ ] Critical API failures or authentication errors

**Rollback Time**: <30 minutes (git revert + redeploy)

---

## Success Metrics

### User Story 1 (Claude Integration)
- ✅ All LLM operations use Claude Code API (no OpenAI references)
- ✅ Korean language queries processed correctly
- ✅ SQL accuracy ≥85% (SC-003)
- ✅ Response times meet SLA (SC-004)
- ✅ Test coverage ≥80%

### User Story 2 (Embedding Quality)
- ✅ Vector search accuracy ≥90% Top-5 (SC-006)
- ✅ Search latency ≤0.5 seconds
- ✅ Korean + English multilingual support verified

### User Story 3 (Budget Monitoring)
- ✅ Monthly cost stays within $100 budget (SC-002)
- ✅ Daily cost tracking operational
- ✅ Alert system functional
- ✅ Constitution updated to version 2.0.0 (SC-007)

### Overall Migration
- ✅ Zero data loss (SC-005)
- ✅ Service interruption <4 hours (SC-005)
- ✅ User satisfaction ≥80% (SC-001)

---

## Notes

- **[P] tasks**: Different files, no dependencies, can run in parallel
- **[Story] label**: Maps task to specific user story (US1, US2, US3) for traceability
- **Each user story**: Independently completable and testable
- **Commit strategy**: Commit after each task or logical group of [P] tasks
- **Checkpoint validation**: Stop at each checkpoint to validate story independently
- **Avoid**: Vague tasks, same file conflicts, cross-story dependencies that break independence
- **Korean language**: All test fixtures and validation use Korean queries to verify language support

---

## Task Count Summary

- **Phase 1 (Setup)**: 7 tasks
- **Phase 2 (Foundational)**: 10 tasks (BLOCKING - must complete before user stories)
- **Phase 3 (User Story 1 - Claude Integration)**: 23 tasks (P1 - MVP)
- **Phase 4 (User Story 2 - Embedding Quality)**: 9 tasks (P2)
- **Phase 5 (User Story 3 - Budget Monitoring)**: 13 tasks (P3)
- **Phase 6 (Polish)**: 11 tasks

**Total**: 73 tasks

### Suggested MVP Scope
**Minimum Viable Product**: Phases 1-3 only (40 tasks)
- Delivers core Claude Code integration
- Maintains existing embedding model
- Enables basic system operation
- Can deploy and validate before adding P2/P3 features

**Recommended Initial Release**: Phases 1-4 (49 tasks)
- Adds embedding quality validation
- Ensures search accuracy meets targets
- More robust for production use

**Full Feature Release**: All phases (73 tasks)
- Complete migration with budget monitoring
- Constitutional compliance
- Production-ready with all success criteria met

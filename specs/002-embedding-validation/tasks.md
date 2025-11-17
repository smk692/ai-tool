# Tasks: Hugging Face 다국어 임베딩 통합 및 검증

**Feature ID**: 002-embedding-validation
**Input**: Design documents from `/specs/002-embedding-validation/`
**Prerequisites**: plan.md, spec.md, data-model.md

**Organization**: Tasks are grouped by phase for systematic implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[US2]**: User Story 2 - Hugging Face Embedding Integration
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root
- All paths assume single Python project structure

---

## Phase 1: Setup ✅ COMPLETE (T001-T017)

**Status**: ✅ Completed in previous phases

**Summary**:
- sentence-transformers>=2.2.0 installed
- paraphrase-multilingual-MiniLM-L12-v2 model selected
- ChromaDB initialized with cosine similarity
- EmbeddingConfiguration entity defined in src/models/embedding.py
- Model download script created (scripts/download_embedding_model.py)
- Vector store initialization script created (scripts/init_vector_store.py)

---

## Phase 2: Foundational ✅ COMPLETE

**Status**: ✅ Completed in Phase 3 (Claude Code migration)

**Summary**:
- Claude Code API integration complete
- All chains migrated to Claude 3.5 Sonnet
- Test coverage target: ≥80%
- OpenAI GPT-4o completely removed

**Checkpoint**: Foundation ready - User Story 2 implementation can now proceed

---

## Phase 3: User Story 2 - Hugging Face 임베딩 통합 (Priority: P1) 🎯

**Goal**: Hugging Face 다국어 임베딩 모델을 통합하고 검색 품질 및 성능 검증

**Independent Test**: 1000개 문서 인덱싱 후 한국어/영어 쿼리로 Top-5 정확도 ≥90%, P95 지연시간 ≤0.5초 달성

### Implementation for User Story 2

- [ ] T041 [US2] 모델 설정 검증 - scripts/download_embedding_model.py 실행 및 384차원 확인
- [ ] T042 [P] [US2] HuggingFaceEmbedding 서비스 구현 in src/services/embeddings.py
- [ ] T043 [US2] ChromaDB 통합 업데이트 - VectorStore에 embedding_service 의존성 주입 in src/services/vector_store.py
- [ ] T044 [P] [US2] 문서 인덱싱 파이프라인 구현 - 배치 처리 및 진행 상황 표시 in scripts/index_documents.py
- [ ] T045 [P] [US2] 한국어/영어 단위 테스트 작성 - embed_text, embed_texts, 에러 케이스 in tests/unit/test_embeddings.py
- [ ] T046 [P] [US2] 벡터 검색 지연시간 테스트 - P95 ≤0.5초 검증 in tests/performance/test_search_latency.py
- [ ] T047 [US2] Top-5 정확도 벤치마크 구현 - 100 쿼리 테스트셋 in tests/benchmarks/test_embedding_accuracy.py
- [ ] T048 [US2] 다국어 지원 테스트 - 한국어/영어 혼합 쿼리, 유니코드 처리 (일본어/중국어 제외) in tests/unit/test_embeddings.py
- [ ] T049 [P] [US2] 문서화 작성 - 모델 사양, API 가이드 in docs/embedding-model.md

**Checkpoint**: User Story 2 완료 - 1000개 문서 인덱싱 성공, Top-5 정확도 ≥90%, P95 ≤0.5초 달성

---

## Phase 4: 한국어/영어 중심 고도화 (T050-T054)

**Goal**: 한국어/영어 검색 품질 개선 및 성능 최적화

**Focus**: 한국어와 영어만 지원 (일본어, 중국어 등 제외)

### Enhancement Tasks

- [ ] T050 [P] [US2] Top-5 정확도 실측 벤치마크 - 실제 한국어/영어 쿼리 100개로 정확도 측정 in tests/benchmarks/test_real_world_accuracy.py
- [ ] T051 [US2] 하이브리드 검색 구현 - BM25 + Vector 결합, 한국어 KoNLPy 통합 in src/services/hybrid_search.py
- [ ] T052 [P] [US2] 한국어/영어 Reranker 모델 통합 - ms-marco-MiniLM 또는 유사 모델 in src/services/reranker.py
- [ ] T053 [P] [US2] 자주 사용하는 쿼리 캐싱 전략 - Redis 또는 LRU 캐시, TTL 1시간 in src/services/query_cache.py
- [ ] T054 [P] [US2] 동적 배치 크기 조정 구현 - 문서 수에 따른 최적 배치 크기 자동 설정 in src/services/embeddings.py

**Checkpoint**: 한국어/영어 검색 품질 및 성능 개선 완료

---

## Phase 5: 버그 수정, 경고 제거 및 레거시 정리 (T055-T062)

**Goal**: 현재 이슈 해결, Pydantic V2 마이그레이션 완료, 레거시 코드 제거

**Current Issues**:
- 3개 단위 테스트 실패 (identity check 오류)
- 14개 Pydantic V1 deprecation 경고
- 커버리지 76.92% (목표: 80%+)

### Bug Fixes and Pydantic V2 Migration

- [ ] T055 [P] 단위 테스트 assertion 수정 - `assert result is True` → `assert result` in tests/unit/test_embeddings.py:271,277,286
- [ ] T056 Pydantic V1→V2 마이그레이션 - @validator → @field_validator, model_config 설정 in src/models/embedding.py
- [ ] T057 [P] Pydantic V1→V2 마이그레이션 - @validator → @field_validator in src/models/llm_config.py
- [ ] T058 [P] Pydantic V1→V2 마이그레이션 - @validator → @field_validator, json_encoders 제거 in src/models/query_request.py
- [ ] T059 [P] pytest 구성 업데이트 - asyncio_default_fixture_loop_scope 추가 in pyproject.toml
- [ ] T060 커버리지 80% 달성 - 추가 테스트 작성으로 embeddings.py 커버리지 향상 in tests/unit/test_embeddings.py

### Legacy Cleanup

- [ ] T061 레거시 코드 제거 작업:
  - 예산 모니터링 관련 코드/파일 삭제 (사용자 요청: "예산 모니터링 필요없어")
  - 일본어/중국어 지원 코드 제거 (사용자 요청: "한국어, 영어 빼고는 다른 언어는 필요없어")
  - Pydantic V1 deprecated 패턴 완전 제거 (V2 마이그레이션 후)
  - 주석 처리된 코드, 디버깅 코드 제거
  - 사용하지 않는 imports 정리
  - 임시 테스트 파일 제거
  - OpenAI 관련 모든 참조 최종 확인 및 제거

### Final Validation

- [ ] T062 최종 검증 및 문서화:
  - 전체 테스트 실행 (46개 테스트 모두 통과 확인)
  - Pydantic 경고 0건 확인 (pytest 실행 시 deprecation 경고 없음)
  - 커버리지 ≥80% 확인
  - 레거시 코드 제거 완료 확인
  - CHANGELOG.md 업데이트 (if exists)
  - tasks.md 상태 업데이트

**Checkpoint**: 모든 버그 수정 완료, 경고 0건, 레거시 코드 완전 제거, 프로덕션 준비 완료

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: ✅ COMPLETE - No dependencies
- **Foundational (Phase 2)**: ✅ COMPLETE - Depends on Setup
- **User Story 2 (Phase 3)**: Depends on Foundational completion
- **Enhancements (Phase 4)**: Depends on Phase 3 completion
- **Bug Fixes & Cleanup (Phase 5)**: Can start after Phase 3, parallel with Phase 4

### Within Each Phase

**Phase 3 (User Story 2)**:
- T041 (모델 검증) → MUST complete first
- T042 (임베딩 서비스) || T045 (단위 테스트) - can run in parallel
- T043 (ChromaDB 통합) → depends on T042
- T044 (인덱싱 파이프라인) || T049 (문서화) - can run in parallel
- T046 (지연시간 테스트) → depends on T043, T044
- T047 (정확도 벤치마크) → depends on T043, T044
- T048 (다국어 테스트) → depends on T042

**Phase 4 (Enhancements)**:
- All tasks can run in parallel after Phase 3 completes
- T050 || T052 || T053 || T054 (모두 독립적)
- T051 (하이브리드 검색) → may benefit from T050 results

**Phase 5 (Bug Fixes & Cleanup)**:
- T055 || T057 || T058 || T059 || T060 (모두 병렬 가능)
- T056 (embedding.py) → 단독 실행 권장 (핵심 파일)
- T061 (레거시 정리) → T056-T060 완료 후 실행 권장
- T062 (최종 검증) → T055-T061 모두 완료 후 실행 필수

### Parallel Opportunities

**Phase 3 (총 4-5개 병렬 작업 가능)**:
```bash
# Round 1
Task T042: HuggingFaceEmbedding 서비스 구현
Task T045: 단위 테스트 작성

# Round 2 (after T042 complete)
Task T044: 문서 인덱싱 파이프라인
Task T049: 문서화 작성

# Round 3 (after T043, T044 complete)
Task T046: 벡터 검색 지연시간 테스트
Task T047: Top-5 정확도 벤치마크
```

**Phase 4 (총 4개 병렬 작업 가능)**:
```bash
Task T050: Top-5 정확도 실측 벤치마크
Task T052: Reranker 모델 통합
Task T053: 쿼리 캐싱 전략
Task T054: 동적 배치 크기 조정
```

**Phase 5 (총 7개 병렬 작업 가능)**:
```bash
# Pydantic V2 마이그레이션 (병렬)
Task T055: 테스트 assertion 수정
Task T057: llm_config.py 마이그레이션
Task T058: query_request.py 마이그레이션
Task T059: pytest 구성 업데이트
Task T060: 커버리지 향상

# 핵심 파일 (단독 또는 순차)
Task T056: embedding.py 마이그레이션

# 최종 작업 (순차)
Task T061: 레거시 코드 제거
Task T062: 최종 검증
```

---

## Implementation Strategy

### MVP First (Phase 3 Only)

1. Complete Phase 3: User Story 2 (T041-T049)
2. **STOP and VALIDATE**:
   - 1000개 문서 인덱싱 성공
   - Top-5 정확도 ≥90%
   - P95 지연시간 ≤0.5초
3. Deploy/demo if ready (기본 임베딩 기능 완료)

### Incremental Delivery

1. Phase 3 완료 → 기본 임베딩 검색 기능 제공 (MVP!)
2. Phase 4 완료 → 한국어/영어 검색 품질 개선 (Enhanced)
3. Phase 5 완료 → 프로덕션 준비 완료 (Production-Ready)

### Recommended Execution Order

**Week 1: Phase 3 - User Story 2**
- Day 1: T041, T042 (모델 검증, 임베딩 서비스)
- Day 2-3: T043, T044 (ChromaDB 통합, 인덱싱 파이프라인)
- Day 4: T045, T046 (단위 테스트, 지연시간 테스트)
- Day 5: T047, T048, T049 (정확도 벤치마크, 다국어 테스트, 문서화)

**Week 2: Phase 4 - Enhancements**
- Day 1: T050, T052 (실측 벤치마크, Reranker)
- Day 2: T051 (하이브리드 검색)
- Day 3: T053, T054 (캐싱, 동적 배치)

**Week 3: Phase 5 - Bug Fixes & Cleanup**
- Day 1: T055, T057, T058, T059 (병렬 작업)
- Day 2: T056, T060 (핵심 파일 마이그레이션, 커버리지)
- Day 3: T061 (레거시 정리)
- Day 4-5: T062 (최종 검증), 문서 업데이트, 배포 준비

---

## Estimated Timeline

| Phase | Task Count | Parallel Capacity | Estimated Time |
|-------|-----------|------------------|----------------|
| Phase 3 (US2) | 9 tasks | 4 parallel | 5-7 days |
| Phase 4 (Enhancement) | 5 tasks | 4 parallel | 3-4 days |
| Phase 5 (Cleanup) | 8 tasks | 7 parallel | 3-4 days |
| **Total** | **22 tasks** | **15 parallel** | **2-3 weeks** |

**Critical Path**: T041 → T042 → T043 → T046/T047 (Phase 3) → T050 (Phase 4) → T061 → T062 (Phase 5)

---

## Success Metrics

### Functional Metrics (Phase 3)

- ✅ Top-5 검색 정확도 ≥90% (한국어/영어)
- ✅ 검색 지연시간 P95 ≤0.5초
- ✅ 1000개 문서 인덱싱 성공률 ≥99%
- ✅ OpenAI 임베딩 참조 0건

### Enhancement Metrics (Phase 4)

- ✅ 하이브리드 검색 정확도 개선 (BM25 + Vector)
- ✅ Reranker 적용 시 Top-5 정확도 ≥95%
- ✅ 캐싱 적용 시 반복 쿼리 지연시간 50%+ 감소
- ✅ 동적 배치 크기 최적화로 처리량 향상

### Quality Metrics (Phase 5)

- ✅ 전체 테스트 통과율 100% (46개 테스트)
- ✅ Pydantic 경고 0건
- ✅ 테스트 커버리지 ≥80%
- ✅ 레거시 코드 완전 제거 확인

### Cost Metrics

- ✅ 임베딩 비용 $0 (완전 무료 오픈소스)
- ✅ 월 예산 절감 (OpenAI API 비용 제거)

---

## Notes

- [P] tasks = 다른 파일, 의존성 없음, 병렬 실행 가능
- [US2] label = User Story 2 관련 작업
- Phase 3 완료 후 Phase 4, 5 동시 진행 가능 (팀 리소스에 따라)
- 각 Phase 완료 시 Checkpoint에서 독립적 검증 필수
- **레거시 정리 (T061)**: 예산 모니터링, 일본어/중국어 지원 등 불필요한 코드 완전 제거
- **언어 지원**: 한국어/영어만 지원하도록 모든 다국어 관련 코드 정리

---

**Document Version**: 2.0.0
**Last Updated**: 2025-11-17
**Status**: Ready for Implementation
**Changes**: Added Phase 4 (Korean/English enhancements), Phase 5 (Bug fixes, Pydantic V2 migration, Legacy cleanup)

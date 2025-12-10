# Checklist: 데이터 흐름 및 통합 요구사항 품질

**Purpose**: 데이터 파이프라인, Slack Bot SDK 소켓모드, Claude Code SDK 통합 요구사항의 완전성/명확성 검증
**Created**: 2025-12-10
**Type**: Pre-commit (경량)
**Focus**: 데이터 흐름, 가드레일, 응답 품질, 출처 링크

---

## 요구사항 완전성 (Completeness)

- [ ] CHK001 - Slack Bot SDK 소켓모드 연결 요구사항이 명시되어 있는가? [Gap, FR-001 관련]
- [ ] CHK002 - Claude Code SDK 사용에 대한 구체적 요구사항이 정의되어 있는가? [Gap, FR-004 관련]
- [ ] CHK003 - 쿼리→임베딩→검색→LLM→응답 전체 파이프라인의 각 단계 요구사항이 정의되어 있는가? [Completeness, FR-002~FR-005]
- [ ] CHK004 - 출처 링크 형식(Notion URL, Swagger 참조 등)이 구체적으로 명시되어 있는가? [Gap, FR-007]

---

## 요구사항 명확성 (Clarity)

- [ ] CHK005 - "정확한 답변"의 기준이 측정 가능한 형태로 정의되어 있는가? [Ambiguity, SC-002]
- [ ] CHK006 - 유사도 임계값 0.7의 선정 근거가 문서화되어 있는가? [Clarity, FR-003]
- [ ] CHK007 - "최근 5개 메시지" 컨텍스트의 토큰 제한이 명시되어 있는가? [Clarity, Clarifications §Session]
- [ ] CHK008 - 출처 표시 시 "제목과 링크"의 정확한 포맷이 정의되어 있는가? [Clarity, Spec §User Story 3]

---

## 가드레일 요구사항 (Security/Guardrails)

- [ ] CHK009 - 민감 정보 패턴(이메일, 전화번호, 주민번호)의 정규식/탐지 기준이 명시되어 있는가? [Gap, Edge Cases]
- [ ] CHK010 - 가드레일 발동 시 "경고" vs "차단" 조건이 구분되어 정의되어 있는가? [Clarity, Clarifications §Session]
- [ ] CHK011 - 가드레일 우회 시나리오(false positive/negative)에 대한 처리 요구사항이 있는가? [Coverage, Gap]

---

## 응답 품질 요구사항 (Quality)

- [ ] CHK012 - LLM 프롬프트 템플릿 또는 시스템 지시문에 대한 요구사항이 정의되어 있는가? [Gap]
- [ ] CHK013 - 응답이 "환각(hallucination)"을 포함할 경우의 처리 방안이 명시되어 있는가? [Gap, Edge Cases]
- [ ] CHK014 - 검색 결과가 없을 때 vs 관련성 낮을 때의 응답 차이가 구분되어 있는가? [Coverage, Edge Cases]

---

## 통합 의존성 (Integration)

- [ ] CHK015 - Slack API 소켓모드 연결 실패 시 재연결 정책이 명시되어 있는가? [Gap, FR-001]
- [ ] CHK016 - Claude API 타임아웃 또는 rate limit 시 재시도 전략이 정의되어 있는가? [Gap, Edge Cases]
- [ ] CHK017 - Qdrant 벡터DB 연결 실패 시 fallback 처리가 정의되어 있는가? [Gap, FR-003]

---

## 측정 가능성 (Measurability)

- [ ] CHK018 - "10초 이내 응답 시작"의 측정 시점(사용자 전송 vs 서버 수신)이 명확한가? [Measurability, SC-001]
- [ ] CHK019 - "90% 정확한 답변" 달성 여부를 어떻게 측정할지 정의되어 있는가? [Measurability, SC-002]
- [ ] CHK020 - "동시 요청 10개 처리"의 부하 테스트 조건이 명시되어 있는가? [Measurability, SC-005]

---

## 요약

| 카테고리 | 항목 수 | 주요 포커스 |
|---------|--------|------------|
| 완전성 | 4 | SDK 통합, 파이프라인, 출처 |
| 명확성 | 4 | 측정 기준, 포맷 정의 |
| 가드레일 | 3 | 민감정보, 차단 조건 |
| 품질 | 3 | LLM 프롬프트, 환각 처리 |
| 통합 | 3 | 재연결, 재시도, fallback |
| 측정 | 3 | 성능, 정확도, 부하 |
| **총계** | **20** | |

<!--
Sync Impact Report
==================
Version Change: INITIAL → 1.0.0
Principles Defined:
  1. 비용 효율성 우선 (Cost-First)
  2. 응답 시간 보장 (Response Time SLA)
  3. 정확도 우선 (Accuracy-First)
  4. 오픈소스 기반 (Open-Source First)
  5. 단순성 유지 (Simplicity)
  6. 보안 우선 (Security-First)

Added Sections:
  - 기술 제약사항 (Technical Constraints)
  - 품질 관리 (Quality Management)
  - 거버넌스 (Governance)

Templates Status:
  ✅ .specify/templates/plan-template.md - Reviewed, no updates needed
  ✅ .specify/templates/spec-template.md - Reviewed, no updates needed
  ✅ .specify/templates/tasks-template.md - Reviewed, no updates needed

Follow-up TODOs:
  - None - All placeholders filled

Last Generated: 2025-01-13
-->

# 물어보새 (Ask-Bird) 프로젝트 헌법

AI 기반 회사 문서 통합 검색 Slack 봇 프로젝트의 핵심 원칙과 거버넌스 규칙을 정의합니다.

---

## 핵심 원칙 (Core Principles)

### I. 비용 효율성 우선 (Cost-First)

**원칙**:
- **MUST**: 월 운영 비용 $50 이하 유지 (LLM API 비용 포함)
- **MUST**: 오픈소스 솔루션 우선 검토, 상용 도구는 명확한 ROI 정당화 필요
- **MUST**: LLM API 호출 최소화 - 캐싱, 프롬프트 압축, 배치 처리 적용
- **MUST**: 비용 추적 및 모니터링 시스템 운영 (일일 비용 알림)

**근거**:
소규모 회사(100명)의 제한된 예산($100/월 Claude Code 구독)으로 지속 가능한 운영을 보장하기 위함. 비용 초과 시 프로젝트 중단 리스크가 높으므로 비용 효율성이 최우선 원칙.

**검증 방법**:
- 월간 LLM API 비용 리포트 생성
- 응답당 평균 비용 계산 (목표: $0.05 이하)
- 캐싱 히트율 80% 이상 유지

---

### II. 응답 시간 보장 (Response Time SLA)

**원칙**:
- **MUST**: Text-to-SQL 응답 시간 30초~1분 (목표), 최대 허용 2분
- **MUST**: 문서 검색(Knowledge Discovery) 응답 시간 3초 이내, 최대 10초
- **MUST**: Vector 검색 0.5초 이내, BM25 검색 0.01초 이내
- **MUST**: 시스템 가용성 99% 이상 (월간 downtime < 7.2시간)
- **SHOULD**: 응답 시간 초과 시 사용자에게 진행 상황 표시

**근거**:
사용자 경험 최우선. 30초~1분 내 답변 제공이 핵심 가치 제안. 응답 지연 시 기존 문서 검색 방식 대비 효율성 저하.

**검증 방법**:
- 모든 응답에 타임스탬프 로깅
- p95 응답 시간 모니터링 (95%의 요청이 목표 시간 내 완료)
- 주간 SLA 리포트 생성

---

### III. 정확도 우선 (Accuracy-First)

**원칙**:
- **MUST**: SQL 쿼리 생성 정확도 85% 이상 (사용자 수정 없이 실행 가능)
- **MUST**: 문서 검색 정확도(Top-5) 90% 이상
- **MUST**: 의도 분류(Router) 정확도 95% 이상
- **MUST**: 불확실한 경우 "모르겠습니다" 솔직한 응답 (환각 방지)
- **MUST**: 모든 답변에 출처 명시 (테이블명, 문서명, 신뢰도 점수)

**근거**:
잘못된 SQL 쿼리나 문서 답변은 업무 혼란 초래. 정확도가 낮으면 사용자 신뢰 상실하여 도구 사용 중단. "모르겠습니다" 응답이 틀린 답변보다 낫다.

**검증 방법**:
- Few-shot 예제 기반 테스트 세트 구축 (최소 100개)
- 사용자 피드백(👍👎) 수집 및 정확도 계산
- 월간 정확도 리포트 생성
- A/B 테스트로 프롬프트 개선

---

### IV. 오픈소스 기반 (Open-Source First)

**원칙**:
- **MUST**: 모든 컴포넌트는 오픈소스 우선 검토 (Apache 2.0, MIT 라이선스 선호)
- **MUST**: 상용 도구 도입 시 다음 정당화 필수:
  - 오픈소스 대안 부재 증명
  - 명확한 비용 대비 효과(ROI) 계산
  - 벤더 락인(vendor lock-in) 리스크 평가
- **MUST**: LLM API는 Anthropic Claude (메인) 또는 Google Gemini (백업) 사용
- **SHOULD**: 커뮤니티 기여 고려 (내부 개선사항 오픈소스화 검토)

**근거**:
제한된 예산으로 지속 가능한 운영 보장. 오픈소스는 라이선스 비용 없음, 커뮤니티 지원, 커스터마이징 용이.

**현재 스택**:
- LangChain (MIT), LangGraph (MIT)
- ChromaDB (Apache 2.0), sentence-transformers (Apache 2.0)
- PostgreSQL (PostgreSQL License), SQLite (Public Domain)
- Streamlit (Apache 2.0), rank-bm25 (Apache 2.0)

**검증 방법**:
- 신규 라이브러리 도입 시 라이선스 검증 체크리스트
- 상용 도구 제안서 작성 (ROI, 대안 분석 포함)

---

### V. 단순성 유지 (Simplicity)

**원칙**:
- **MUST**: KISS(Keep It Simple, Stupid) 원칙 준수
- **MUST**: YAGNI(You Aren't Gonna Need It) - 현재 필요한 기능만 구현, 추측성 개발 금지
- **MUST**: 복잡도 증가는 명확한 비즈니스 가치로 정당화
- **MUST**: 코드 리뷰 시 단순화 가능성 검토
- **SHOULD**: 3개월마다 불필요한 코드 제거(리팩토링)

**근거**:
소규모 팀(1-2명)의 유지보수 부담 최소화. 복잡한 시스템은 버그 증가, 수정 시간 증가, 신규 개발자 온보딩 어려움.

**예시**:
- ❌ 나쁜 예: 미래 확장을 위한 추상화 레이어 10개
- ✅ 좋은 예: 현재 2개 기능(Text-to-SQL, 문서 검색)만 구현, 필요 시 확장

**검증 방법**:
- Cyclomatic Complexity < 10 (코드 복잡도 측정)
- 함수 길이 < 50줄
- 클래스 메서드 수 < 10개
- 코드 리뷰 체크리스트에 "단순화 가능?" 항목 포함

---

### VI. 보안 우선 (Security-First)

**원칙**:
- **MUST**: 읽기 전용 DB 계정 사용 (SELECT만 허용)
- **MUST**: DELETE, DROP, TRUNCATE, ALTER 등 쿼리 실행 금지
- **MUST**: SQL Injection 방지 - 파라미터 바인딩, 입력 검증, 화이트리스트
- **MUST**: API 키 및 DB 비밀번호 환경 변수 관리 (.env, 절대 Git 커밋 금지)
- **MUST**: 로그에 개인정보(이메일, 주민번호 등) 미포함
- **SHOULD**: 쿼리 실행 전 사용자 확인(Confirm) 단계 추가

**근거**:
데이터베이스 손상 방지. SQL Injection 공격으로 인한 데이터 유출/삭제 방지. 회사 민감 정보 보호.

**검증 방법**:
- DB 계정 권한 확인 (SELECT 외 권한 없음)
- SQL Injection 테스트 (OWASP Top 10)
- 환경 변수 검증 (.env 파일 gitignore 확인)
- 로그 스캔 (정규표현식으로 이메일/전화번호 탐지)

---

## 기술 제약사항 (Technical Constraints)

### 필수 기술 스택

**프로그래밍 언어**:
- Python 3.10 이상 (타입 힌팅 필수)

**LLM**:
- OpenAI GPT-4o (유일한 상용 컴포넌트, 월 $50 예산 내)

**프레임워크**:
- LangChain 0.1.0+ (MIT)
- LangGraph 0.0.20+ (MIT)

**데이터베이스**:
- PostgreSQL 13+ (운영 DB)
- SQLite 3.35+ (대화 메모리)
- ChromaDB 0.4.0+ (Vector Store)

**검색**:
- sentence-transformers 2.2.0+ (임베딩)
- rank-bm25 0.2.2+ (키워드 검색)

**UI**:
- Streamlit 1.28+ (초기 프로토타입)
- Slack Bot API (Phase 2 확장)

### 인프라 요구사항

**최소 사양**:
- CPU: 4 core
- RAM: 8GB
- Disk: 50GB SSD
- Network: 100Mbps

**권장 사양** (운영 환경):
- CPU: 8 core
- RAM: 16GB
- Disk: 100GB SSD
- GPU: NVIDIA (임베딩 가속, 선택 사항)

### 금지 사항

**절대 사용 금지**:
- 상용 Vector DB (Pinecone, Weaviate 등) - ChromaDB 또는 Qdrant 사용
- 비오픈소스 라이브러리 (명확한 정당화 없이)

**허용된 상용 LLM**:
- Anthropic Claude (메인)
- Google Gemini (백업)

---

## 품질 관리 (Quality Management)

### 코드 품질 기준

**테스팅**:
- **MUST**: 단위 테스트 커버리지 80% 이상
- **MUST**: 통합 테스트 (E2E) 주요 기능별 최소 1개
- **MUST**: 모든 PR은 테스트 통과 필수
- **SHOULD**: pytest-benchmark로 성능 회귀 방지

**코드 스타일**:
- **MUST**: flake8 린팅 통과
- **MUST**: black 포매터 적용
- **SHOULD**: mypy 타입 체킹 (점진적 적용)

**문서화**:
- **MUST**: 모든 public 함수/클래스에 docstring (Google 스타일)
- **MUST**: README.md 최신 상태 유지 (설치, 사용법, 트러블슈팅)
- **SHOULD**: API 문서 자동 생성 (Sphinx)

### 성능 기준

| 메트릭 | 목표 | 측정 방법 |
|--------|------|----------|
| Text-to-SQL 응답 시간 | 30~60초 | 평균, p95 |
| 문서 검색 응답 시간 | <3초 | 평균, p95 |
| SQL 쿼리 정확도 | ≥85% | 사용자 피드백 |
| 검색 정확도 (Top-5) | ≥90% | 테스트 세트 |
| 시스템 가용성 | ≥99% | Uptime 모니터링 |
| LLM API 비용 | <$50/월 | 일일 비용 추적 |

### 코드 리뷰 체크리스트

모든 PR은 다음 검증 필수:

- [ ] 헌법 원칙 준수 확인
- [ ] 비용 영향 평가 (LLM API 호출 증가 여부)
- [ ] 응답 시간 영향 평가 (성능 저하 여부)
- [ ] 보안 검증 (SQL Injection, API 키 노출 등)
- [ ] 테스트 커버리지 유지/증가
- [ ] 단순성 검토 (불필요한 복잡도 제거 가능?)
- [ ] 문서화 업데이트 (README, docstring)

---

## 거버넌스 (Governance)

### 헌법 수정 절차

**MAJOR 버전 (X.0.0)** - 역호환 불가능한 원칙 변경:
- 기존 원칙 제거 또는 근본적 재정의
- 기술 스택 전면 교체
- 승인 필요: 프로젝트 리더 + 기술 검토

**MINOR 버전 (x.Y.0)** - 새로운 원칙 추가:
- 신규 원칙 추가
- 기존 원칙 확장 (비호환 없음)
- 승인 필요: 프로젝트 리더

**PATCH 버전 (x.y.Z)** - 명확화 및 개선:
- 오타 수정
- 문구 명확화
- 예시 추가
- 승인 필요: 코드 리뷰어

### 수정 프로세스

1. **제안서 작성**:
   - 변경 사유
   - 영향 분석 (기존 코드, 비용, 성능)
   - 마이그레이션 계획

2. **검토 및 승인**:
   - 기술 검토 (아키텍트)
   - 비용 검토 (프로젝트 관리자)
   - 최종 승인

3. **문서 업데이트**:
   - constitution.md 수정
   - 버전 업데이트
   - Sync Impact Report 생성
   - 관련 템플릿 업데이트 (plan, spec, tasks)

4. **전파**:
   - 팀 공지
   - 기존 코드 마이그레이션 (필요 시)

### 준수 검증

**자동화**:
- CI/CD 파이프라인에서 체크리스트 자동 검증
- 비용 추적 스크립트 (일일 실행)
- 성능 모니터링 대시보드

**수동 검증**:
- 주간 헌법 준수 리뷰 (20분)
- 월간 메트릭 리포트 (정확도, 비용, 성능)
- 분기별 회고 (헌법 개선 제안)

### 예외 처리

**긴급 상황**:
- 보안 취약점 발견 시 즉시 수정 (헌법 위반 허용)
- 사후 헌법 업데이트 및 승인

**실험적 기능**:
- Feature Flag로 격리
- 헌법 위반 기능은 기본 비활성화
- 검증 후 헌법 업데이트 또는 기능 제거

---

## 참고 자료 (References)

**프로젝트 문서**:
- [기능 명세서](../../../spec.md)
- [구현 계획서](../../../plan.md)
- [작업 분해](../../../tasks.md)

**외부 참고**:
- [물어보새 블로그 1부](https://techblog.woowahan.com/18144/)
- [물어보새 블로그 2부](https://techblog.woowahan.com/18362/)
- [물어보새 블로그 3부](https://techblog.woowahan.com/23273/)
- [WOOWACON 2024 발표](https://youtu.be/_QPhoKItI2k?si=AJ5e0LTT8lzRUKUX&t=2165)

**개발 가이드**:
- 런타임 개발 가이드: 별도 작성 예정 (dev-guide.md)

---

**Version**: 1.1.0 | **Ratified**: 2025-01-13 | **Last Amended**: 2025-12-10

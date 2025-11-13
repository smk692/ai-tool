# AI 데이터 분석 어시스턴트 - 기능 명세서

## 문서 정보

- **프로젝트명**: AI 데이터 분석 어시스턴트 (물어보새 기반)
- **버전**: 1.0.0
- **작성일**: 2025년 1월
- **기술 스택**: 100% 오픈소스 (LLM API 제외)
- **참고**: 우아한형제들 '물어보새' 시스템 사례 연구

---

## 📋 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [핵심 기능 명세](#2-핵심-기능-명세)
3. [시스템 아키텍처](#3-시스템-아키텍처)
4. [기술 요구사항](#4-기술-요구사항)
5. [데이터 요구사항](#5-데이터-요구사항)
6. [성능 요구사항](#6-성능-요구사항)
7. [보안 요구사항](#7-보안-요구사항)
8. [사용자 인터페이스](#8-사용자-인터페이스)
9. [확장성 요구사항](#9-확장성-요구사항)

---

## 1. 프로젝트 개요

### 1.1 프로젝트 배경

**문제 인식**:
- SQL 학습 장벽으로 인한 데이터 활용 제약
- 복잡한 쿼리 이해 및 유지보수 어려움
- 분산된 지식 자원으로 인한 정보 접근 비효율
- 반복적인 업무 문의로 인한 담당자 부담

**해결 방안**:
자연어 인터페이스를 통한 데이터 분석 민주화 및 지식 통합 시스템 구축

### 1.2 프로젝트 목표

**핵심 목표**: 구성원의 데이터 리터러시 상향 평준화

**구체적 목표**:
1. SQL 비전문가도 자연어로 데이터 추출 가능
2. 30초~1분 내 업무 참조 가능 수준의 쿼리 생성
3. 사내 모든 지식 자원 통합 검색
4. 24시간 즉시 응대 체계 구축
5. 월 $50 이하 운영 비용 달성 (오픈소스 활용)

### 1.3 성공 지표

| 지표 | 목표 | 측정 방법 |
|------|------|----------|
| 쿼리 생성 정확도 | 85% 이상 | 사용자 수정 없이 실행 가능한 쿼리 비율 |
| 응답 시간 | Text-to-SQL 30초~1분 | 평균 응답 시간 측정 |
| 사용자 만족도 | 80% 이상 | 월간 사용자 설문 조사 |
| 시스템 가용성 | 99% 이상 | 월간 uptime 비율 |
| 운영 비용 | $50/월 이하 | LLM API 비용 추적 |

---

## 2. 핵심 기능 명세

### 2.1 Text-to-SQL (자연어 → SQL 변환)

#### 기능 설명
사용자의 자연어 질문을 실행 가능한 SQL 쿼리로 자동 변환

#### 기능 요구사항

**FR-SQL-001**: 자연어 질문 이해
- 한글 자연어 질문을 SQL로 변환
- 비즈니스 용어를 DB 스키마 용어로 자동 매핑
- 시제 및 날짜 표현 자동 해석 (예: "지난달", "이번 주")

**FR-SQL-002**: 스키마 자동 선별
- 질문과 관련된 테이블/칼럼 자동 식별
- 최대 3-5개 테이블로 프롬프트 크기 최적화
- 유사도 기반 관련성 점수 계산 (임베딩 + BM25)

**FR-SQL-003**: Few-shot 학습
- 유사 질문-쿼리 예제 자동 검색 (top-k=3)
- 예제 성공률 추적 및 가중치 반영
- 사용자 피드백 기반 예제 품질 개선

**FR-SQL-004**: 쿼리 생성
- PostgreSQL 문법 준수
- JOIN 조건 자동 추론
- GROUP BY, ORDER BY 자동 추가
- 날짜/시간 처리 (타임존 자동 변환)

**FR-SQL-005**: 쿼리 검증
- 문법 오류 사전 검증 (sqlparse 활용)
- 테이블/칼럼 존재 여부 확인
- 실행 계획 분석 (EXPLAIN)
- 자동 수정 제안

**FR-SQL-006**: 쿼리 실행 (선택 사항)
- 사용자 확인 후 쿼리 실행
- 결과 포맷팅 (표, CSV, JSON)
- 실행 시간 및 행 수 표시

#### 입출력 명세

**입력**:
```
질문: "지난달 배민 신규 가입자 수는?"
```

**출력**:
```sql
SELECT COUNT(*) as 신규_가입자_수
FROM users
WHERE service = 'baemin'
  AND created_at >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')
  AND created_at < DATE_TRUNC('month', CURRENT_DATE);
```

**메타데이터**:
```json
{
  "confidence": 0.92,
  "selected_tables": ["users"],
  "similar_examples": 2,
  "generation_time": 1.2,
  "validation_passed": true
}
```

#### 예외 처리

**EX-SQL-001**: 모호한 질문
- 명확화 질문 반환
- 예: "어느 기간의 데이터를 조회하시겠습니까?"

**EX-SQL-002**: 관련 테이블 없음
- 사용 가능한 테이블 목록 제시
- Data Discovery 기능 안내

**EX-SQL-003**: 복잡도 초과
- 질문 분해 제안
- 단계별 쿼리 생성 가이드

---

### 2.2 Data Discovery (데이터 탐색)

#### 기능 설명
데이터베이스 스키마, 메타데이터, 활용 예시를 자연어로 설명

#### 기능 요구사항

**FR-DD-001**: 테이블 목록 조회
- 전체 테이블 목록 표시
- 테이블별 설명 및 주요 용도
- 데이터 품질 정보 (행 수, 최신 업데이트 시각)

**FR-DD-002**: 테이블 상세 정보
- 칼럼 목록 및 데이터 타입
- 칼럼별 설명 및 샘플 값
- Primary Key, Foreign Key 관계
- 인덱스 정보

**FR-DD-003**: 칼럼 해설
- 칼럼 의미 및 비즈니스 정의
- 허용 값 범위 및 제약 조건
- NULL 허용 여부
- 샘플 데이터 (3-5개)

**FR-DD-004**: 관계 시각화
- ERD (Entity Relationship Diagram) 생성
- 테이블 간 JOIN 경로 표시
- 의존성 그래프

**FR-DD-005**: 활용 예시
- 테이블별 대표 쿼리 예제
- 일반적인 분석 시나리오
- 주의사항 및 베스트 프랙티스

#### 입출력 명세

**입력**:
```
질문: "users 테이블에 대해 알려줘"
```

**출력**:
```
📊 테이블: users

**목적**: 사용자 계정 정보 관리

**주요 칼럼**:
- id (BIGINT, PK): 사용자 고유 식별자
- email (VARCHAR, UNIQUE): 이메일 주소
- created_at (TIMESTAMP): 계정 생성 시각 (KST)
- status (VARCHAR): 계정 상태 (active/inactive/suspended)

**데이터 품질**:
- 총 행 수: 1,234,567
- 최신 업데이트: 2025-01-13 14:30:00 KST
- Completeness: 99.8% (created_at)

**활용 예시**:
1. 신규 가입자 추이 분석
2. 활성 사용자 수 집계
3. 이메일 도메인별 분포 확인

**관련 테이블**:
- orders (users.id = orders.user_id)
- user_profiles (users.id = user_profiles.user_id)
```

---

### 2.3 Knowledge Discovery (지식 검색)

#### 기능 설명
사내 문서, Wiki, Notion 등 통합 검색 및 답변 생성

#### 기능 요구사항

**FR-KD-001**: 문서 인덱싱
- Markdown, PDF, 이미지 등 다양한 포맷 지원
- ChromaDB Vector 저장소에 임베딩
- BM25 키워드 인덱스 병행 구축
- 증분 인덱싱 (신규/수정 문서만)

**FR-KD-002**: 하이브리드 검색
- Vector 유사도 검색 (0.5 가중치)
- BM25 키워드 검색 (0.5 가중치)
- Reciprocal Rank Fusion (RRF) 스코어 결합
- Top-k 문서 반환 (k=5)

**FR-KD-003**: 컨텍스트 기반 답변
- 검색된 문서 기반 답변 생성
- 출처 명시 (문서명, 페이지, URL)
- 신뢰도 점수 표시
- "모르겠습니다" 솔직한 응답

**FR-KD-004**: 이미지 검색 (선택 사항)
- 이미지 내 텍스트 OCR 추출
- 이미지 캡션 생성 (멀티모달 LLM)
- 이미지 임베딩 검색

**FR-KD-005**: 실시간 정보 접근 (확장)
- MCP 서버 연동 (Slack, Jira, Notion)
- API 기반 최신 정보 조회
- 캐싱으로 중복 요청 방지

#### 입출력 명세

**입력**:
```
질문: "배포 프로세스에 대해 알려줘"
```

**출력**:
```
📚 배포 프로세스 안내

**프로세스 개요**:
1. 개발 완료 후 PR 생성
2. 코드 리뷰 및 승인 (최소 2명)
3. CI/CD 파이프라인 자동 실행
4. Staging 환경 배포 및 테스트
5. Production 배포 (승인 필요)

**주의사항**:
- 배포 시간: 평일 14:00-16:00 권장
- 롤백 계획 필수 준비
- 데이터베이스 마이그레이션 사전 검토

**출처**:
- 📄 개발 가이드 > 배포 프로세스 (v2.3)
- 🔗 https://wiki.company.com/deploy-process
- 신뢰도: 95%
```

---

### 2.4 Multi-turn 대화 (대화 맥락 유지)

#### 기능 설명
이전 대화 맥락을 이해하고 연속적인 질문에 답변

#### 기능 요구사항

**FR-MT-001**: 대화 이력 저장
- 질문, 답변, 생성 쿼리 저장
- SQLite 기반 로컬 저장소
- 세션 ID 기반 그룹화

**FR-MT-002**: 맥락 재구성
- 최근 N개 (N=5) 대화 참조
- 대화 요약으로 토큰 절약
- 중요 정보 우선 보존

**FR-MT-003**: 대명사 해결
- "그 테이블", "이거", "그거" 등 해석
- 이전 언급 엔티티 추적
- 명확화 질문 생성

**FR-MT-004**: 연속 질문 이해
- "그럼 지난주는?", "비율로 보여줘" 등
- 이전 쿼리 기반 수정
- 점진적 쿼리 개선

**FR-MT-005**: 세션 관리
- 세션 시작/종료 감지
- 세션 요약 저장
- 장기 기억 (선택 사항)

#### 입출력 명세

**대화 예시**:

```
사용자: "지난달 신규 가입자 수는?"
AI: [쿼리 생성 및 결과: 1,234명]

사용자: "그럼 작년 같은 달은?"
AI: [맥락 이해 → "작년 같은 달" = 2024년 1월, 쿼리 생성]

사용자: "증감률은?"
AI: [이전 두 결과 활용 → 증감률 계산 쿼리 생성]
```

**저장 구조**:
```python
{
  "session_id": "uuid-1234",
  "conversations": [
    {
      "turn": 1,
      "user_message": "지난달 신규 가입자 수는?",
      "assistant_message": "...",
      "generated_query": "SELECT COUNT(*) FROM ...",
      "result_summary": "1,234명",
      "timestamp": "2025-01-13T10:00:00"
    },
    ...
  ]
}
```

---

### 2.5 Query Assistant (쿼리 해설 및 검증)

#### 기능 설명
기존 SQL 쿼리를 자연어로 설명하고 문법 오류 검증

#### 기능 요구사항

**FR-QA-001**: 쿼리 해설
- SQL 쿼리를 자연어로 번역
- 단계별 실행 과정 설명
- 비즈니스 의미 해석

**FR-QA-002**: 문법 검증
- sqlparse 기반 문법 체크
- 테이블/칼럼 존재 여부 확인
- 오타 자동 수정 제안

**FR-QA-003**: 성능 분석
- EXPLAIN ANALYZE 실행
- 인덱스 사용 여부 확인
- 최적화 제안

**FR-QA-004**: 보안 검증
- SQL Injection 패턴 감지
- 권한 검증 (읽기 전용)
- 위험 쿼리 경고 (DELETE, DROP 등)

#### 입출력 명세

**입력**:
```sql
SELECT u.email, COUNT(o.id)
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE o.created_at >= '2025-01-01'
GROUP BY u.email
HAVING COUNT(o.id) > 10
ORDER BY COUNT(o.id) DESC;
```

**출력**:
```
📖 쿼리 해설:

**목적**: 2025년 1월 1일 이후 주문을 10건 이상 한 사용자 조회

**단계별 설명**:
1. users 테이블과 orders 테이블을 LEFT JOIN
   - 조건: 사용자 ID가 일치
2. 2025년 1월 1일 이후 주문만 필터링
3. 이메일별로 그룹화하여 주문 건수 집계
4. 주문 10건 초과 사용자만 선택
5. 주문 건수 많은 순으로 정렬

**결과**: 상위 VIP 고객 목록

⚠️ 개선 제안:
- orders.created_at에 인덱스 추가 권장
- u.email 대신 u.id로 GROUP BY 하면 더 빠름
```

---

## 3. 시스템 아키텍처

### 3.1 고수준 아키텍처

```
┌─────────────────────────────────────────────────────┐
│                사용자 인터페이스                      │
│              (Streamlit / Gradio)                   │
└───────────────────┬─────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────┐
│            Router Supervisor Chain                  │
│         (의도 분석 및 체인 라우팅)                   │
└───┬────────┬────────┬────────┬────────┬─────────────┘
    │        │        │        │        │
    ▼        ▼        ▼        ▼        ▼
┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐
│Text │  │Data │  │Know │  │Query│  │Memo │
│-to- │  │Disc │  │ledge│  │Asst │  │ry  │
│SQL  │  │overy│  │Disc │  │     │  │Mgr  │
└──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘
   │        │        │        │        │
   └────────┴────────┴────────┴────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
    ┌───▼────┐            ┌─────▼────┐
    │  RAG   │            │   LLM    │
    │ System │            │   API    │
    │        │            │ (GPT-4o) │
    └───┬────┘            └──────────┘
        │
  ┌─────┴─────┐
  │           │
┌─▼──┐    ┌──▼──┐
│Vec │    │BM25 │
│tor │    │Search│
│DB  │    │      │
└────┘    └─────┘
```

### 3.2 컴포넌트 명세

#### 3.2.1 Router Supervisor Chain

**역할**: 사용자 질문의 의도를 분석하고 적절한 체인으로 라우팅

**입력**: 사용자 질문 (자연어)

**출력**: 체인 선택 결과 + 신뢰도

**로직**:
```python
if "테이블" in question or "칼럼" in question:
    return "data_discovery"
elif "문서" in question or "Wiki" in question:
    return "knowledge_discovery"
elif "설명해줘" in question and contains_sql(question):
    return "query_assistant"
else:
    return "text_to_sql"  # default
```

#### 3.2.2 RAG System

**컴포넌트**:
- ChromaDB Vector Store
- BM25 Keyword Search
- Hybrid Search Combiner

**데이터 흐름**:
```
Query → Embedding → Vector Search (top-10)
     └→ Tokenize → BM25 Search (top-10)
              └→ RRF Fusion → Top-5 Results
```

#### 3.2.3 Memory Manager

**저장소**: SQLite

**스키마**:
```sql
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY,
    session_id TEXT,
    turn_number INTEGER,
    user_message TEXT,
    assistant_message TEXT,
    generated_query TEXT,
    result_summary TEXT,
    timestamp DATETIME
);
```

---

## 4. 기술 요구사항

### 4.1 필수 기술 스택

| 레이어 | 기술 | 버전 | 라이선스 |
|--------|------|------|----------|
| **LLM** | OpenAI GPT-4o | API | 상용 (필수 비용) |
| **프레임워크** | LangChain | 0.1.0+ | MIT |
| **프레임워크** | LangGraph | 0.0.20+ | MIT |
| **Vector DB** | ChromaDB | 0.4.0+ | Apache 2.0 |
| **검색** | rank-bm25 | 0.2.2+ | Apache 2.0 |
| **임베딩** | sentence-transformers | 2.2.0+ | Apache 2.0 |
| **RDBMS** | PostgreSQL | 13+ | PostgreSQL |
| **로컬 DB** | SQLite | 3.35+ | Public Domain |
| **UI** | Streamlit | 1.28+ | Apache 2.0 |

### 4.2 개발 환경

**Python**: 3.10+

**필수 패키지**:
```
langchain==0.1.0
langgraph==0.0.20
chromadb==0.4.22
sentence-transformers==2.2.2
rank-bm25==0.2.2
psycopg2-binary==2.9.9
sqlparse==0.4.4
streamlit==1.28.0
python-dotenv==1.0.0
tiktoken==0.5.2
```

### 4.3 인프라 요구사항

**최소 사양**:
- CPU: 4 core
- RAM: 8GB
- Disk: 50GB SSD
- Network: 100Mbps

**권장 사양**:
- CPU: 8 core
- RAM: 16GB
- Disk: 100GB SSD
- GPU: NVIDIA (임베딩 가속, 선택 사항)

---

## 5. 데이터 요구사항

### 5.1 메타데이터

**테이블 메타데이터**:
```python
{
    "table_name": "users",
    "description": "사용자 계정 정보",
    "business_terms": ["회원", "고객", "유저"],
    "columns": [
        {
            "name": "id",
            "type": "BIGINT",
            "description": "사용자 고유 식별자",
            "sample_values": [1, 2, 3]
        },
        ...
    ],
    "usage_examples": [
        "신규 가입자 수 조회",
        "활성 사용자 집계"
    ]
}
```

### 5.2 Few-shot 예제

**최소 요구사항**: 테이블당 3-5개

**예제 구조**:
```python
{
    "question": "지난달 신규 가입자 수는?",
    "sql_query": "SELECT COUNT(*) FROM users WHERE ...",
    "category": "aggregation",
    "difficulty": "easy",
    "success_rate": 0.95
}
```

### 5.3 지식 베이스 문서

**포맷**: Markdown, PDF, TXT

**메타데이터**:
```python
{
    "doc_id": "uuid",
    "title": "배포 가이드",
    "source": "Wiki",
    "url": "https://...",
    "last_updated": "2025-01-01",
    "tags": ["배포", "DevOps"]
}
```

---

## 6. 성능 요구사항

### 6.1 응답 시간

| 기능 | 목표 응답 시간 | 최대 허용 |
|------|----------------|-----------|
| Text-to-SQL | 30초 | 60초 |
| Data Discovery | 2초 | 5초 |
| Knowledge Discovery | 3초 | 10초 |
| Query Assistant | 5초 | 15초 |
| Vector 검색 | 0.5초 | 2초 |
| BM25 검색 | 0.01초 | 0.1초 |

### 6.2 처리량

- 동시 사용자: 50명
- 일일 쿼리: 1,000건
- 피크 시간 QPS: 10 queries/sec

### 6.3 정확도

- 쿼리 생성 정확도: 85% 이상
- 검색 정확도 (Top-5): 90% 이상
- 의도 분류 정확도: 95% 이상

---

## 7. 보안 요구사항

### 7.1 데이터 접근 제어

- 읽기 전용 DB 계정 사용
- SELECT 쿼리만 허용
- DELETE, DROP, TRUNCATE 금지

### 7.2 SQL Injection 방지

- 파라미터 바인딩 사용
- 입력 검증 및 이스케이프
- 쿼리 패턴 화이트리스트

### 7.3 민감 정보 보호

- 환경 변수로 API 키 관리
- DB 비밀번호 암호화
- 로그에 개인정보 미포함

---

## 8. 사용자 인터페이스

### 8.1 Streamlit UI 구성

**페이지 구조**:
```
┌─────────────────────────────────────┐
│  🤖 AI 데이터 분석 어시스턴트       │
├─────────────────────────────────────┤
│  [사이드바]                         │
│  - 대화 기록                        │
│  - 설정                             │
│  - 도움말                           │
├─────────────────────────────────────┤
│  [메인 영역]                        │
│  ┌───────────────────────────────┐ │
│  │ 💬 대화 내역                  │ │
│  ├───────────────────────────────┤ │
│  │ 사용자: 지난달 가입자는?      │ │
│  │ AI: [쿼리 생성 및 결과]       │ │
│  └───────────────────────────────┘ │
│                                     │
│  ┌───────────────────────────────┐ │
│  │ 💬 질문 입력...               │ │
│  │ [전송]                        │ │
│  └───────────────────────────────┘ │
└─────────────────────────────────────┘
```

### 8.2 주요 인터랙션

1. **질문 입력**: 텍스트 입력창
2. **응답 표시**: Markdown 렌더링
3. **쿼리 표시**: SQL 코드 블록 (복사 가능)
4. **결과 표시**: 데이터프레임 테이블
5. **피드백**: 👍 👎 버튼

---

## 9. 확장성 요구사항

### 9.1 단기 확장 (3개월)

- Multi-turn 대화 고도화
- 쿼리 자동 실행 옵션
- 결과 시각화 (차트)

### 9.2 중기 확장 (6개월)

- 스케줄 쿼리 (정기 리포트)
- 알림 설정 (임계값 기반)
- 협업 기능 (쿼리 공유)

### 9.3 장기 확장 (12개월)

- Analytics Assistant (자동 인사이트)
- Agent Ecosystem (전문 에이전트)
- 멀티 DB 지원 (MySQL, MongoDB)

---

## 부록

### A. 용어 정의

- **RAG**: Retrieval-Augmented Generation
- **Few-shot**: 소수 예제 기반 학습
- **BM25**: Best Matching 25 (키워드 검색 알고리즘)
- **RRF**: Reciprocal Rank Fusion
- **ERD**: Entity Relationship Diagram

### B. 참고 자료

- [물어보새 블로그 1부](https://techblog.woowahan.com/18144/)
- [물어보새 블로그 2부](https://techblog.woowahan.com/18362/)
- [물어보새 블로그 3부](https://techblog.woowahan.com/23273/)
- [WOOWACON 2024 발표](https://youtu.be/_QPhoKItI2k?si=AJ5e0LTT8lzRUKUX&t=2165)

---

**문서 버전**: 1.0.0
**최종 수정**: 2025-01-13
**승인자**: -
**상태**: Draft

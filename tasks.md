# AI 데이터 분석 어시스턴트 - 작업 분해 (Tasks Breakdown)

## 문서 정보

- **프로젝트명**: AI 데이터 분석 어시스턴트
- **버전**: 1.0.0
- **총 작업 수**: 150+ tasks
- **전체 기간**: 4-6주
- **우선순위**: P0 (필수) > P1 (중요) > P2 (선택)

---

## 📋 작업 추적 방식

각 작업은 다음 형식으로 관리:
- `[ ]` 미완료
- `[x]` 완료
- `[~]` 진행 중
- `[!]` 블로킹 이슈

---

## Phase 0: 환경 설정 (Week 1)

### 0.1 Python 환경 설정

**담당**: Backend Developer | **우선순위**: P0 | **예상 시간**: 2h

- [ ] Python 3.10+ 설치 확인
- [ ] 가상 환경 생성 (`python -m venv venv`)
- [ ] 가상 환경 활성화
- [ ] pip 업그레이드 (`pip install --upgrade pip`)
- [ ] requirements.txt 작성
- [ ] 패키지 설치 (`pip install -r requirements.txt`)
- [ ] 설치 확인 테스트

**산출물**: ✅ Python 가상 환경

---

### 0.2 requirements.txt 작성

**담당**: Backend Developer | **우선순위**: P0 | **예상 시간**: 1h

```txt
# LLM & Frameworks
langchain==0.1.0
langgraph==0.0.20
langchain-openai==0.0.5

# Vector DB & Search
chromadb==0.4.22
sentence-transformers==2.2.2
rank-bm25==0.2.2

# Database
psycopg2-binary==2.9.9
sqlparse==0.4.4

# UI
streamlit==1.28.0

# Utils
python-dotenv==1.0.0
tiktoken==0.5.2
pydantic==2.5.0

# Testing
pytest==7.4.3
pytest-benchmark==4.0.0

# Code Quality
flake8==6.1.0
black==23.12.0
mypy==1.7.1
```

**검증**:
- [ ] `pip install -r requirements.txt` 성공
- [ ] `python -c "import langchain"` 성공
- [ ] `python -c "import chromadb"` 성공

---

### 0.3 PostgreSQL 설정

**담당**: Data Engineer | **우선순위**: P0 | **예상 시간**: 4h

- [ ] Docker 설치 확인
- [ ] PostgreSQL Docker 이미지 pull
- [ ] Docker 컨테이너 실행 (포트 5432)
- [ ] 데이터베이스 생성
- [ ] 연결 테스트 (psql 또는 pgAdmin)
- [ ] .env 파일에 DB 연결 정보 저장
- [ ] Python에서 연결 테스트

**산출물**: ✅ PostgreSQL 실행 중

**Docker 명령어**:
```bash
docker run --name postgres-db \
  -e POSTGRES_PASSWORD=yourpassword \
  -e POSTGRES_DB=yourdb \
  -p 5432:5432 \
  -d postgres:15
```

**검증**:
```python
import psycopg2
conn = psycopg2.connect(
    host="localhost",
    database="yourdb",
    user="postgres",
    password="yourpassword"
)
print("✅ 연결 성공!")
conn.close()
```

---

### 0.4 샘플 데이터 생성

**담당**: Data Engineer | **우선순위**: P0 | **예상 시간**: 4h

- [ ] users 테이블 스키마 작성
- [ ] orders 테이블 스키마 작성
- [ ] users 샘플 데이터 10,000건 생성
- [ ] orders 샘플 데이터 50,000건 생성
- [ ] 데이터 무결성 확인 (FK 제약)
- [ ] 인덱스 생성 (created_at, user_id 등)
- [ ] 샘플 쿼리 실행 테스트

**산출물**: ✅ 샘플 DB 완성

**SQL 스크립트** (`data/sample_data.sql`):
```sql
-- users 테이블
CREATE TABLE users (
    id BIGSERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'active'
);

-- orders 테이블
CREATE TABLE orders (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT REFERENCES users(id),
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_amount DECIMAL(10, 2),
    status VARCHAR(20)
);

-- 샘플 데이터 생성
INSERT INTO users (email, created_at, status)
SELECT
    'user' || i || '@example.com',
    CURRENT_TIMESTAMP - (random() * INTERVAL '365 days'),
    CASE WHEN random() > 0.1 THEN 'active' ELSE 'inactive' END
FROM generate_series(1, 10000) i;

INSERT INTO orders (user_id, order_date, total_amount, status)
SELECT
    floor(random() * 10000 + 1)::BIGINT,
    CURRENT_TIMESTAMP - (random() * INTERVAL '180 days'),
    (random() * 500 + 10)::DECIMAL(10, 2),
    CASE
        WHEN random() < 0.7 THEN 'completed'
        WHEN random() < 0.9 THEN 'pending'
        ELSE 'cancelled'
    END
FROM generate_series(1, 50000) i;
```

---

### 0.5 OpenAI API 설정

**담당**: Backend Developer | **우선순위**: P0 | **예상 시간**: 1h

- [ ] OpenAI 계정 생성
- [ ] API 키 발급 (https://platform.openai.com/api-keys)
- [ ] `.env.example` 작성
- [ ] `.env` 파일 생성 (gitignore에 추가)
- [ ] API 키 `.env`에 저장
- [ ] 연결 테스트 코드 작성
- [ ] API 호출 성공 확인

**`.env` 예시**:
```
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/yourdb
```

**테스트 코드**:
```python
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = AzureChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4o",
    temperature=0
)

response = llm.invoke("Hello!")
print(response.content)
```

---

### 0.6 프로젝트 구조 생성

**담당**: Backend Developer | **우선순위**: P0 | **예상 시간**: 1h

- [ ] 프로젝트 루트 디렉토리 생성
- [ ] `src/` 디렉토리 구조 생성
- [ ] `data/` 디렉토리 구조 생성
- [ ] `tests/` 디렉토리 구조 생성
- [ ] `docs/` 디렉토리 구조 생성
- [ ] `.gitignore` 작성
- [ ] Git 저장소 초기화
- [ ] README.md 기본 작성

**디렉토리 구조**:
```
ai-data-assistant/
├── src/
│   ├── __init__.py
│   ├── chains/             # LangChain 체인
│   │   ├── __init__.py
│   │   ├── text_to_sql_chain.py
│   │   ├── data_discovery_chain.py
│   │   ├── knowledge_discovery_chain.py
│   │   └── query_validator.py
│   ├── rag/                # RAG 시스템
│   │   ├── __init__.py
│   │   ├── vector_store.py
│   │   ├── bm25_search.py
│   │   └── hybrid_search.py
│   ├── database/           # DB 연결
│   │   ├── __init__.py
│   │   ├── connection.py
│   │   └── metadata_collector.py
│   ├── memory/             # 대화 메모리
│   │   ├── __init__.py
│   │   └── conversation_memory.py
│   └── ui/                 # Streamlit UI
│       ├── __init__.py
│       └── app.py
├── data/
│   ├── metadata/           # 테이블 메타데이터
│   ├── fewshot/            # Few-shot 예제
│   │   └── examples.json
│   └── documents/          # 지식베이스 문서
├── tests/
│   ├── __init__.py
│   ├── test_rag.py
│   ├── test_text_to_sql.py
│   └── test_chains.py
├── docs/
├── chromadb_data/          # ChromaDB 저장소
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

### 0.7 Git 설정

**담당**: Backend Developer | **우선순위**: P0 | **예상 시간**: 30min

- [ ] `git init` 실행
- [ ] `.gitignore` 작성
- [ ] 초기 커밋 (`git add . && git commit -m "Initial commit"`)
- [ ] GitHub/GitLab 저장소 생성 (선택)
- [ ] 원격 저장소 연결 (선택)
- [ ] 첫 push (선택)

**`.gitignore`**:
```
# Python
venv/
__pycache__/
*.pyc
*.pyo
.pytest_cache/

# Environment
.env

# Data
chromadb_data/
*.db
*.sqlite

# IDE
.vscode/
.idea/
```

---

## Phase 1: RAG 시스템 (Week 2)

### 1.1 VectorStore 구현

**담당**: Backend Developer | **우선순위**: P0 | **예상 시간**: 16h

- [ ] `src/rag/vector_store.py` 파일 생성
- [ ] ChromaDB 클라이언트 설정
- [ ] sentence-transformers 모델 로드
- [ ] `VectorStore` 클래스 작성
- [ ] `add_documents()` 메서드 구현
- [ ] `search()` 메서드 구현
- [ ] 단위 테스트 작성
- [ ] 통합 테스트 작성

**산출물**: ✅ `vector_store.py`

**검증**:
```python
from src.rag.vector_store import VectorStore

vs = VectorStore()
vs.add_documents([
    {"id": "1", "content": "users 테이블은 사용자 정보를 저장합니다."}
])
results = vs.search("사용자 테이블", top_k=1)
assert len(results['ids'][0]) == 1
```

---

### 1.2 메타데이터 수집기 구현

**담당**: Data Engineer | **우선순위**: P0 | **예상 시간**: 8h

- [ ] `src/database/metadata_collector.py` 생성
- [ ] `MetadataCollector` 클래스 작성
- [ ] `get_tables()` 메서드 구현
- [ ] `get_columns()` 메서드 구현
- [ ] `get_sample_values()` 메서드 구현
- [ ] `build_table_document()` 메서드 구현
- [ ] 테이블 설명 수동 작성 (data/metadata/)
- [ ] 메타데이터 Vector DB 저장

**산출물**: ✅ 테이블 메타데이터 JSON

**메타데이터 예시** (`data/metadata/users.json`):
```json
{
    "table_name": "users",
    "description": "사용자 계정 정보를 저장하는 테이블",
    "business_terms": ["회원", "고객", "유저"],
    "columns": [
        {
            "name": "id",
            "type": "BIGINT",
            "description": "사용자 고유 식별자",
            "sample_values": [1, 2, 3]
        },
        {
            "name": "email",
            "type": "VARCHAR",
            "description": "사용자 이메일 주소",
            "sample_values": ["user1@example.com", "user2@example.com"]
        }
    ],
    "usage_examples": [
        "신규 가입자 수 조회",
        "활성 사용자 집계",
        "이메일 도메인별 분포"
    ]
}
```

---

### 1.3 BM25Search 구현

**담당**: Backend Developer | **우선순위**: P0 | **예상 시간**: 8h

- [ ] `src/rag/bm25_search.py` 생성
- [ ] `BM25Search` 클래스 작성
- [ ] `build_index()` 메서드 구현
- [ ] `load_index()` 메서드 구현
- [ ] `tokenize()` 메서드 구현 (한글 지원)
- [ ] `search()` 메서드 구현
- [ ] 인덱스 저장/로드 테스트
- [ ] 검색 성능 테스트

**산출물**: ✅ `bm25_search.py`

---

### 1.4 HybridSearch 구현

**담당**: Backend Developer | **우선순위**: P0 | **예상 시간**: 16h

- [ ] `src/rag/hybrid_search.py` 생성
- [ ] `HybridSearch` 클래스 작성
- [ ] Vector 검색 통합
- [ ] BM25 검색 통합
- [ ] RRF (Reciprocal Rank Fusion) 구현
- [ ] 스코어 정규화 및 결합
- [ ] 검색 정확도 테스트
- [ ] 성능 벤치마크

**산출물**: ✅ `hybrid_search.py`

**검증**:
```python
from src.rag.hybrid_search import HybridSearch

hs = HybridSearch()
results = hs.search("사용자 테이블", top_k=5)
assert len(results) == 5
assert "users" in results[0]['content'].lower()
```

---

### 1.5 RAG 테스트

**담당**: QA Engineer | **우선순위**: P0 | **예상 시간**: 8h

- [ ] `tests/test_rag.py` 작성
- [ ] Vector 검색 단위 테스트
- [ ] BM25 검색 단위 테스트
- [ ] 하이브리드 검색 단위 테스트
- [ ] 검색 정확도 테스트 (Top-5 accuracy)
- [ ] 성능 테스트 (응답 시간 < 1초)
- [ ] Edge case 테스트
- [ ] 테스트 커버리지 80% 이상 확보

**테스트 케이스**:
```python
def test_vector_search():
    """Vector 검색 정확도"""
    pass

def test_bm25_search():
    """BM25 검색 정확도"""
    pass

def test_hybrid_search():
    """하이브리드 검색 정확도"""
    pass

def test_search_performance():
    """검색 응답 시간 < 1초"""
    pass
```

---

## Phase 2: Text-to-SQL (Week 3)

### 2.1 Few-shot 예제 작성

**담당**: Data Engineer | **우선순위**: P0 | **예상 시간**: 8h

- [ ] `data/fewshot/examples.json` 생성
- [ ] 집계 쿼리 예제 5개 작성
- [ ] JOIN 쿼리 예제 5개 작성
- [ ] 날짜 필터링 예제 3개 작성
- [ ] GROUP BY 예제 3개 작성
- [ ] 복잡 쿼리 예제 4개 작성
- [ ] 각 예제에 카테고리/난이도 태깅
- [ ] 예제 검증 (실제 실행 가능)

**산출물**: ✅ `examples.json` (최소 20개)

---

### 2.2 TextToSQLChain 구현

**담당**: Backend Developer | **우선순위**: P0 | **예상 시간**: 24h

- [ ] `src/chains/text_to_sql_chain.py` 생성
- [ ] `TextToSQLChain` 클래스 작성
- [ ] LLM 초기화 (GPT-4o)
- [ ] SQLDatabase 연결
- [ ] 프롬프트 템플릿 작성
- [ ] `select_relevant_tables()` 구현 (RAG 활용)
- [ ] `retrieve_examples()` 구현 (Few-shot 검색)
- [ ] `generate_sql()` 메서드 구현
- [ ] 단위 테스트 작성

**산출물**: ✅ `text_to_sql_chain.py`

---

### 2.3 QueryValidator 구현

**담당**: Backend Developer | **우선순위**: P0 | **예상 시간**: 8h

- [ ] `src/chains/query_validator.py` 생성
- [ ] `QueryValidator` 클래스 작성
- [ ] `validate_syntax()` 메서드 (sqlparse)
- [ ] `validate_existence()` 메서드 (테이블/칼럼 확인)
- [ ] `validate_security()` 메서드 (SQL Injection)
- [ ] `auto_correct()` 메서드 (오타 수정)
- [ ] `validate()` 통합 메서드
- [ ] 테스트 작성

**산출물**: ✅ `query_validator.py`

---

### 2.4 Text-to-SQL 통합 테스트

**담당**: QA Engineer | **우선순위**: P0 | **예상 시간**: 16h

- [ ] `tests/test_text_to_sql.py` 작성
- [ ] 집계 쿼리 테스트 (10개)
- [ ] JOIN 쿼리 테스트 (10개)
- [ ] 날짜 필터링 테스트 (5개)
- [ ] 복잡 쿼리 테스트 (5개)
- [ ] 검증 시스템 테스트
- [ ] E2E 테스트 (질문 → SQL → 실행)
- [ ] 정확도 측정 (목표: 85%)

**산출물**: ✅ 테스트 리포트

---

## Phase 3: 추가 기능 (Week 4)

### 3.1 Data Discovery 구현

**담당**: Backend Developer | **우선순위**: P1 | **예상 시간**: 16h

- [ ] `src/chains/data_discovery_chain.py` 생성
- [ ] `DataDiscoveryChain` 클래스 작성
- [ ] `list_tables()` 메서드
- [ ] `explain_table()` 메서드
- [ ] `explain_column()` 메서드
- [ ] `get_relationships()` 메서드 (ERD)
- [ ] 테스트 작성

---

### 3.2 ConversationMemory 구현

**담당**: Backend Developer | **우선순위**: P1 | **예상 시간**: 16h

- [ ] `src/memory/conversation_memory.py` 생성
- [ ] SQLite DB 스키마 작성
- [ ] `ConversationMemory` 클래스 작성
- [ ] `save_conversation()` 메서드
- [ ] `get_recent_conversations()` 메서드
- [ ] `clear_session()` 메서드
- [ ] 세션 관리 기능
- [ ] 테스트 작성

---

### 3.3 Knowledge Discovery 구현

**담당**: Backend Developer | **우선순위**: P2 | **예상 시간**: 24h

- [ ] `src/chains/knowledge_discovery_chain.py` 생성
- [ ] 문서 로더 구현 (Markdown, PDF)
- [ ] 문서 Vector DB 저장
- [ ] `KnowledgeDiscoveryChain` 클래스 작성
- [ ] `answer_question()` 메서드
- [ ] 출처 표시 기능
- [ ] 테스트 작성

---

## Phase 4: UI 구축 (Week 5)

### 4.1 Streamlit 기본 UI

**담당**: Frontend Developer | **우선순위**: P0 | **예상 시간**: 16h

- [ ] `src/ui/app.py` 생성
- [ ] 페이지 레이아웃 구성
- [ ] 타이틀 및 헤더
- [ ] 사이드바 설정 UI
- [ ] 대화 기록 표시 영역
- [ ] 입력창 구현
- [ ] 세션 상태 관리
- [ ] 기본 스타일링

---

### 4.2 Text-to-SQL UI 통합

**담당**: Frontend Developer | **우선순위**: P0 | **예상 시간**: 16h

- [ ] 질문 입력 처리
- [ ] Text-to-SQL Chain 호출
- [ ] SQL 쿼리 표시 (코드 블록)
- [ ] 쿼리 복사 버튼
- [ ] 쿼리 실행 버튼 (선택)
- [ ] 결과 테이블 표시
- [ ] 에러 처리 및 표시

---

### 4.3 추가 기능 UI

**담당**: Frontend Developer | **우선순위**: P1 | **예상 시간**: 8h

- [ ] Data Discovery 탭
- [ ] Knowledge Discovery 탭
- [ ] 대화 기록 사이드바
- [ ] 설정 페이지
- [ ] 피드백 버튼 (👍 👎)

---

### 4.4 UI 테스트

**담당**: QA Engineer | **우선순위**: P1 | **예상 시간**: 8h

- [ ] 사용자 플로우 테스트
- [ ] 반응형 레이아웃 확인
- [ ] 에러 핸들링 확인
- [ ] 사용성 테스트
- [ ] 크로스 브라우저 테스트

---

## Phase 5: 테스팅 (Week 6)

### 5.1 단위 테스트

**담당**: QA Engineer | **우선순위**: P0 | **예상 시간**: 16h

- [ ] 전체 모듈 단위 테스트 작성
- [ ] 테스트 커버리지 80% 이상
- [ ] pytest 실행 및 리포트
- [ ] 실패 테스트 수정

---

### 5.2 통합 테스트

**담당**: QA Engineer | **우선순위**: P0 | **예상 시간**: 16h

- [ ] E2E 시나리오 작성
- [ ] 전체 플로우 테스트
- [ ] 성능 테스트 (응답 시간)
- [ ] 부하 테스트 (동시 사용자)

---

### 5.3 성능 최적화

**담당**: Backend Developer | **우선순위**: P1 | **예상 시간**: 16h

- [ ] 프로파일링 실행
- [ ] 병목 지점 식별
- [ ] BM25 사전 계산
- [ ] 캐싱 구현
- [ ] 병렬 처리 적용
- [ ] 최적화 전후 비교

---

## Phase 6: 배포 (Optional)

### 6.1 Docker 설정

**담당**: DevOps | **우선순위**: P2 | **예상 시간**: 8h

- [ ] `Dockerfile` 작성
- [ ] `docker-compose.yml` 작성
- [ ] 멀티 스테이지 빌드
- [ ] 이미지 빌드 테스트
- [ ] 컨테이너 실행 테스트

---

### 6.2 배포 문서

**담당**: DevOps | **우선순위**: P2 | **예상 시간**: 4h

- [ ] 배포 가이드 작성
- [ ] 환경 변수 문서화
- [ ] 트러블슈팅 가이드
- [ ] 모니터링 설정 가이드

---

## 마일스톤 체크리스트

### M1: 개발 환경 구축 (Week 1 완료)
- [ ] Python 환경 설정
- [ ] PostgreSQL 설정
- [ ] 샘플 데이터 생성
- [ ] OpenAI API 설정
- [ ] 프로젝트 구조 생성

### M2: RAG 시스템 완성 (Week 2 완료)
- [ ] VectorStore 구현
- [ ] BM25Search 구현
- [ ] HybridSearch 구현
- [ ] 검색 정확도 90% 이상

### M3: MVP 완성 (Week 3 완료)
- [ ] Text-to-SQL Chain 구현
- [ ] Query Validator 구현
- [ ] Few-shot 예제 20개
- [ ] 쿼리 생성 정확도 85% 이상

### M4: 기능 확장 (Week 4 완료)
- [ ] Data Discovery 구현
- [ ] Conversation Memory 구현
- [ ] Knowledge Discovery 구현 (선택)

### M5: UI 완성 (Week 5 완료)
- [ ] Streamlit 앱 기본 UI
- [ ] Text-to-SQL UI 통합
- [ ] 사용성 테스트 통과

### M6: 프로덕션 배포 (Week 6 완료)
- [ ] 전체 테스트 통과
- [ ] 성능 최적화 완료
- [ ] Docker 배포 (선택)

---

## 우선순위 매트릭스

### P0 (필수 - MVP)
- Python 환경 설정
- PostgreSQL 설정
- OpenAI API 설정
- RAG 시스템 (Vector + BM25)
- Text-to-SQL Chain
- Query Validator
- Streamlit UI (기본)

### P1 (중요)
- Data Discovery
- Conversation Memory
- UI 개선 (피드백, 기록)
- 성능 최적화
- 테스트 커버리지 80%

### P2 (선택)
- Knowledge Discovery
- Docker 배포
- 고급 UI 기능
- 모니터링 대시보드

---

## 리스크 및 블로킹 이슈 추적

| 이슈 | 우선순위 | 상태 | 담당 | 해결 방안 |
|------|----------|------|------|----------|
| LLM API 비용 초과 | 높음 | 모니터링 | Backend | 캐싱, 프롬프트 압축 |
| 쿼리 정확도 부족 | 높음 | 진행 중 | Backend | Few-shot 예제 확충 |
| 응답 시간 초과 | 중간 | 해결 | Backend | BM25 사전 계산 |
| ChromaDB 메모리 | 낮음 | 모니터링 | Backend | HNSW 최적화 |

---

**문서 버전**: 1.0.0
**최종 수정**: 2025-01-13
**상태**: Active

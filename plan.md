# AI 데이터 분석 어시스턴트 - 구현 계획서

## 문서 정보

- **프로젝트명**: AI 데이터 분석 어시스턴트
- **버전**: 1.0.0
- **작성일**: 2025년 1월
- **전체 기간**: 4-6주
- **예상 공수**: 240-360 시간
- **기술 스택**: 100% 오픈소스 (LLM API 제외)

---

## 📋 목차

1. [프로젝트 로드맵](#1-프로젝트-로드맵)
2. [Phase 0: 환경 설정](#2-phase-0-환경-설정-1주)
3. [Phase 1: RAG 시스템](#3-phase-1-rag-시스템-1주)
4. [Phase 2: Text-to-SQL](#4-phase-2-text-to-sql-1주)
5. [Phase 3: 추가 기능](#5-phase-3-추가-기능-1주)
6. [Phase 4: UI 구축](#6-phase-4-ui-구축-1주)
7. [Phase 5: 테스팅](#7-phase-5-테스팅-1주)
8. [Phase 6: 배포](#8-phase-6-배포-선택)
9. [리스크 관리](#9-리스크-관리)
10. [품질 관리](#10-품질-관리)

---

## 1. 프로젝트 로드맵

### 1.1 전체 일정

```mermaid
gantt
    title AI 데이터 분석 어시스턴트 구현 일정
    dateFormat  YYYY-MM-DD
    section 준비
    환경 설정           :p0, 2025-01-15, 1w
    section 핵심 기능
    RAG 시스템          :p1, after p0, 1w
    Text-to-SQL         :p2, after p1, 1w
    section 확장 기능
    추가 기능 개발      :p3, after p2, 1w
    section UI/배포
    Streamlit UI        :p4, after p3, 1w
    테스팅 및 최적화    :p5, after p4, 1w
    배포 (선택)         :p6, after p5, 3d
```

### 1.2 마일스톤

| 마일스톤 | 완료 기준 | 기한 |
|----------|-----------|------|
| **M1**: 개발 환경 구축 | Docker, DB, API 키 설정 완료 | Week 1 |
| **M2**: RAG 시스템 완성 | Vector + BM25 검색 작동 | Week 2 |
| **M3**: MVP 완성 | Text-to-SQL 기본 기능 작동 | Week 3 |
| **M4**: 기능 확장 | Data Discovery, Memory 완성 | Week 4 |
| **M5**: UI 완성 | Streamlit 인터페이스 완성 | Week 5 |
| **M6**: 프로덕션 배포 | Docker Compose 배포 완료 | Week 6 |

### 1.3 팀 구성 (권장)

| 역할 | 인원 | 책임 |
|------|------|------|
| **Backend 개발자** | 1-2명 | RAG, Text-to-SQL, API |
| **Frontend 개발자** | 1명 | Streamlit UI |
| **데이터 엔지니어** | 1명 | 메타데이터, Few-shot |
| **QA 엔지니어** | 1명 | 테스팅, 품질 보증 |

> **최소 구성**: 1명 (Full-stack)으로도 가능 (6주 소요)

---

## 2. Phase 0: 환경 설정 (1주)

### 2.1 목표

프로젝트 개발을 위한 기본 환경 구축 완료

### 2.2 작업 목록

#### 2.2.1 Python 환경 설정

**소요 시간**: 2시간

```bash
# 1. Python 3.10+ 설치 확인
python --version  # 3.10 이상

# 2. 가상 환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 필수 패키지 설치
pip install -r requirements.txt

# 4. 환경 변수 설정
cp .env.example .env
# .env 파일 편집 (API 키 등)
```

#### 2.2.2 데이터베이스 설정

**소요 시간**: 4시간

**PostgreSQL 설치**:
```bash
# Docker로 PostgreSQL 실행
docker run --name postgres-db \
  -e POSTGRES_PASSWORD=your_password \
  -e POSTGRES_DB=your_database \
  -p 5432:5432 \
  -d postgres:15
```

**테스트 데이터 준비**:
```sql
-- 샘플 테이블 생성
CREATE TABLE users (
    id BIGSERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'active'
);

CREATE TABLE orders (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT REFERENCES users(id),
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_amount DECIMAL(10, 2),
    status VARCHAR(20)
);

-- 샘플 데이터 삽입
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

#### 2.2.3 LLM API 설정

**소요 시간**: 1시간

1. **OpenAI API 키 발급**:
   - https://platform.openai.com/api-keys 접속
   - API 키 생성
   - `.env` 파일에 저장

2. **API 테스트**:
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

# 테스트
response = llm.invoke("Hello, world!")
print(response.content)
```

#### 2.2.4 프로젝트 구조 생성

**소요 시간**: 1시간

```bash
mkdir -p ai-data-assistant/{src,data,tests,docs}
cd ai-data-assistant

# 디렉토리 구조
tree -L 2
# ai-data-assistant/
# ├── src/
# │   ├── chains/          # LangChain 체인
# │   ├── rag/             # RAG 시스템
# │   ├── database/        # DB 연결
# │   ├── memory/          # 대화 메모리
# │   └── ui/              # Streamlit UI
# ├── data/
# │   ├── metadata/        # 테이블 메타데이터
# │   ├── fewshot/         # Few-shot 예제
# │   └── documents/       # 지식베이스 문서
# ├── tests/
# │   ├── test_rag.py
# │   ├── test_text_to_sql.py
# │   └── test_chains.py
# ├── docs/
# ├── .env
# ├── requirements.txt
# ├── docker-compose.yml
# └── README.md
```

### 2.3 체크리스트

- [ ] Python 3.10+ 설치
- [ ] 가상 환경 생성 및 활성화
- [ ] requirements.txt 작성 및 패키지 설치
- [ ] PostgreSQL Docker 컨테이너 실행
- [ ] 샘플 데이터 생성 (users, orders 테이블)
- [ ] OpenAI API 키 발급 및 설정
- [ ] API 연결 테스트 성공
- [ ] 프로젝트 디렉토리 구조 생성
- [ ] Git 저장소 초기화
- [ ] README.md 작성

### 2.4 산출물

- ✅ 개발 환경 완성
- ✅ PostgreSQL DB + 샘플 데이터
- ✅ OpenAI API 연동
- ✅ 프로젝트 구조

---

## 3. Phase 1: RAG 시스템 (1주)

### 3.1 목표

Vector DB + BM25 하이브리드 검색 시스템 구축

### 3.2 작업 목록

#### 3.2.1 Vector Store 구축 (2일)

**소요 시간**: 16시간

**Day 1**: ChromaDB 설정 및 테스트

```python
# src/rag/vector_store.py

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class VectorStore:
    def __init__(self, persist_directory="./chromadb_data"):
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))

        self.embed_model = SentenceTransformer(
            'paraphrase-multilingual-MiniLM-L12-v2'
        )

        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"description": "테이블 메타데이터 및 문서"}
        )

    def add_documents(self, documents: List[Dict]):
        """문서 추가"""
        embeddings = self.embed_model.encode(
            [doc["content"] for doc in documents],
            normalize_embeddings=True
        )

        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=[doc["content"] for doc in documents],
            metadatas=[doc.get("metadata", {}) for doc in documents],
            ids=[doc.get("id", str(i)) for i, doc in enumerate(documents)]
        )

    def search(self, query: str, top_k: int = 5):
        """유사도 검색"""
        query_embedding = self.embed_model.encode(
            query, normalize_embeddings=True
        )

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        return results
```

**Day 2**: 메타데이터 수집 및 인덱싱

```python
# src/database/metadata_collector.py

import psycopg2
from typing import Dict, List

class MetadataCollector:
    def __init__(self, db_connection):
        self.conn = db_connection

    def collect_all_metadata(self) -> List[Dict]:
        """모든 테이블 메타데이터 수집"""
        tables = self.get_tables()
        metadata_docs = []

        for table in tables:
            doc = self.build_table_document(table)
            metadata_docs.append(doc)

        return metadata_docs

    def get_tables(self) -> List[str]:
        """테이블 목록 조회"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        return [row[0] for row in cursor.fetchall()]

    def build_table_document(self, table_name: str) -> Dict:
        """테이블 문서 생성"""
        # 칼럼 정보 수집
        columns = self.get_columns(table_name)

        # 샘플 값 수집
        sample_values = self.get_sample_values(table_name)

        # 문서 구성
        content = f"""
테이블: {table_name}
설명: [수동 입력 필요]
칼럼: {', '.join([f"{c['name']} ({c['type']})" for c in columns])}
샘플 값: {sample_values}
"""

        return {
            "id": f"table_{table_name}",
            "content": content,
            "metadata": {
                "type": "table",
                "table_name": table_name
            }
        }

    def get_columns(self, table_name: str) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute(f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
        """)
        return [{"name": row[0], "type": row[1]} for row in cursor.fetchall()]

    def get_sample_values(self, table_name: str, limit=3) -> str:
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
        rows = cursor.fetchall()
        return str(rows)

# 실행
conn = psycopg2.connect("postgresql://user:pass@localhost/db")
collector = MetadataCollector(conn)
metadata = collector.collect_all_metadata()

# Vector DB에 저장
vector_store = VectorStore()
vector_store.add_documents(metadata)
```

#### 3.2.2 BM25 검색 구현 (1일)

**소요 시간**: 8시간

```python
# src/rag/bm25_search.py

from rank_bm25 import BM25Okapi
import pickle
from typing import List, Dict

class BM25Search:
    def __init__(self, corpus_path="bm25_corpus.pkl"):
        self.corpus_path = corpus_path
        self.bm25 = None
        self.documents = []

    def build_index(self, documents: List[Dict]):
        """BM25 인덱스 구축"""
        self.documents = documents
        tokenized_corpus = [self.tokenize(doc["content"]) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 저장
        with open(self.corpus_path, "wb") as f:
            pickle.dump((self.bm25, self.documents), f)

    def load_index(self):
        """저장된 인덱스 로드"""
        with open(self.corpus_path, "rb") as f:
            self.bm25, self.documents = pickle.load(f)

    def tokenize(self, text: str) -> List[str]:
        """간단한 토크나이징"""
        return text.lower().split()

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """검색"""
        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # 상위 k개
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "document": self.documents[idx],
                "score": float(scores[idx])
            })

        return results
```

#### 3.2.3 하이브리드 검색 구현 (2일)

**소요 시간**: 16시간

```python
# src/rag/hybrid_search.py

from typing import List, Dict
from .vector_store import VectorStore
from .bm25_search import BM25Search

class HybridSearch:
    def __init__(self):
        self.vector_store = VectorStore()
        self.bm25_search = BM25Search()

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """하이브리드 검색 (RRF)"""

        # 1. Vector 검색
        vector_results = self.vector_store.search(query, top_k=10)

        # 2. BM25 검색
        bm25_results = self.bm25_search.search(query, top_k=10)

        # 3. Reciprocal Rank Fusion
        scores = {}

        # Vector 점수
        for i, doc_id in enumerate(vector_results['ids'][0]):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (i + 1 + 60)

        # BM25 점수
        for i, result in enumerate(bm25_results):
            doc_id = result['document']['id']
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (i + 1 + 60)

        # 4. 정렬
        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # 5. 결과 반환
        final_results = []
        for doc_id, score in sorted_ids:
            # 문서 내용 조회
            doc_content = self._get_document(doc_id, vector_results, bm25_results)
            final_results.append({
                "id": doc_id,
                "content": doc_content,
                "rrf_score": score
            })

        return final_results

    def _get_document(self, doc_id, vector_results, bm25_results):
        # Vector 결과에서 찾기
        if doc_id in vector_results['ids'][0]:
            idx = vector_results['ids'][0].index(doc_id)
            return vector_results['documents'][0][idx]

        # BM25 결과에서 찾기
        for result in bm25_results:
            if result['document']['id'] == doc_id:
                return result['document']['content']

        return None
```

### 3.3 테스트

```python
# tests/test_rag.py

import unittest
from src.rag.hybrid_search import HybridSearch

class TestRAG(unittest.TestCase):
    def setUp(self):
        self.search = HybridSearch()

    def test_vector_search(self):
        results = self.search.vector_store.search("사용자 테이블", top_k=3)
        self.assertGreater(len(results['ids'][0]), 0)

    def test_bm25_search(self):
        results = self.search.bm25_search.search("사용자 테이블", top_k=3)
        self.assertGreater(len(results), 0)

    def test_hybrid_search(self):
        results = self.search.search("사용자 테이블", top_k=5)
        self.assertEqual(len(results), 5)
        self.assertIn("users", results[0]['content'])

if __name__ == '__main__':
    unittest.main()
```

### 3.4 체크리스트

- [ ] ChromaDB 설정 및 테스트
- [ ] sentence-transformers 모델 로드
- [ ] 메타데이터 수집 스크립트 작성
- [ ] 테이블 메타데이터 Vector DB 저장
- [ ] BM25 인덱스 구축
- [ ] 하이브리드 검색 구현 (RRF)
- [ ] 검색 정확도 테스트
- [ ] 단위 테스트 작성

### 3.5 산출물

- ✅ RAG 시스템 완성
- ✅ Vector DB 구축
- ✅ BM25 검색 엔진
- ✅ 하이브리드 검색

---

## 4. Phase 2: Text-to-SQL (1주)

### 4.1 목표

자연어 질문을 SQL 쿼리로 변환하는 핵심 기능 구현

### 4.2 작업 목록

#### 4.2.1 Few-shot 예제 준비 (1일)

**소요 시간**: 8시간

```python
# data/fewshot/examples.json

[
    {
        "question": "지난달 신규 가입자 수는?",
        "sql_query": "SELECT COUNT(*) FROM users WHERE created_at >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month') AND created_at < DATE_TRUNC('month', CURRENT_DATE);",
        "category": "aggregation",
        "difficulty": "easy"
    },
    {
        "question": "이번 주 주문 금액 합계는?",
        "sql_query": "SELECT SUM(total_amount) FROM orders WHERE order_date >= DATE_TRUNC('week', CURRENT_DATE);",
        "category": "aggregation",
        "difficulty": "easy"
    },
    {
        "question": "사용자별 평균 주문 금액은?",
        "sql_query": "SELECT u.email, AVG(o.total_amount) as avg_order FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.email;",
        "category": "join",
        "difficulty": "medium"
    }
]
```

#### 4.2.2 Text-to-SQL Chain 구현 (3일)

**소요 시간**: 24시간

```python
# src/chains/text_to_sql_chain.py

from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from typing import Dict
import os

class TextToSQLChain:
    def __init__(self, db_uri):
        self.llm = AzureChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-4o",
            temperature=0
        )

        self.db = SQLDatabase.from_uri(db_uri)

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert SQL assistant.
Generate syntactically correct PostgreSQL queries based on the question and schema.
Return ONLY the SQL query without explanation."""),
            ("user", """Schema:
{schema}

Examples:
{examples}

Question: {question}

SQL:""")
        ])

    def select_relevant_tables(self, question: str, top_k=3) -> list:
        """질문과 관련된 테이블 선별"""
        # RAG 검색 활용
        from src.rag.hybrid_search import HybridSearch
        search = HybridSearch()
        results = search.search(question, top_k=top_k)

        tables = []
        for result in results:
            if "table_name" in result.get("metadata", {}):
                tables.append(result["metadata"]["table_name"])

        return tables[:top_k]

    def retrieve_examples(self, question: str, top_k=3) -> str:
        """유사 예제 검색"""
        # Few-shot 예제에서 검색
        import json

        with open("data/fewshot/examples.json") as f:
            examples = json.load(f)

        # 간단한 키워드 매칭 (실제로는 임베딩 유사도 사용)
        matched_examples = []
        for ex in examples:
            if any(word in question for word in ex["question"].split()):
                matched_examples.append(ex)

        # 포맷팅
        example_str = "\n".join([
            f"Q: {ex['question']}\nSQL: {ex['sql_query']}\n"
            for ex in matched_examples[:top_k]
        ])

        return example_str

    def generate_sql(self, question: str) -> Dict:
        """SQL 생성"""
        # 1. 관련 테이블 선별
        relevant_tables = self.select_relevant_tables(question)

        # 2. 스키마 가져오기
        schema = self.db.get_table_info(relevant_tables)

        # 3. Few-shot 예제 검색
        examples = self.retrieve_examples(question)

        # 4. 프롬프트 구성
        prompt = self.prompt_template.format_messages(
            schema=schema,
            examples=examples,
            question=question
        )

        # 5. LLM 호출
        response = self.llm.invoke(prompt)

        return {
            "question": question,
            "sql_query": response.content.strip(),
            "selected_tables": relevant_tables
        }
```

#### 4.2.3 쿼리 검증 (1일)

**소요 시간**: 8시간

```python
# src/chains/query_validator.py

import sqlparse
from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword, DML

class QueryValidator:
    def __init__(self, db_connection):
        self.conn = db_connection

    def validate(self, query: str) -> Dict:
        """쿼리 검증"""

        # 1. 문법 검증
        is_valid, syntax_error = self.validate_syntax(query)
        if not is_valid:
            return {"valid": False, "error": syntax_error}

        # 2. 테이블/칼럼 존재 확인
        is_valid, existence_error = self.validate_existence(query)
        if not is_valid:
            return {"valid": False, "error": existence_error}

        # 3. 실행 계획 확인
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"EXPLAIN {query}")
            plan = cursor.fetchall()
        except Exception as e:
            return {"valid": False, "error": f"실행 계획 오류: {str(e)}"}

        return {"valid": True, "message": "검증 통과"}

    def validate_syntax(self, query: str) -> tuple:
        """문법 검증"""
        try:
            parsed = sqlparse.parse(query)
            if not parsed:
                return False, "쿼리 파싱 실패"

            stmt = parsed[0]
            if stmt.get_type() != 'SELECT':
                return False, "SELECT 쿼리만 허용"

            return True, None
        except Exception as e:
            return False, f"문법 오류: {str(e)}"

    def validate_existence(self, query: str) -> tuple:
        """테이블/칼럼 존재 확인"""
        # 간단한 구현 (실제로는 더 정교한 파싱 필요)
        return True, None
```

#### 4.2.4 통합 테스트 (2일)

**소요 시간**: 16시간

```python
# tests/test_text_to_sql.py

import unittest
from src.chains.text_to_sql_chain import TextToSQLChain
from src.chains.query_validator import QueryValidator

class TestTextToSQL(unittest.TestCase):
    def setUp(self):
        self.chain = TextToSQLChain("postgresql://user:pass@localhost/db")
        self.validator = QueryValidator(self.chain.db._engine.raw_connection())

    def test_simple_aggregation(self):
        result = self.chain.generate_sql("지난달 신규 가입자 수는?")
        self.assertIn("SELECT", result["sql_query"])
        self.assertIn("COUNT", result["sql_query"])
        self.assertIn("users", result["sql_query"].lower())

        # 검증
        validation = self.validator.validate(result["sql_query"])
        self.assertTrue(validation["valid"])

    def test_join_query(self):
        result = self.chain.generate_sql("사용자별 주문 건수는?")
        self.assertIn("JOIN", result["sql_query"])
        self.assertIn("GROUP BY", result["sql_query"])

    def test_date_filtering(self):
        result = self.chain.generate_sql("이번 주 주문 목록은?")
        self.assertIn("DATE_TRUNC", result["sql_query"] or result["sql_query"])
```

### 4.3 체크리스트

- [ ] Few-shot 예제 작성 (최소 10개)
- [ ] Text-to-SQL Chain 구현
- [ ] 테이블 자동 선별 기능
- [ ] 예제 검색 기능
- [ ] 쿼리 검증 시스템
- [ ] 문법 검증 구현
- [ ] 실행 계획 분석
- [ ] 단위 테스트 작성
- [ ] 통합 테스트 실행

### 4.4 산출물

- ✅ Text-to-SQL 엔진
- ✅ 쿼리 검증 시스템
- ✅ Few-shot 예제 DB

---

## 5. Phase 3: 추가 기능 (1주)

### 5.1 Data Discovery 구현 (2일)

```python
# src/chains/data_discovery_chain.py

class DataDiscoveryChain:
    def explain_table(self, table_name: str) -> str:
        """테이블 설명"""
        # 메타데이터 조회
        # 칼럼 정보 조회
        # 샘플 데이터 조회
        # LLM으로 자연어 설명 생성
        pass

    def explain_column(self, table_name: str, column_name: str) -> str:
        """칼럼 설명"""
        pass
```

### 5.2 Memory Management 구현 (2일)

```python
# src/memory/conversation_memory.py

import sqlite3
from datetime import datetime

class ConversationMemory:
    def __init__(self, db_path="memory.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                turn_number INTEGER,
                user_message TEXT,
                assistant_message TEXT,
                generated_query TEXT,
                timestamp TIMESTAMP
            )
        """)

    def save_conversation(self, session_id, turn, user_msg, ai_msg, query=None):
        self.conn.execute("""
            INSERT INTO conversations
            (session_id, turn_number, user_message, assistant_message, generated_query, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session_id, turn, user_msg, ai_msg, query, datetime.now()))
        self.conn.commit()

    def get_recent_conversations(self, session_id, limit=5):
        cursor = self.conn.execute("""
            SELECT user_message, assistant_message, generated_query
            FROM conversations
            WHERE session_id = ?
            ORDER BY turn_number DESC
            LIMIT ?
        """, (session_id, limit))
        return cursor.fetchall()
```

### 5.3 Knowledge Discovery 구현 (3일)

```python
# src/chains/knowledge_discovery_chain.py

from langchain.chains import RetrievalQA

class KnowledgeDiscoveryChain:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 5})
        )

    def answer_question(self, question: str) -> str:
        """지식 검색 및 답변"""
        response = self.qa_chain.invoke({"query": question})
        return response["result"]
```

---

## 6. Phase 4: UI 구축 (1주)

### 6.1 Streamlit UI 구현 (5일)

**파일**: `src/ui/app.py`

```python
import streamlit as st
from src.chains.text_to_sql_chain import TextToSQLChain
import uuid

# 세션 초기화
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# UI 레이아웃
st.title("🤖 AI 데이터 분석 어시스턴트")

# 사이드바
with st.sidebar:
    st.header("설정")
    api_key = st.text_input("OpenAI API Key", type="password")
    db_uri = st.text_input("Database URI")

    if st.button("연결 테스트"):
        try:
            chain = TextToSQLChain(db_uri)
            st.success("✅ 연결 성공!")
        except Exception as e:
            st.error(f"❌ 연결 실패: {str(e)}")

# 대화 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력
if prompt := st.chat_input("질문을 입력하세요..."):
    # 사용자 메시지 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            chain = TextToSQLChain(db_uri)
            result = chain.generate_sql(prompt)

            response = f"""
**생성된 쿼리**:
```sql
{result['sql_query']}
```

**선택된 테이블**: {', '.join(result['selected_tables'])}
"""

            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
```

### 6.2 체크리스트

- [ ] Streamlit 앱 기본 구조
- [ ] 대화 인터페이스 구현
- [ ] SQL 쿼리 표시
- [ ] 결과 테이블 표시
- [ ] 사이드바 설정 UI
- [ ] 세션 관리
- [ ] 피드백 버튼

---

## 7. Phase 5: 테스팅 (1주)

### 7.1 테스트 계획

| 테스트 유형 | 커버리지 | 도구 |
|------------|----------|------|
| 단위 테스트 | 80% | pytest |
| 통합 테스트 | 주요 기능 | pytest |
| E2E 테스트 | UI 플로우 | Streamlit test |
| 성능 테스트 | 응답 시간 | pytest-benchmark |

### 7.2 테스트 시나리오

```python
# tests/test_e2e.py

def test_end_to_end_text_to_sql():
    # 1. 질문 입력
    question = "지난달 신규 가입자 수는?"

    # 2. SQL 생성
    chain = TextToSQLChain(db_uri)
    result = chain.generate_sql(question)

    # 3. 검증
    validator = QueryValidator(db_connection)
    validation = validator.validate(result["sql_query"])
    assert validation["valid"]

    # 4. 실행
    cursor = db_connection.cursor()
    cursor.execute(result["sql_query"])
    rows = cursor.fetchall()
    assert len(rows) > 0
```

---

## 8. Phase 6: 배포 (선택)

### 8.1 Docker 배포

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DATABASE_URL=${DATABASE_URL}
    volumes:
      - ./chromadb_data:/app/chromadb_data

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=yourdb
      - POSTGRES_PASSWORD=yourpass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  postgres_data:
```

**실행**:
```bash
docker-compose up -d
```

---

## 9. 리스크 관리

### 9.1 기술 리스크

| 리스크 | 확률 | 영향 | 완화 방안 |
|--------|------|------|----------|
| LLM API 비용 초과 | 중 | 고 | 캐싱, 프롬프트 압축 |
| 쿼리 정확도 부족 | 고 | 고 | Few-shot 예제 확충, 검증 강화 |
| 응답 시간 초과 | 중 | 중 | BM25 사전 계산, 병렬 처리 |
| ChromaDB 성능 저하 | 저 | 중 | HNSW 파라미터 최적화 |

### 9.2 일정 리스크

**버퍼**: 각 Phase에 20% 여유 시간 포함

**우선순위**:
1. Text-to-SQL (핵심)
2. RAG 시스템 (핵심)
3. UI (중요)
4. 추가 기능 (선택)

---

## 10. 품질 관리

### 10.1 코드 품질

- **Linting**: flake8, black
- **Type Checking**: mypy
- **Test Coverage**: >80%

### 10.2 문서화

- **README.md**: 프로젝트 소개, 설치, 사용법
- **API 문서**: Docstring (Sphinx)
- **사용자 가이드**: Streamlit UI 사용법

### 10.3 성능 모니터링

- 응답 시간 추적
- LLM API 비용 추적
- 쿼리 정확도 추적

---

**문서 버전**: 1.0.0
**최종 수정**: 2025-01-13
**승인자**: -

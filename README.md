# AI Assistant - Claude Code + Hugging Face Embeddings

한국어 지원 AI 어시스턴트 시스템 (OpenAI → Anthropic Claude Code 마이그레이션)

## 주요 기능

- **Intent Classification**: 쿼리 유형 자동 분류 (Text-to-SQL, Knowledge Discovery, General Chat)
- **Text-to-SQL**: 한국어 자연어를 PostgreSQL 쿼리로 변환
- **RAG-based Knowledge Discovery**: ChromaDB 벡터 검색으로 문서 기반 질의응답
- **Multi-turn Conversations**: 세션 기반 대화 히스토리 관리
- **Token Usage Tracking**: API 사용량 추적 및 예산 모니터링

## 기술 스택

- **LLM**: Anthropic Claude 3.5 Sonnet (langchain-anthropic)
- **Embeddings**: Hugging Face sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2, 384 dimensions)
- **Vector Store**: ChromaDB (문서 임베딩 및 검색)
- **Database**: PostgreSQL (읽기 전용 분석), SQLite (대화 메모리)
- **Framework**: LangChain (체인 오케스트레이션)
- **Language**: Python 3.10+

## 프로젝트 구조

```
ai-tool/
├── config/                  # 설정 파일
│   ├── settings.py         # Pydantic 설정 관리
│   └── .env.example        # 환경 변수 예제
├── src/
│   ├── chains/             # LangChain 체인
│   │   ├── router.py       # Intent classification
│   │   ├── text_to_sql.py  # SQL 생성
│   │   ├── knowledge.py    # RAG 검색
│   │   └── multi_turn.py   # 대화 관리
│   ├── models/             # 데이터 모델
│   │   ├── llm_config.py   # LLM 설정
│   │   └── query_response.py # 요청/응답
│   ├── services/           # 핵심 서비스
│   │   ├── llm_client.py   # Claude API 클라이언트
│   │   ├── embedding.py    # Hugging Face 임베딩
│   │   └── memory.py       # SQLite 메모리
│   └── utils/              # 유틸리티
│       ├── prompts.py      # 프롬프트 템플릿
│       ├── logging.py      # 로깅
│       └── errors.py       # 커스텀 예외
├── scripts/                # 실행 스크립트
│   ├── test_claude_connection.py
│   └── init_vector_store.py
└── tests/                  # 테스트
    ├── fixtures/           # 테스트 데이터
    └── unit/              # 단위 테스트
```

## 설치 및 설정

### 1. 의존성 설치

```bash
# Python 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
# .env 파일 생성
cp config/.env.example .env

# .env 파일 수정
# === LLM Configuration ===
ANTHROPIC_API_KEY=sk-ant-YOUR-ACTUAL-API-KEY

# === Database Configuration ===
POSTGRES_HOST=your-db-host
POSTGRES_PORT=5432
POSTGRES_DB=your_database
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password

# === Vector Store Configuration ===
CHROMA_PERSIST_DIRECTORY=./data/chroma
CHROMA_COLLECTION_NAME=documents

# === Embedding Configuration ===
EMBEDDING_MODEL_NAME=paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_DEVICE=cpu  # 또는 cuda (GPU 사용 시)
```

### 3. Claude API 연결 테스트

```bash
python scripts/test_claude_connection.py
```

**예상 출력:**
```
============================================================
Claude API Connection Test
============================================================
✅ API Key found: sk-ant-***...
✅ LLM connection test passed

Testing Korean support...
Query: 안녕하세요! 오늘 날씨는 어떤가요?
Response: [Claude의 한국어 응답]
Token Usage: Input=20, Output=35, Total=55
```

### 4. 벡터 저장소 초기화

```bash
python scripts/init_vector_store.py
```

이 스크립트는 ChromaDB를 초기화하고 샘플 문서를 임베딩합니다.

## 사용 예제

### Intent Classification (Router Chain)

```python
from src.chains.router import RouterChain
from src.services.llm_client import LLMClient
from src.models.query_response import QueryRequest, QueryType

llm_client = LLMClient()
router = RouterChain(llm_client)

# Text-to-SQL 쿼리
query = QueryRequest(
    user_id="user123",
    query_text="지난달 신규 가입자 수는?"
)
query_type = router.classify(query)
print(query_type)  # QueryType.TEXT_TO_SQL

# Knowledge 쿼리
query = QueryRequest(
    user_id="user123",
    query_text="회원가입 절차가 어떻게 되나요?"
)
query_type = router.classify(query)
print(query_type)  # QueryType.KNOWLEDGE

# General Assistant 쿼리
query = QueryRequest(
    user_id="user123",
    query_text="안녕하세요"
)
query_type = router.classify(query)
print(query_type)  # QueryType.ASSISTANT
```

### Text-to-SQL Chain

```python
from src.chains.text_to_sql import TextToSQLChain
from src.services.llm_client import LLMClient
from src.models.query_response import QueryRequest

llm_client = LLMClient()
text_to_sql = TextToSQLChain(llm_client)

query = QueryRequest(
    user_id="user123",
    query_text="지난 7일간 일별 신규 가입자 수를 조회해주세요"
)

response = text_to_sql.generate_sql(query)
print(response.sql_query)
# SELECT DATE(created_at) as date, COUNT(*) as new_users
# FROM users
# WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
# GROUP BY DATE(created_at)
# ORDER BY date DESC;

print(f"Confidence: {response.confidence_score}")
print(f"Token Usage: {response.token_usage.total_tokens}")
```

### Knowledge Discovery Chain

```python
from src.chains.knowledge import KnowledgeChain
from src.services.llm_client import LLMClient
from src.services.embedding import HuggingFaceEmbedding
from src.models.query_response import QueryRequest

llm_client = LLMClient()
embedding_service = HuggingFaceEmbedding()
knowledge_chain = KnowledgeChain(llm_client, embedding_service)

query = QueryRequest(
    user_id="user123",
    query_text="회원가입할 때 이메일 인증이 필요한가요?"
)

response = knowledge_chain.search(query, top_k=3)
print(response.answer)
# 네, 회원가입 시 이메일 인증이 필요합니다.
# 절차는 다음과 같습니다:
# 1. 이메일 주소 입력
# 2. 비밀번호 설정
# 3. 이메일 인증 (이 단계에서 인증 메일 확인)
# ...

print(f"Source Documents: {len(response.source_documents)}")
for doc in response.source_documents:
    print(f"- {doc.title} (relevance: {doc.relevance_score:.2f})")
```

### Multi-turn Conversation

```python
from src.chains.multi_turn import MultiTurnChain
from src.services.llm_client import LLMClient
from src.services.memory import SQLiteConversationMemory
from src.models.query_response import QueryRequest

llm_client = LLMClient()
memory = SQLiteConversationMemory()
chat = MultiTurnChain(llm_client, memory)

session_id = "session123"

# 첫 번째 대화
query1 = QueryRequest(
    user_id="user123",
    session_id=session_id,
    query_text="안녕하세요!"
)
response1 = chat.chat(query1)
print(response1.answer)
# 안녕하세요! 무엇을 도와드릴까요?

# 두 번째 대화 (히스토리 참조)
query2 = QueryRequest(
    user_id="user123",
    session_id=session_id,
    query_text="주문 내역을 확인하고 싶어요"
)
response2 = chat.chat(query2)
print(response2.answer)
# 주문 내역 확인을 도와드리겠습니다.
# 어떤 기간의 주문 내역을 확인하시겠어요?

# 대화 히스토리 조회
history = memory.get_conversation_history(session_id, limit=10)
for turn in history:
    print(f"User: {turn['user_message']}")
    print(f"Assistant: {turn['assistant_message']}")
```

## 테스트 실행

```bash
# 전체 테스트 실행
pytest

# 특정 테스트 파일 실행
pytest tests/unit/test_llm_client.py

# Coverage 리포트
pytest --cov=src --cov-report=html
```

## 성능 및 비용

### Claude 3.5 Sonnet Pricing
- **Input**: $3 / 1M tokens
- **Output**: $15 / 1M tokens

### 예상 토큰 사용량
- **Intent Classification**: ~50 tokens/query
- **Text-to-SQL**: ~200-500 tokens/query
- **Knowledge Discovery**: ~500-1000 tokens/query (문서 길이에 따라)
- **Multi-turn Chat**: ~100-300 tokens/turn (히스토리에 따라)

### 임베딩 성능
- **Model**: paraphrase-multilingual-MiniLM-L12-v2
- **Dimensions**: 384
- **Speed**: ~1000 sentences/sec (CPU), ~10000 sentences/sec (GPU)
- **Cost**: Free (로컬 실행)

## 문제 해결

### API 인증 오류
```
AuthenticationError: Invalid Anthropic API key
```
**해결**: `.env` 파일의 `ANTHROPIC_API_KEY`가 올바른지 확인하세요.

### 데이터베이스 연결 오류
```
DatabaseConnectionError: Could not connect to PostgreSQL
```
**해결**: PostgreSQL 연결 정보 (호스트, 포트, 사용자, 비밀번호)를 확인하세요.

### 임베딩 모델 다운로드 실패
```
OSError: Can't load tokenizer for 'paraphrase-multilingual-MiniLM-L12-v2'
```
**해결**: 인터넷 연결을 확인하고 Hugging Face Hub에서 모델이 자동 다운로드될 때까지 기다립니다.

### ChromaDB 초기화 오류
```
ChromaDB collection not found
```
**해결**: `python scripts/init_vector_store.py`를 실행하여 벡터 저장소를 초기화하세요.

## 개발 로드맵

### Phase 1-3: MVP 완료 ✅
- Claude Code API 통합
- Intent classification
- Text-to-SQL, Knowledge, Multi-turn chains
- 한국어 지원
- 단위 테스트

### Phase 4: Embedding 품질 검증 (예정)
- Retrieval accuracy 측정
- 임베딩 모델 벤치마크
- 검색 성능 최적화

### Phase 5: 예산 모니터링 (예정)
- Token usage 대시보드
- 비용 알림 시스템
- 사용량 리포트

### Phase 6: 프로덕션 배포 (예정)
- API 서버 구축 (FastAPI)
- 로깅 및 모니터링
- 성능 최적화

## 라이선스

MIT License

## 연락처

프로젝트 관련 문의: [your-email@example.com]

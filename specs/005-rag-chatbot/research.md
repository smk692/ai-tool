# Research: Slack RAG Chatbot

**Feature**: 005-rag-chatbot
**Date**: 2025-12-10
**Status**: Complete

이 문서는 Slack RAG 챗봇 구현을 위한 기술 조사 결과를 정리합니다.

---

## 1. Slack Bolt Python 통합

### 결정: Socket Mode 사용

**Rationale**:
- 공용 HTTP 엔드포인트 불필요 (방화벽 뒤에서도 작동)
- 로컬 개발 및 테스트 용이
- WebSocket 기반 실시간 이벤트 처리
- 소규모 회사 환경에 적합

**대안 검토**:
- HTTP Mode: 공용 URL 필요, HTTPS 인증서 필요 → 인프라 복잡도 증가
- Flask/FastAPI 어댑터: 웹 서버 필요 → 단순성 원칙 위반

### 구현 패턴

```python
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

@app.event("app_mention")
def handle_mention(body, say):
    """멘션 이벤트 처리"""
    text = body["event"].get("text")
    thread_ts = body["event"].get("thread_ts") or body["event"].get("ts")
    say(f"답변: {text}", thread_ts=thread_ts)

if __name__ == "__main__":
    handler = SocketModeHandler(app, os.environ.get("SLACK_APP_TOKEN"))
    handler.start()
```

### 필요한 Slack 권한

**Bot Token Scopes (xoxb-)**:
- `app_mentions:read`: 멘션 이벤트 수신
- `chat:write`: 메시지 전송
- `im:history`: DM 기록 읽기
- `im:read`: DM 채널 정보
- `im:write`: DM 메시지 전송
- `reactions:read`: 리액션 이벤트 수신

**Event Subscriptions**:
- `app_mention`: 멘션 이벤트
- `message.im`: DM 메시지 이벤트

---

## 2. Claude API 통합

### 결정: Anthropic Python SDK + Streaming

**Rationale**:
- 공식 SDK로 안정성 보장
- 스트리밍으로 사용자 대기 시간 감소
- 비동기 지원으로 동시 요청 처리 효율
- 타입 힌팅 지원

**대안 검토**:
- OpenAI GPT-4o (Constitution 권장): 프로젝트 CLAUDE.md에서 Claude 명시 → Claude 선택
- 직접 HTTP 호출: SDK 대비 유지보수 부담 증가

### 구현 패턴

```python
from anthropic import AsyncAnthropic

client = AsyncAnthropic()

async def generate_response(context: str, question: str) -> str:
    """RAG 기반 답변 생성"""
    async with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": f"""다음 문서를 참고하여 질문에 답변하세요.

문서:
{context}

질문: {question}

답변:"""
            }
        ]
    ) as stream:
        chunks = []
        async for text in stream.text_stream:
            chunks.append(text)
        return "".join(chunks)
```

### 비용 최적화 전략

| 전략 | 절감 효과 | 구현 복잡도 |
|------|----------|------------|
| 프롬프트 압축 | 20-30% | 낮음 |
| 응답 캐싱 | 30-50% | 중간 |
| 모델 선택 (Haiku) | 50-70% | 낮음 |
| 배치 처리 | 10-20% | 중간 |

**선택**:
- 기본: claude-sonnet-4 (품질 우선)
- 비용 초과 시: claude-haiku로 폴백

---

## 3. Redis 대화 컨텍스트 저장

### 결정: Redis + TTL 기반 세션 관리

**Rationale**:
- 기존 인프라 활용 (docker-compose에 이미 포함)
- TTL 자동 만료로 메모리 관리
- 빠른 읽기/쓰기 (문서 검색 3초 SLA 충족)
- 간단한 키-값 구조

**대안 검토**:
- SQLite: 파일 기반, 동시성 제한 → Redis 대비 성능 저하
- PostgreSQL: 오버엔지니어링, 운영 복잡도 증가
- 메모리 저장: 재시작 시 데이터 손실

### 데이터 구조 설계

```python
# 대화 컨텍스트 키 패턴
CONVERSATION_KEY = "conversation:{thread_ts}"
CONVERSATION_TTL = 3600  # 1시간

# 저장 형식 (JSON)
{
    "thread_ts": "1234567890.123456",
    "channel_id": "C1234567890",
    "messages": [
        {"role": "user", "content": "질문1", "ts": "..."},
        {"role": "assistant", "content": "답변1", "ts": "..."}
    ],
    "created_at": "2025-12-10T10:00:00Z",
    "updated_at": "2025-12-10T10:05:00Z"
}
```

### 구현 패턴

```python
import redis
import json
from datetime import datetime

class ConversationStore:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.ttl = 3600  # 1시간
        self.max_messages = 5  # 최근 5개 메시지 유지

    def get_context(self, thread_ts: str) -> list[dict]:
        """대화 컨텍스트 조회"""
        key = f"conversation:{thread_ts}"
        data = self.redis.get(key)
        if data:
            return json.loads(data).get("messages", [])
        return []

    def add_message(self, thread_ts: str, role: str, content: str):
        """메시지 추가"""
        key = f"conversation:{thread_ts}"
        data = self.redis.get(key)

        if data:
            conversation = json.loads(data)
        else:
            conversation = {"thread_ts": thread_ts, "messages": []}

        conversation["messages"].append({
            "role": role,
            "content": content,
            "ts": datetime.utcnow().isoformat()
        })

        # 최근 N개만 유지
        conversation["messages"] = conversation["messages"][-self.max_messages:]
        conversation["updated_at"] = datetime.utcnow().isoformat()

        self.redis.set(key, json.dumps(conversation), ex=self.ttl)
```

---

## 4. 피드백 수집 및 저장

### 결정: Redis + JSON 파일 백업

**Rationale**:
- Redis로 빠른 수집
- JSON/CSV로 정기 내보내기 (분석용)
- 단순한 구조로 유지보수 용이

### 데이터 구조

```python
# 피드백 키 패턴
FEEDBACK_KEY = "feedback:{message_ts}"

# 저장 형식
{
    "message_ts": "1234567890.123456",
    "thread_ts": "1234567890.000000",
    "channel_id": "C1234567890",
    "user_id": "U1234567890",
    "question": "사용자 질문",
    "answer": "챗봇 답변",
    "rating": "positive",  # positive | negative
    "created_at": "2025-12-10T10:00:00Z"
}
```

### Slack 리액션 매핑

| 리액션 | 의미 | rating 값 |
|--------|------|----------|
| `:+1:` / `:thumbsup:` | 도움됨 | positive |
| `:-1:` / `:thumbsdown:` | 도움안됨 | negative |

---

## 5. 민감 정보 가드레일

### 결정: 정규표현식 기반 패턴 감지

**Rationale**:
- 간단하고 빠른 처리
- 외부 의존성 없음
- 커스터마이징 용이

### 감지 패턴

```python
import re

SENSITIVE_PATTERNS = {
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "phone_kr": r"01[0-9]-?\d{3,4}-?\d{4}",
    "ssn_kr": r"\d{6}-?[1-4]\d{6}",  # 주민등록번호
    "card_number": r"\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}",
}

def detect_sensitive_info(text: str) -> list[str]:
    """민감 정보 유형 감지"""
    detected = []
    for pattern_name, pattern in SENSITIVE_PATTERNS.items():
        if re.search(pattern, text):
            detected.append(pattern_name)
    return detected
```

### 대응 전략

1. **경고 모드**: 민감 정보 감지 시 경고 메시지 추가
2. **차단 모드**: 민감 정보 포함 시 응답 거부 (선택적)

---

## 6. 벡터 검색 통합

### 결정: shared 모듈 재사용

**Rationale**:
- 기존 rag-indexer와 동일한 임베딩/벡터 스토어 사용
- 코드 중복 방지
- 일관된 검색 품질

### 구현 패턴

```python
from shared.embedding import EmbeddingClient
from shared.vector_store import QdrantStore

class RAGService:
    def __init__(self):
        self.embedding = EmbeddingClient()
        self.vector_store = QdrantStore()

    async def search(self, query: str, top_k: int = 5) -> list[dict]:
        """유사 문서 검색"""
        query_vector = self.embedding.embed(query)
        results = self.vector_store.search(
            query_vector=query_vector,
            limit=top_k,
            score_threshold=0.7  # 유사도 임계값
        )
        return results
```

---

## 7. 에러 처리 및 복원력

### Slack API 오류

| 오류 | 원인 | 대응 |
|------|------|------|
| `rate_limited` | API 속도 제한 | 지수 백오프 재시도 |
| `channel_not_found` | 채널 접근 불가 | 사용자에게 권한 안내 |
| `invalid_auth` | 토큰 만료 | 관리자 알림 |

### Claude API 오류

| 오류 | 원인 | 대응 |
|------|------|------|
| `rate_limit_error` | API 속도 제한 | 큐잉 + 재시도 |
| `overloaded_error` | 서버 과부하 | 잠시 후 재시도 안내 |
| `invalid_api_key` | API 키 오류 | 관리자 알림 |

### 구현 패턴

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def call_claude_with_retry(prompt: str) -> str:
    """재시도 로직이 포함된 Claude API 호출"""
    return await generate_response(prompt)
```

---

## 8. 성능 최적화 전략

### 응답 시간 목표: 10초 이내

| 단계 | 목표 시간 | 최적화 방법 |
|------|----------|------------|
| 질문 임베딩 | 0.5초 | 로컬 모델 사용 |
| 벡터 검색 | 0.5초 | Qdrant 최적화 |
| 컨텍스트 조회 | 0.1초 | Redis 캐싱 |
| LLM 생성 | 8초 | 스트리밍 |
| Slack 응답 | 0.5초 | 비동기 처리 |

### 동시성 처리

```python
# SocketModeHandler 설정
handler = SocketModeHandler(
    app=app,
    app_token=os.environ.get("SLACK_APP_TOKEN"),
    concurrency=10  # 동시 처리 스레드 수
)
```

---

## 결론

모든 기술 선택이 완료되었습니다:

| 영역 | 선택 | 근거 |
|------|------|------|
| Slack 통합 | Socket Mode + Bolt Python | 단순성, 로컬 개발 용이 |
| LLM | Claude API (Streaming) | 프로젝트 요구사항, 품질 |
| 대화 저장 | Redis (TTL) | 기존 인프라, 성능 |
| 피드백 | Redis + JSON 백업 | 단순성 |
| 가드레일 | 정규표현식 | 외부 의존성 없음 |
| 벡터 검색 | shared 모듈 재사용 | 일관성 |

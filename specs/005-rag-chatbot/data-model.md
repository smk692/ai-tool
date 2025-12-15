# Data Model: Slack RAG Chatbot

**Feature**: 005-rag-chatbot
**Date**: 2025-12-10
**Status**: Draft

---

## 1. Query (질문 요청)

사용자가 Slack에서 보낸 질문을 나타냅니다.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | `str` | ✅ | 질문 텍스트 (멘션 태그 제거됨) |
| `user_id` | `str` | ✅ | Slack 사용자 ID (e.g., U1234567890) |
| `channel_id` | `str` | ✅ | Slack 채널 ID (e.g., C1234567890) |
| `thread_ts` | `str` | ✅ | 스레드 타임스탬프 (대화 식별자) |
| `message_ts` | `str` | ✅ | 메시지 타임스탬프 (고유 식별자) |
| `is_dm` | `bool` | ✅ | DM 여부 |
| `created_at` | `datetime` | ✅ | 질문 수신 시간 (UTC) |
| `files` | `list[SlackFileInfo]` | ❌ | 첨부 파일 목록 (이미지 등) |

### Validation Rules

- `text`: 1자 이상, 4000자 이하 (Slack 메시지 제한)
- `user_id`: `U`로 시작하는 11자 문자열
- `channel_id`: `C` 또는 `D`로 시작하는 11자 문자열
- `thread_ts`, `message_ts`: Slack 타임스탬프 형식 (`\d+\.\d+`)

### Pydantic Model

```python
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
import re

class SlackFileInfo(BaseModel):
    """Slack 첨부 파일 정보"""
    id: str
    name: str
    mimetype: str
    url_private: str
    size: int  # bytes

class Query(BaseModel):
    """사용자 질문 모델"""
    text: str = Field(..., min_length=1, max_length=4000)
    user_id: str = Field(..., pattern=r"^U[A-Z0-9]{10}$")
    channel_id: str = Field(..., pattern=r"^[CD][A-Z0-9]{10}$")
    thread_ts: str = Field(..., pattern=r"^\d+\.\d+$")
    message_ts: str = Field(..., pattern=r"^\d+\.\d+$")
    is_dm: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    files: list[SlackFileInfo] = Field(default_factory=list)

    @field_validator("text")
    @classmethod
    def strip_mention(cls, v: str) -> str:
        """멘션 태그 제거"""
        return re.sub(r"<@[A-Z0-9]+>", "", v).strip()
```

---

## 2. SearchResult (검색 결과)

벡터DB에서 검색된 문서 청크를 나타냅니다.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `chunk_id` | `str` | ✅ | 청크 고유 ID |
| `content` | `str` | ✅ | 청크 텍스트 내용 |
| `score` | `float` | ✅ | 유사도 점수 (0.0 ~ 1.0) |
| `source_type` | `str` | ✅ | 문서 소스 타입 (notion, swagger) |
| `source_id` | `str` | ✅ | 원본 문서 ID |
| `source_title` | `str` | ✅ | 원본 문서 제목 |
| `source_url` | `str | None` | ❌ | 원본 문서 URL (있는 경우) |
| `metadata` | `dict` | ❌ | 추가 메타데이터 |

### Validation Rules

- `score`: 0.0 이상 1.0 이하, 검색 시 0.7 이상만 반환
- `source_type`: `notion` | `swagger` 중 하나
- `content`: 비어있지 않음

### Pydantic Model

```python
from typing import Literal
from pydantic import BaseModel, Field

class SearchResult(BaseModel):
    """벡터DB 검색 결과 모델"""
    chunk_id: str
    content: str = Field(..., min_length=1)
    score: float = Field(..., ge=0.0, le=1.0)
    source_type: Literal["notion", "swagger"]
    source_id: str
    source_title: str
    source_url: str | None = None
    metadata: dict = Field(default_factory=dict)

    @property
    def is_relevant(self) -> bool:
        """유사도 임계값 충족 여부"""
        return self.score >= 0.7
```

---

## 3. Response (답변)

Claude LLM이 생성한 답변을 나타냅니다.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | `str` | ✅ | 답변 텍스트 |
| `sources` | `list[SourceReference]` | ✅ | 참조 문서 목록 |
| `model` | `str` | ✅ | 사용된 LLM 모델 |
| `tokens_used` | `int` | ✅ | 사용된 토큰 수 |
| `generation_time_ms` | `int` | ✅ | 생성 시간 (밀리초) |
| `created_at` | `datetime` | ✅ | 생성 시간 (UTC) |
| `is_fallback` | `bool` | ❌ | 폴백 응답 여부 |

### Nested: SourceReference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `title` | `str` | ✅ | 문서 제목 |
| `url` | `str | None` | ❌ | 문서 URL |
| `source_type` | `str` | ✅ | 소스 타입 |

### Validation Rules

- `text`: 1자 이상, 4000자 이하 (Slack 메시지 제한)
- `tokens_used`: 0 이상
- `generation_time_ms`: 0 이상

### Pydantic Model

```python
from datetime import datetime
from pydantic import BaseModel, Field

class SourceReference(BaseModel):
    """참조 문서 정보"""
    title: str
    url: str | None = None
    source_type: Literal["notion", "swagger"]

class Response(BaseModel):
    """LLM 생성 답변 모델"""
    text: str = Field(..., min_length=1, max_length=4000)
    sources: list[SourceReference] = Field(default_factory=list)
    model: str = "claude-sonnet-4-20250514"
    tokens_used: int = Field(..., ge=0)
    generation_time_ms: int = Field(..., ge=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_fallback: bool = False

    def format_for_slack(self) -> str:
        """Slack 메시지 형식으로 포맷팅"""
        message = self.text
        if self.sources:
            message += "\n\n📚 *참조 문서:*\n"
            for src in self.sources:
                if src.url:
                    message += f"• <{src.url}|{src.title}>\n"
                else:
                    message += f"• {src.title}\n"
        return message
```

---

## 4. Conversation (대화 컨텍스트)

스레드 내 대화 기록을 나타냅니다. Redis에 저장됩니다.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `thread_ts` | `str` | ✅ | 스레드 타임스탬프 (Primary Key) |
| `channel_id` | `str` | ✅ | 채널 ID |
| `messages` | `list[ConversationMessage]` | ✅ | 대화 메시지 목록 (최대 5개) |
| `created_at` | `datetime` | ✅ | 대화 시작 시간 |
| `updated_at` | `datetime` | ✅ | 마지막 업데이트 시간 |

### Nested: ConversationMessage

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `role` | `str` | ✅ | 역할 (user, assistant) |
| `content` | `str` | ✅ | 메시지 내용 |
| `ts` | `str` | ✅ | 메시지 타임스탬프 |

### Storage

- **Key Pattern**: `conversation:{thread_ts}`
- **TTL**: 3600초 (1시간)
- **Max Messages**: 5개 (FIFO)

### State Transitions

```
[Empty] → add_message(user) → [1 message]
[N messages] → add_message(user/assistant) → [N+1 messages] (max 5)
[5 messages] → add_message() → [5 messages] (oldest removed)
[Any] → TTL expired → [Empty/Deleted]
```

### Pydantic Model

```python
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field

class ConversationMessage(BaseModel):
    """대화 메시지"""
    role: Literal["user", "assistant"]
    content: str
    ts: str

class Conversation(BaseModel):
    """대화 컨텍스트 모델 (Redis 저장)"""
    thread_ts: str = Field(..., pattern=r"^\d+\.\d+$")
    channel_id: str = Field(..., pattern=r"^[CD][A-Z0-9]{10}$")
    messages: list[ConversationMessage] = Field(default_factory=list, max_length=5)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def add_message(self, role: Literal["user", "assistant"], content: str, ts: str) -> None:
        """메시지 추가 (최대 5개 유지)"""
        self.messages.append(ConversationMessage(role=role, content=content, ts=ts))
        if len(self.messages) > 5:
            self.messages = self.messages[-5:]
        self.updated_at = datetime.utcnow()

    def to_claude_messages(self) -> list[dict]:
        """Claude API 형식으로 변환"""
        return [{"role": m.role, "content": m.content} for m in self.messages]

    @classmethod
    def redis_key(cls, thread_ts: str) -> str:
        """Redis 키 생성"""
        return f"conversation:{thread_ts}"
```

---

## 5. Feedback (피드백)

사용자 피드백을 나타냅니다. Redis에 저장 후 JSON으로 백업됩니다.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message_ts` | `str` | ✅ | 답변 메시지 타임스탬프 (Primary Key) |
| `thread_ts` | `str` | ✅ | 스레드 타임스탬프 |
| `channel_id` | `str` | ✅ | 채널 ID |
| `user_id` | `str` | ✅ | 피드백 제공자 ID |
| `question` | `str` | ✅ | 원본 질문 |
| `answer` | `str` | ✅ | 챗봇 답변 |
| `rating` | `str` | ✅ | 평가 (positive, negative) |
| `reaction` | `str` | ✅ | Slack 리액션 이름 |
| `created_at` | `datetime` | ✅ | 피드백 시간 |

### Reaction Mapping

| Slack Reaction | Rating |
|----------------|--------|
| `:+1:`, `:thumbsup:` | positive |
| `:-1:`, `:thumbsdown:` | negative |

### Storage

- **Redis Key Pattern**: `feedback:{message_ts}`
- **Redis TTL**: 604800초 (7일)
- **Backup**: JSON 파일로 정기 내보내기

### Pydantic Model

```python
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field

class Feedback(BaseModel):
    """사용자 피드백 모델 (Redis + JSON 백업)"""
    message_ts: str = Field(..., pattern=r"^\d+\.\d+$")
    thread_ts: str = Field(..., pattern=r"^\d+\.\d+$")
    channel_id: str = Field(..., pattern=r"^[CD][A-Z0-9]{10}$")
    user_id: str = Field(..., pattern=r"^U[A-Z0-9]{10}$")
    question: str
    answer: str
    rating: Literal["positive", "negative"]
    reaction: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @classmethod
    def from_reaction(cls, reaction: str) -> Literal["positive", "negative"] | None:
        """리액션 이름을 rating으로 변환"""
        positive_reactions = {"+1", "thumbsup", "white_check_mark", "heavy_check_mark"}
        negative_reactions = {"-1", "thumbsdown", "x", "no_entry"}

        if reaction in positive_reactions:
            return "positive"
        elif reaction in negative_reactions:
            return "negative"
        return None

    @classmethod
    def redis_key(cls, message_ts: str) -> str:
        """Redis 키 생성"""
        return f"feedback:{message_ts}"
```

---

## 6. ImageContent (이미지 콘텐츠)

Claude Vision API에 전달할 이미지 데이터를 나타냅니다.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `media_type` | `str` | ✅ | 이미지 MIME 타입 (image/jpeg, image/png, image/gif, image/webp) |
| `data` | `str` | ✅ | Base64 인코딩된 이미지 데이터 |
| `filename` | `str` | ❌ | 원본 파일명 |
| `size_bytes` | `int` | ❌ | 원본 파일 크기 (bytes) |

### Validation Rules

- `media_type`: `image/jpeg` | `image/png` | `image/gif` | `image/webp` 중 하나
- `data`: 비어있지 않은 Base64 문자열
- `size_bytes`: 20MB (20,971,520 bytes) 이하

### Supported MIME Types

| MIME Type | Extension |
|-----------|-----------|
| `image/jpeg` | .jpg, .jpeg |
| `image/png` | .png |
| `image/gif` | .gif |
| `image/webp` | .webp |

### Pydantic Model

```python
from typing import Literal
from pydantic import BaseModel, Field

class ImageContent(BaseModel):
    """Claude Vision API용 이미지 콘텐츠 모델"""
    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
    data: str = Field(..., min_length=1)  # Base64 인코딩 데이터
    filename: str | None = None
    size_bytes: int | None = Field(None, le=20_971_520)  # 최대 20MB

    def to_claude_format(self) -> dict:
        """Claude API ImageBlock 형식으로 변환"""
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": self.media_type,
                "data": self.data,
            },
        }
```

---

## 7. SlackFileInfo (Slack 파일 정보)

Slack 메시지에 첨부된 파일 정보를 나타냅니다.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | `str` | ✅ | Slack 파일 고유 ID |
| `name` | `str` | ✅ | 파일명 |
| `mimetype` | `str` | ✅ | MIME 타입 |
| `url_private` | `str` | ✅ | 다운로드 URL (인증 필요) |
| `size` | `int` | ✅ | 파일 크기 (bytes) |

### Validation Rules

- `id`: 비어있지 않음
- `size`: 20MB 이하
- `mimetype`: 이미지 처리 시 지원 타입 확인

### Pydantic Model

```python
from pydantic import BaseModel, Field

class SlackFileInfo(BaseModel):
    """Slack 첨부 파일 정보"""
    id: str = Field(..., min_length=1)
    name: str
    mimetype: str
    url_private: str
    size: int = Field(..., le=20_971_520)  # 최대 20MB

    @property
    def is_image(self) -> bool:
        """이미지 파일 여부"""
        return self.mimetype.startswith("image/")

    @property
    def is_supported_image(self) -> bool:
        """지원되는 이미지 형식 여부"""
        supported = {"image/jpeg", "image/png", "image/gif", "image/webp"}
        return self.mimetype in supported
```

---

## Entity Relationships

```
Query (1) -----> (N) SearchResult
  |                      |
  |                      v
  +-----> (N) SlackFileInfo -----> ImageContent (변환)
  |                      |
  v                      v
Response (1) <---- uses for context (text + images)
  |
  v
Conversation (1) -----> (N) ConversationMessage
  |
  v
Feedback (1) <---- references Question + Answer
```

### Relationship Details

1. **Query → SearchResult**: 하나의 질문에 대해 여러 검색 결과 반환 (top_k=5)
2. **Query → SlackFileInfo**: 하나의 질문에 여러 첨부 파일 가능 (최대 5개 이미지)
3. **SlackFileInfo → ImageContent**: Slack 파일을 Claude Vision API 형식으로 변환
4. **Query + SearchResult + ImageContent → Response**: 질문, 검색 결과, 이미지를 컨텍스트로 답변 생성
5. **Conversation → ConversationMessage**: 스레드 내 최대 5개 메시지 유지
6. **Response → Feedback**: 답변에 대한 사용자 피드백 수집

---

## Redis Schema Summary

| Key Pattern | Value Type | TTL | Description |
|-------------|------------|-----|-------------|
| `conversation:{thread_ts}` | JSON (Conversation) | 1시간 | 대화 컨텍스트 |
| `feedback:{message_ts}` | JSON (Feedback) | 7일 | 사용자 피드백 |

---

## Validation Summary

| Entity | Key Validations |
|--------|-----------------|
| Query | 텍스트 길이, Slack ID 형식, 타임스탬프 형식, 첨부 파일 목록 |
| SearchResult | 유사도 점수 범위, 소스 타입 enum |
| Response | 텍스트 길이 (Slack 제한), 토큰 수 |
| Conversation | 메시지 최대 5개, TTL 관리 |
| Feedback | 리액션 매핑, rating enum |
| ImageContent | MIME 타입 enum, Base64 데이터 존재, 크기 제한 (20MB) |
| SlackFileInfo | 파일 ID 존재, 크기 제한 (20MB), 지원 이미지 형식 검증 |

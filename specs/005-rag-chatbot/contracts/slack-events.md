# Slack Events Contract

**Feature**: 005-rag-chatbot
**Date**: 2025-12-10
**Status**: Draft

이 문서는 Slack RAG 챗봇이 처리하는 이벤트 스키마를 정의합니다.

---

## 1. App Mention Event

사용자가 챗봇을 멘션할 때 발생하는 이벤트입니다.

### Event Type
`app_mention`

### Required Scopes
- `app_mentions:read`

### Event Payload

```json
{
  "type": "event_callback",
  "event": {
    "type": "app_mention",
    "user": "U1234567890",
    "text": "<@U0LAN0Z89> 회사 휴가 정책이 어떻게 되나요?",
    "ts": "1234567890.123456",
    "channel": "C1234567890",
    "thread_ts": "1234567890.000000",
    "event_ts": "1234567890.123456"
  }
}
```

### Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `event.type` | `string` | ✅ | 항상 `app_mention` |
| `event.user` | `string` | ✅ | 멘션한 사용자 ID |
| `event.text` | `string` | ✅ | 전체 메시지 텍스트 (멘션 포함) |
| `event.ts` | `string` | ✅ | 메시지 타임스탬프 |
| `event.channel` | `string` | ✅ | 채널 ID |
| `event.thread_ts` | `string` | ❌ | 스레드 타임스탬프 (스레드 내 멘션 시) |
| `event.event_ts` | `string` | ✅ | 이벤트 타임스탬프 |

### Handler Behavior

1. 멘션 태그 제거 (`<@U0LAN0Z89>` → 빈 문자열)
2. `thread_ts` 결정: 있으면 사용, 없으면 `ts` 사용
3. Query 객체 생성 및 RAG 파이프라인 실행
4. 동일 스레드에 답변 전송

### Example Handler

```python
@app.event("app_mention")
def handle_app_mention(body, say, logger):
    event = body["event"]
    text = re.sub(r"<@[A-Z0-9]+>", "", event["text"]).strip()
    thread_ts = event.get("thread_ts") or event["ts"]

    # RAG 파이프라인 실행
    response = rag_service.generate_response(text, thread_ts)

    # 스레드에 답변
    say(text=response.format_for_slack(), thread_ts=thread_ts)
```

---

## 2. Direct Message Event

사용자가 챗봇에게 DM을 보낼 때 발생하는 이벤트입니다.

### Event Type
`message.im`

### Required Scopes
- `im:history`
- `im:read`
- `im:write`

### Event Payload

```json
{
  "type": "event_callback",
  "event": {
    "type": "message",
    "channel_type": "im",
    "user": "U1234567890",
    "text": "API 문서는 어디서 볼 수 있나요?",
    "ts": "1234567890.123456",
    "channel": "D1234567890",
    "event_ts": "1234567890.123456"
  }
}
```

### Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `event.type` | `string` | ✅ | `message` |
| `event.channel_type` | `string` | ✅ | `im` (DM 식별) |
| `event.user` | `string` | ✅ | 메시지 보낸 사용자 ID |
| `event.text` | `string` | ✅ | 메시지 텍스트 |
| `event.ts` | `string` | ✅ | 메시지 타임스탬프 |
| `event.channel` | `string` | ✅ | DM 채널 ID (`D`로 시작) |
| `event.event_ts` | `string` | ✅ | 이벤트 타임스탬프 |

### Handler Behavior

1. 봇 자신의 메시지 무시 (`event.bot_id` 체크)
2. Query 객체 생성 (is_dm=True)
3. RAG 파이프라인 실행
4. DM 채널에 직접 답변

### Example Handler

```python
@app.event("message")
def handle_dm(body, say, logger):
    event = body["event"]

    # DM만 처리
    if event.get("channel_type") != "im":
        return

    # 봇 메시지 무시
    if event.get("bot_id"):
        return

    text = event["text"]
    thread_ts = event["ts"]  # DM은 각 메시지가 독립적

    # RAG 파이프라인 실행
    response = rag_service.generate_response(text, thread_ts, is_dm=True)

    # 답변
    say(text=response.format_for_slack())
```

---

## 3. Reaction Added Event

사용자가 메시지에 리액션을 추가할 때 발생하는 이벤트입니다.

### Event Type
`reaction_added`

### Required Scopes
- `reactions:read`

### Event Payload

```json
{
  "type": "event_callback",
  "event": {
    "type": "reaction_added",
    "user": "U1234567890",
    "reaction": "+1",
    "item": {
      "type": "message",
      "channel": "C1234567890",
      "ts": "1234567890.123456"
    },
    "item_user": "U0LAN0Z89",
    "event_ts": "1234567890.789012"
  }
}
```

### Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `event.type` | `string` | ✅ | `reaction_added` |
| `event.user` | `string` | ✅ | 리액션 추가한 사용자 ID |
| `event.reaction` | `string` | ✅ | 리액션 이름 (콜론 없이) |
| `event.item.type` | `string` | ✅ | 항상 `message` |
| `event.item.channel` | `string` | ✅ | 메시지가 있는 채널 ID |
| `event.item.ts` | `string` | ✅ | 리액션이 추가된 메시지 타임스탬프 |
| `event.item_user` | `string` | ✅ | 원본 메시지 작성자 ID |
| `event.event_ts` | `string` | ✅ | 이벤트 타임스탬프 |

### Supported Reactions

| Reaction | Aliases | Rating |
|----------|---------|--------|
| `+1` | `thumbsup`, `white_check_mark` | positive |
| `-1` | `thumbsdown`, `x`, `no_entry` | negative |

### Handler Behavior

1. 봇 자신의 메시지에 대한 리액션만 처리 (`item_user` == 봇 ID)
2. 지원되는 리액션인지 확인
3. Feedback 객체 생성
4. Redis에 저장

### Example Handler

```python
@app.event("reaction_added")
def handle_reaction(body, client, logger):
    event = body["event"]

    # 봇 메시지에 대한 리액션만 처리
    if event["item_user"] != BOT_USER_ID:
        return

    # 피드백 rating 결정
    rating = Feedback.from_reaction(event["reaction"])
    if rating is None:
        return  # 지원되지 않는 리액션

    # 원본 메시지 조회 (질문과 답변 가져오기)
    result = client.conversations_history(
        channel=event["item"]["channel"],
        latest=event["item"]["ts"],
        inclusive=True,
        limit=1
    )

    if not result["messages"]:
        return

    message = result["messages"][0]

    # 스레드 내 원본 질문 조회
    thread_ts = message.get("thread_ts", message["ts"])

    # Feedback 저장
    feedback = Feedback(
        message_ts=event["item"]["ts"],
        thread_ts=thread_ts,
        channel_id=event["item"]["channel"],
        user_id=event["user"],
        question="",  # 별도 조회 필요
        answer=message["text"],
        rating=rating,
        reaction=event["reaction"]
    )

    feedback_service.save(feedback)
```

---

## 4. Response Format

챗봇이 Slack에 전송하는 응답 형식입니다.

### Basic Response

```json
{
  "channel": "C1234567890",
  "thread_ts": "1234567890.000000",
  "text": "회사 휴가 정책은 다음과 같습니다...\n\n📚 *참조 문서:*\n• <https://notion.so/xxx|휴가 정책 가이드>"
}
```

### Response with Blocks (Optional)

```json
{
  "channel": "C1234567890",
  "thread_ts": "1234567890.000000",
  "text": "회사 휴가 정책은...",
  "blocks": [
    {
      "type": "section",
      "text": {
        "type": "mrkdwn",
        "text": "회사 휴가 정책은 다음과 같습니다..."
      }
    },
    {
      "type": "divider"
    },
    {
      "type": "context",
      "elements": [
        {
          "type": "mrkdwn",
          "text": "📚 *참조 문서:* <https://notion.so/xxx|휴가 정책 가이드>"
        }
      ]
    }
  ]
}
```

### Error Response

```json
{
  "channel": "C1234567890",
  "thread_ts": "1234567890.000000",
  "text": "⚠️ 죄송합니다, 관련 정보를 찾지 못했습니다.\n질문을 더 구체적으로 해주시거나, 담당자에게 문의해 주세요."
}
```

### Fallback Response (No Results)

```json
{
  "channel": "C1234567890",
  "thread_ts": "1234567890.000000",
  "text": "🤔 죄송합니다, 해당 질문에 대한 정보를 찾지 못했습니다.\n\n다음 방법을 시도해 보세요:\n• 질문을 다른 키워드로 다시 해주세요\n• 더 구체적인 내용을 포함해 주세요"
}
```

---

## 5. Error Handling

### Slack API Errors

| Error Code | Description | Action |
|------------|-------------|--------|
| `rate_limited` | API 속도 제한 | 지수 백오프 재시도 (최대 3회) |
| `channel_not_found` | 채널 접근 불가 | 사용자에게 권한 안내 |
| `invalid_auth` | 토큰 만료/오류 | 관리자 알림, 로그 기록 |
| `not_in_channel` | 봇이 채널에 없음 | 사용자에게 초대 요청 안내 |

### Retry Policy

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(SlackApiError)
)
def send_message(client, channel, text, thread_ts):
    return client.chat_postMessage(
        channel=channel,
        text=text,
        thread_ts=thread_ts
    )
```

---

## 6. Security Considerations

### Input Validation
- 모든 Slack ID 형식 검증
- 메시지 길이 제한 (4000자)
- 민감 정보 패턴 감지

### Rate Limiting
- Slack Tier 2: 20+ requests/minute
- 동시 요청 10개 이하 처리

### Logging
- 사용자 ID 해시화 (개인정보 보호)
- 민감 정보 마스킹
- 오류 로그에 스택 트레이스 포함

---

## 7. Testing

### Mock Event Examples

```python
# App Mention Event
MOCK_APP_MENTION = {
    "type": "event_callback",
    "event": {
        "type": "app_mention",
        "user": "U1234567890",
        "text": "<@U0LAN0Z89> 테스트 질문입니다",
        "ts": "1234567890.123456",
        "channel": "C1234567890",
        "event_ts": "1234567890.123456"
    }
}

# DM Event
MOCK_DM = {
    "type": "event_callback",
    "event": {
        "type": "message",
        "channel_type": "im",
        "user": "U1234567890",
        "text": "DM 테스트 질문입니다",
        "ts": "1234567890.123456",
        "channel": "D1234567890",
        "event_ts": "1234567890.123456"
    }
}

# Reaction Event
MOCK_REACTION = {
    "type": "event_callback",
    "event": {
        "type": "reaction_added",
        "user": "U1234567890",
        "reaction": "+1",
        "item": {
            "type": "message",
            "channel": "C1234567890",
            "ts": "1234567890.123456"
        },
        "item_user": "U0LAN0Z89",
        "event_ts": "1234567890.789012"
    }
}
```

### Test Scenarios

1. **멘션 이벤트 처리**: 멘션 태그 제거, Query 생성, 스레드 응답
2. **DM 이벤트 처리**: 봇 메시지 필터링, DM 응답
3. **리액션 이벤트 처리**: 피드백 rating 변환, Redis 저장
4. **에러 처리**: API 오류 시 재시도, 폴백 응답

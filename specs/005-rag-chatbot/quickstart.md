# Quickstart: Slack RAG Chatbot

**Feature**: 005-rag-chatbot
**Date**: 2025-12-10
**Estimated Setup Time**: 15-20분

---

## 사전 요구사항

### 필수 서비스
- ✅ Python 3.10+
- ✅ Docker & Docker Compose (Qdrant, Redis)
- ✅ Slack 워크스페이스 관리자 권한
- ✅ Anthropic API 키

### 선택적 요구사항
- rag-indexer 실행 완료 (벡터DB에 문서 인덱싱됨)

---

## Step 1: Slack App 생성

### 1.1 Slack App 생성
1. [Slack API](https://api.slack.com/apps) 접속
2. **Create New App** → **From scratch** 선택
3. App Name: `RAG Chatbot` (또는 원하는 이름)
4. Workspace: 대상 워크스페이스 선택

### 1.2 Socket Mode 활성화
1. **Settings** → **Socket Mode** → **Enable Socket Mode**
2. App-Level Token 생성:
   - Token Name: `socket-mode-token`
   - Scope: `connections:write`
3. 생성된 토큰 저장 (`xapp-...`)

### 1.3 Bot Token Scopes 설정
**OAuth & Permissions** → **Scopes** → **Bot Token Scopes** 추가:

| Scope | 용도 |
|-------|------|
| `app_mentions:read` | 멘션 이벤트 수신 |
| `chat:write` | 메시지 전송 |
| `im:history` | DM 기록 읽기 |
| `im:read` | DM 채널 정보 |
| `im:write` | DM 메시지 전송 |
| `reactions:read` | 리액션 이벤트 수신 |

### 1.4 Event Subscriptions 설정
**Event Subscriptions** → **Enable Events** 활성화

**Subscribe to bot events** 추가:
- `app_mention`
- `message.im`
- `reaction_added`

### 1.5 App 설치
**Install App** → **Install to Workspace**
- Bot User OAuth Token 저장 (`xoxb-...`)

---

## Step 2: 환경 설정

### 2.1 인프라 시작
```bash
cd /path/to/ai-tool
make infra-up
```

### 2.2 환경 변수 설정
```bash
cd rag-chatbot
cp .env.example .env
```

`.env` 파일 편집:
```bash
# Slack
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_APP_TOKEN=xapp-your-app-token
SLACK_SIGNING_SECRET=your-signing-secret

# Anthropic Claude API
ANTHROPIC_API_KEY=your-anthropic-api-key

# Vector DB (shared 모듈 사용)
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Redis (대화 컨텍스트, 피드백)
REDIS_HOST=localhost
REDIS_PORT=6379

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Optional
LOG_LEVEL=INFO
SIMILARITY_THRESHOLD=0.7
MAX_CONTEXT_MESSAGES=5
CONVERSATION_TTL=3600
```

### 2.3 의존성 설치
```bash
make install-chatbot
```

또는 수동:
```bash
cd rag-chatbot
pip install -e .
pip install -e ../shared
```

---

## Step 3: 챗봇 실행

### 3.1 개발 모드 실행
```bash
cd rag-chatbot
python -m src.main
```

예상 출력:
```
INFO: Slack Bolt app starting...
INFO: Connected to Qdrant at localhost:6333
INFO: Connected to Redis at localhost:6379
INFO: Socket Mode connection established
```

### 3.2 Docker 실행 (선택사항)
```bash
docker-compose -f infra/docker/docker-compose.yml up rag-chatbot
```

---

## Step 4: 동작 확인

### 4.1 기본 테스트
Slack에서 챗봇 테스트:
1. 채널에 봇 초대: `/invite @RAG Chatbot`
2. 멘션으로 질문: `@RAG Chatbot 회사 휴가 정책이 어떻게 되나요?`
3. DM으로 질문: 봇에게 직접 메시지

### 4.2 예상 응답

**정상 응답**:
```
회사 휴가 정책은 다음과 같습니다:
- 연차: 입사 1년 후 15일 부여
- 병가: 연간 3일
...

📚 참조 문서:
• 휴가 정책 가이드
```

**관련 문서 없음**:
```
🤔 죄송합니다, 해당 질문에 대한 정보를 찾지 못했습니다.

다음 방법을 시도해 보세요:
• 질문을 다른 키워드로 다시 해주세요
• 더 구체적인 내용을 포함해 주세요
```

### 4.3 피드백 테스트
1. 챗봇 응답에 👍 리액션 추가
2. Redis에서 피드백 확인:
```bash
redis-cli
> KEYS feedback:*
> GET feedback:1234567890.123456
```

---

## Step 5: 트러블슈팅

### 연결 오류

| 오류 | 원인 | 해결 |
|------|------|------|
| `invalid_auth` | 잘못된 Slack 토큰 | 토큰 재발급 및 확인 |
| `Connection refused (Qdrant)` | Qdrant 미실행 | `make infra-up` |
| `Connection refused (Redis)` | Redis 미실행 | `make infra-up` |
| `rate_limit_error` | Claude API 제한 | 잠시 후 재시도 |

### 로그 확인
```bash
# 챗봇 로그
LOG_LEVEL=DEBUG python -m src.main

# 인프라 로그
make infra-logs
```

### 상태 확인
```bash
# 인프라 상태
make infra-status

# Qdrant 컬렉션 확인
curl http://localhost:6333/collections

# Redis 연결 확인
redis-cli ping
```

---

## 다음 단계

### 개발 환경
- [ ] 테스트 실행: `make test-chatbot`
- [ ] 린트 확인: `make lint`

### 운영 환경
- [ ] Docker 이미지 빌드
- [ ] 환경 변수 보안 설정
- [ ] 로그 모니터링 설정
- [ ] 알림 설정 (오류 발생 시)

### 추가 기능
- [ ] rag-indexer로 문서 인덱싱
- [ ] 피드백 분석 리포트 설정
- [ ] 성능 모니터링 대시보드

---

## 빠른 명령어 요약

```bash
# 인프라 시작
make infra-up

# 챗봇 설치
make install-chatbot

# 챗봇 실행
cd rag-chatbot && python -m src.main

# 테스트
make test-chatbot

# 로그 확인
make infra-logs
```

# ai-tool Development Guidelines

RAG 시스템 - 문서 인덱싱 및 Slack 챗봇 서비스

Last updated: 2025-12-12

## Active Technologies

- Python 3.10+ (타입 힌팅 필수)
- Qdrant (Vector DB)
- Redis (Cache - 선택적)
- HuggingFace sentence-transformers (Embeddings)
- Anthropic Claude API (LLM)
- Slack Bolt (Bot Framework)

## Project Structure

```text
ai-tool/
├── rag-indexer/          # 문서 인덱싱 서비스 (004-rag-indexer) ✅ 구현 완료
│   ├── src/
│   ├── tests/
│   └── pyproject.toml
├── rag-chatbot/          # Slack RAG 챗봇 서비스 (005-rag-chatbot) ✅ 구현 완료
│   ├── src/
│   ├── tests/
│   └── pyproject.toml
├── shared/               # 공통 모듈 (임베딩, 벡터DB 클라이언트) ✅ 구현 완료
│   ├── shared/
│   ├── tests/
│   └── pyproject.toml
├── infra/docker/         # Docker 인프라
│   ├── docker-compose.yml
│   └── .env.example
├── specs/                # Feature 명세 (SpecKit)
│   ├── 004-rag-indexer/
│   └── 005-rag-chatbot/
├── .specify/             # SpecKit 설정 및 문서
│   ├── USAGE.md          # 워크플로우 가이드
│   ├── memory/
│   ├── scripts/
│   └── templates/
├── _legacy/              # 아카이브된 기존 코드 (참고용)
└── Makefile
```

## SpecKit Workflow

새 기능 개발 시 SpecKit 워크플로우 사용:
```
/speckit.specify → /speckit.clarify → /speckit.plan → /speckit.tasks → /speckit.implement
```

자세한 사용법: `.specify/USAGE.md` 참조

## Commands

```bash
# 초기 설정
make setup             # 전체 프로젝트 초기 설정 (가상환경 + 의존성 + 인프라)
make setup-indexer     # rag-indexer 전용 가상환경 설정
make setup-chatbot     # rag-chatbot 전용 가상환경 설정
make download-model    # 임베딩 모델 사전 다운로드

# Infrastructure
make infra-up          # Start Qdrant + Redis
make infra-down        # Stop infrastructure
make infra-logs        # View logs
make infra-status      # Check status
make infra-reset       # 인프라 초기화 (볼륨 포함 삭제)

# Installation
make install-shared    # Install shared module
make install-indexer   # Install rag-indexer
make install-chatbot   # Install rag-chatbot
make install-all       # Install all

# 실행
make run-chatbot       # Slack 챗봇 실행
make run-chatbot-bg    # Slack 챗봇 백그라운드 실행
make stop-chatbot      # Slack 챗봇 중지
make run-indexer       # Indexer CLI 도움말 표시

# Testing
make test-shared       # Test shared module
make test-indexer      # Test rag-indexer
make test-chatbot      # Test rag-chatbot
make test              # Run all tests
make test-cov          # 커버리지 포함 테스트

# 코드 품질
make lint              # Check code style
make lint-fix          # Auto-fix lint issues
make format            # 코드 포맷팅 (Ruff)
make check             # 린트 + 테스트 전체 검사

# 헬스체크 & 정리
make health            # 서비스 헬스체크
make clean-cache       # Python 캐시 파일 삭제
make clean-venv        # 가상환경 삭제
make clean-all         # 모든 생성 파일 삭제
```

## Code Style

- Python 3.10+: Follow PEP 8, use type hints
- Line length: 100 characters
- Linter: Ruff
- Formatter: Ruff

## Services Overview

### rag-indexer (004-rag-indexer) ✅
문서 등록 파이프라인 - Notion 데이터와 Swagger.json을 벡터DB에 등록
- Notion API 연동 (페이지 + 데이터베이스)
- Swagger/OpenAPI 파싱
- 텍스트 청킹 및 임베딩
- 스케줄러 자동 동기화
- CLI 수동 트리거

### rag-chatbot (005-rag-chatbot) ✅
Slack RAG 챗봇 - 벡터DB 검색 및 Claude LLM을 통한 Slack 응답
- Slack 이벤트 처리 (멘션, DM)
- **이미지 분석 지원** (Slack 이미지 → Claude Vision API)
- 벡터DB 유사 문서 검색
- Claude LLM 답변 생성 (Claude Agent SDK)
- MCP 서버 연동 (Jira, Notion, Slack, Grafana, Sentry, AWS, Swagger)
- MCP 체이닝 (응답 내 외부 링크 자동 추가 조회)
- 대화 컨텍스트 유지 (Redis)
- 출처 표시 및 피드백 수집
- 가드레일 (민감 정보 탐지)

#### 아키텍처 특징
- **MessageProcessor 패턴**: DM/멘션 핸들러 공통 로직 추상화 (Template Method)
- **ImageProcessor 서비스**: Slack 파일 다운로드 및 Base64 변환
- **모듈화된 서비스**: RAGService, ConversationService, FeedbackService, ImageProcessor

### shared ✅
공통 모듈 - rag-indexer와 rag-chatbot에서 공유
- `shared.embedding`: HuggingFace 임베딩 클라이언트
- `shared.vector_store`: Qdrant 벡터 스토어 클라이언트

## Environment Variables

```bash
# Vector DB
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Notion API
NOTION_API_KEY=your_notion_api_key_here

# Anthropic Claude API
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Slack
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_APP_TOKEN=xapp-your-app-token
SLACK_SIGNING_SECRET=your_signing_secret

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Image Processing (선택적)
IMAGE_PROCESSING_ENABLED=true
IMAGE_MAX_SIZE_MB=20
IMAGE_MAX_COUNT=5
IMAGE_DOWNLOAD_TIMEOUT=30
```

## Quick Start

### 1. 환경 변수 설정
```bash
cd infra/docker
cp .env.example .env
# .env 파일 편집하여 API 키 입력
```

### 2. 인프라 실행
```bash
make infra-up  # Qdrant + Redis 시작
```

### 3. rag-indexer 실행 (문서 인덱싱)
```bash
cd rag-indexer
python -m venv .venv && source .venv/bin/activate
pip install -e ../shared && pip install -e ".[dev]"

# Notion 문서 인덱싱
python -m src.cli index-notion --database-id <DB_ID>
```

### 4. rag-chatbot 실행 (Slack 봇)
```bash
cd rag-chatbot
python -m venv .venv && source .venv/bin/activate
pip install -e ../shared && pip install -e ".[dev]"

# 봇 실행
python -m src.main
```

## Recent Changes
- 2025-12-12: 이미지 분석 기능 추가 (Slack 이미지 → Claude Vision API)
- 2025-12-12: 핸들러 리팩토링 (MessageProcessor 패턴으로 DM/멘션 코드 통합)
- 2025-12-12: MCP 체이닝 기능 추가 (응답 내 외부 링크 자동 추가 조회)
- 2025-12-12: Makefile 개선 (setup, run-chatbot, health 등 명령어 추가)
- 2025-12-12: CLAUDE.md 문서 현행화
- 2025-12-10: 005-rag-chatbot 구현 완료 ✅
- 2025-12-10: SpecKit USAGE.md 문서 추가
- 2025-12-10: 프로젝트 구조 개편 완료

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->

## 중요
모든 주석과 Summary는 한글로!

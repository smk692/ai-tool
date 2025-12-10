# ai-tool Development Guidelines

RAG 시스템 - 문서 인덱싱 및 Slack 챗봇 서비스

Last updated: 2025-12-05

## Active Technologies
- Qdrant (Vector DB), Redis (캐싱 - 선택적) (004-rag-indexer)
- Python 3.10+ (타입 힌팅 필수) (004-rag-indexer)

- Python 3.10+
- Qdrant (Vector DB)
- Redis (Cache)
- HuggingFace sentence-transformers (Embeddings)
- Anthropic Claude API (LLM)
- Slack Bolt (Bot Framework)

## Project Structure

```text
ai-tool/
├── rag-indexer/          # 문서 인덱싱 서비스
│   ├── src/
│   ├── tests/
│   └── pyproject.toml
├── rag-chatbot/          # Slack RAG 챗봇 서비스
│   ├── src/
│   ├── tests/
│   └── pyproject.toml
├── shared/               # 공통 모듈 (임베딩, 벡터DB 클라이언트)
│   ├── src/
│   ├── tests/
│   └── pyproject.toml
├── infra/docker/         # Docker 인프라
│   ├── docker-compose.yml
│   └── .env.example
├── specs/                # Feature 명세
│   ├── 004-rag-indexer/
│   └── 005-rag-chatbot/
├── _legacy/              # 아카이브된 기존 코드 (참고용)
└── Makefile
```

## Commands

```bash
# Infrastructure
make infra-up          # Start Qdrant + Redis
make infra-down        # Stop infrastructure
make infra-logs        # View logs
make infra-status      # Check status

# Installation
make install-shared    # Install shared module
make install-indexer   # Install rag-indexer
make install-chatbot   # Install rag-chatbot
make install-all       # Install all

# Testing
make test-shared       # Test shared module
make test-indexer      # Test rag-indexer
make test-chatbot      # Test rag-chatbot
make test              # Run all tests

# Linting
make lint              # Check code style
make lint-fix          # Auto-fix lint issues
```

## Code Style

- Python 3.10+: Follow PEP 8, use type hints
- Line length: 100 characters
- Linter: Ruff
- Formatter: Ruff

## Services Overview

### rag-indexer (004-rag-indexer)
문서 등록 파이프라인 - Notion 데이터와 Swagger.json을 벡터DB에 등록
- Notion API 연동 (페이지 + 데이터베이스)
- Swagger/OpenAPI 파싱
- 텍스트 청킹 및 임베딩
- 스케줄러 자동 동기화
- CLI 수동 트리거

### rag-chatbot (005-rag-chatbot)
Slack RAG 챗봇 - 벡터DB 검색 및 Claude LLM을 통한 Slack 응답
- Slack 이벤트 처리 (멘션, DM)
- 벡터DB 유사 문서 검색
- Claude LLM 답변 생성
- 대화 컨텍스트 유지
- 출처 표시 및 피드백 수집

## Environment Variables

```bash
# Vector DB
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Notion API
NOTION_API_KEY=your_notion_api_key

# Anthropic Claude API
ANTHROPIC_API_KEY=your_anthropic_api_key

# Slack
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_APP_TOKEN=xapp-your-app-token
SLACK_SIGNING_SECRET=your_signing_secret

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## Recent Changes
- 004-rag-indexer: Added Python 3.10+ (타입 힌팅 필수)
- 004-rag-indexer: Added Python 3.10+

- 레거시 코드 아카이브: src/, tests/ → _legacy/

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->


모든 주석과 Summery는 한글로!
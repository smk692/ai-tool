# AI Tool - RAG 시스템

문서 인덱싱 및 Slack RAG 챗봇 서비스

## 개요

Notion 문서와 Swagger API 스펙을 벡터 데이터베이스에 색인화하고, Slack을 통해 RAG 기반 질의응답을 제공하는 시스템입니다.

## 기술 스택

| 분류 | 기술 |
|------|------|
| Language | Python 3.10+ |
| Vector DB | Qdrant |
| Cache | Redis |
| Embeddings | HuggingFace sentence-transformers |
| LLM | Anthropic Claude API |
| Bot Framework | Slack Bolt |

## 프로젝트 구조

```
ai-tool/
├── rag-indexer/          # 문서 인덱싱 서비스
│   ├── src/
│   │   ├── cli.py        # CLI 인터페이스
│   │   ├── config.py     # 설정 관리
│   │   ├── connectors/   # Notion, Swagger 커넥터
│   │   ├── models/       # 데이터 모델
│   │   ├── services/     # 핵심 서비스
│   │   └── scheduler/    # 자동 동기화 스케줄러
│   └── tests/
├── rag-chatbot/          # Slack RAG 챗봇 서비스
│   ├── src/
│   │   ├── main.py       # 봇 엔트리포인트
│   │   ├── config.py     # 설정 관리
│   │   ├── handlers/     # Slack 이벤트 핸들러
│   │   ├── services/     # 검색, 컨텍스트 서비스
│   │   ├── llm/          # Claude LLM 클라이언트
│   │   ├── guardrails/   # 민감 정보 탐지
│   │   └── models/       # 데이터 모델
│   └── tests/
├── shared/               # 공통 모듈
│   └── shared/
│       ├── embedding.py  # HuggingFace 임베딩 클라이언트
│       └── vector_store.py # Qdrant 벡터 스토어
├── infra/docker/         # Docker 인프라
│   ├── docker-compose.yml
│   └── .env.example
├── specs/                # Feature 명세 (SpecKit)
├── scripts/              # 유틸리티 스크립트
└── Makefile
```

## 빠른 시작

### 1. 환경 변수 설정

```bash
cd infra/docker
cp .env.example .env
# .env 파일에 API 키 설정
```

### 2. 전체 설정 (권장)

```bash
make setup
```

이 명령어는 다음을 자동 수행합니다:
- 환경 변수 파일 생성
- 모든 의존성 설치
- Docker 인프라 시작 (Qdrant, Redis)
- 헬스체크

### 3. 개별 서비스 설정

```bash
# rag-indexer 설정
make setup-indexer

# rag-chatbot 설정
make setup-chatbot
```

## 서비스 실행

### Slack 챗봇 실행

```bash
# 포그라운드 실행
make run-chatbot

# 백그라운드 실행
make run-chatbot-bg

# 중지
make stop-chatbot
```

### 문서 인덱싱

```bash
# CLI 도움말
make run-indexer

# Notion 문서 인덱싱
cd rag-indexer
python -m src.cli index-notion --database-id <DB_ID>

# Swagger 인덱싱
python -m src.cli index-swagger --url <SWAGGER_URL>

# 스케줄러 실행 (자동 동기화)
python -m src.cli scheduler
```

## 주요 명령어

```bash
# 인프라
make infra-up          # Qdrant + Redis 시작
make infra-down        # 인프라 중지
make infra-status      # 상태 확인
make health            # 헬스체크

# 테스트
make test              # 전체 테스트
make test-cov          # 커버리지 포함

# 코드 품질
make lint              # 린트 검사
make lint-fix          # 자동 수정
make format            # 코드 포맷팅

# 정리
make clean-cache       # 캐시 삭제
make clean-all         # 전체 정리
```

## 환경 변수

| 변수 | 설명 | 필수 |
|------|------|------|
| `QDRANT_HOST` | Qdrant 호스트 | ✅ |
| `QDRANT_PORT` | Qdrant 포트 (기본: 6333) | ✅ |
| `REDIS_HOST` | Redis 호스트 | ⬚ |
| `REDIS_PORT` | Redis 포트 (기본: 6379) | ⬚ |
| `NOTION_API_KEY` | Notion API 키 | ✅ (Notion 사용 시) |
| `ANTHROPIC_API_KEY` | Claude API 키 | ✅ (챗봇 사용 시) |
| `SLACK_BOT_TOKEN` | Slack Bot Token | ✅ (챗봇 사용 시) |
| `SLACK_APP_TOKEN` | Slack App Token | ✅ (챗봇 사용 시) |
| `EMBEDDING_MODEL` | 임베딩 모델 | ⬚ (기본값 있음) |
| `IMAGE_PROCESSING_ENABLED` | 이미지 처리 활성화 | ⬚ (기본: true) |
| `IMAGE_MAX_SIZE_MB` | 최대 이미지 크기 (MB) | ⬚ (기본: 20) |
| `IMAGE_MAX_COUNT` | 요청당 최대 이미지 수 | ⬚ (기본: 5) |

## 서비스 상세

### rag-indexer

문서 인덱싱 파이프라인:
- **Notion 커넥터**: 페이지 및 데이터베이스 동기화
- **Swagger 커넥터**: API 문서 파싱
- **청킹**: LangChain 기반 텍스트 분할
- **임베딩**: multilingual-e5-large-instruct (1024차원)
- **스케줄러**: cron 기반 자동 동기화

### rag-chatbot

Slack RAG 챗봇:
- **이벤트 처리**: 멘션, DM 응답 (MessageProcessor 패턴)
- **벡터 검색**: Qdrant 유사 문서 검색
- **LLM 응답**: Claude를 통한 답변 생성
- **이미지 분석**: Slack 이미지 → Claude Vision API 분석
- **컨텍스트 관리**: Redis 기반 대화 히스토리
- **가드레일**: 민감 정보 탐지 및 필터링
- **피드백**: 사용자 피드백 수집

### shared

공통 모듈:
- `shared.embedding`: HuggingFace 임베딩 클라이언트
- `shared.vector_store`: Qdrant 벡터 스토어 클라이언트

## 개발

### 테스트

```bash
# 단위 테스트
cd rag-indexer && pytest tests/unit -v
cd rag-chatbot && pytest tests/unit -v

# 통합 테스트
cd rag-indexer && pytest tests/integration -v
```

### 코드 스타일

- Python 3.10+
- PEP 8 준수
- 타입 힌팅 필수
- Ruff 린터/포매터

## SpecKit 워크플로우

새 기능 개발 시:
```
/speckit.specify → /speckit.clarify → /speckit.plan → /speckit.tasks → /speckit.implement
```

자세한 사용법: `.specify/USAGE.md` 참조

## License

MIT

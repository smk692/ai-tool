# Quick Start Guide

빠르게 시작하기 위한 최소 설정 가이드입니다.

## 1. 환경 설정 (5분)

### 필수 요구사항
- Python 3.10 이상
- pip (Python 패키지 관리자)
- Anthropic API Key (https://console.anthropic.com/)

### 설치

```bash
# 1. 저장소 클론 (또는 압축 해제)
cd /path/to/ai-tool

# 2. 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 패키지 설치
pip install -r requirements.txt
```

## 2. API Key 설정 (2분)

```bash
# .env 파일 생성
cp config/.env.example .env

# .env 파일 편집
# ANTHROPIC_API_KEY에 실제 API 키 입력
nano .env  # 또는 원하는 에디터 사용
```

**최소 필수 설정:**
```env
ANTHROPIC_API_KEY=sk-ant-YOUR-ACTUAL-API-KEY-HERE
```

나머지 설정은 기본값 사용 가능합니다.

## 3. 연결 테스트 (1분)

```bash
# Claude API 연결 확인
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
Query: 안녕하세요
Response: [Claude의 한국어 응답]
Token Usage: Input=20, Output=35, Total=55
```

## 4. 벡터 저장소 초기화 (2분)

```bash
# ChromaDB 초기화 및 샘플 문서 임베딩
python scripts/init_vector_store.py
```

**예상 출력:**
```
================================================================================
 ChromaDB Vector Store Initialization
================================================================================

🔄 Initializing Hugging Face embedding service...
✅ Embedding service initialized: paraphrase-multilingual-MiniLM-L12-v2 (384 dimensions)

📄 Preparing 5 documents for embedding...
🔄 Embedding and storing documents in ChromaDB...
✅ Successfully embedded 5 documents!

--------------------------------------------------------------------------------
 Testing Vector Search
--------------------------------------------------------------------------------

Test 1: 회원가입하는 방법을 알려주세요
  Found 2 relevant documents:
    1. 회원가입 가이드 (계정관리) - similarity: 0.85
    2. 비밀번호 재설정 방법 (계정관리) - similarity: 0.62
```

## 5. 예제 실행 (5분)

```bash
# 모든 체인 예제 실행
python scripts/example_usage.py
```

이 스크립트는 다음을 테스트합니다:
- Intent Classification (Router Chain)
- Text-to-SQL Generation
- Knowledge Discovery (RAG)
- Multi-turn Conversation

## 6. (선택) 임베딩 품질 검증

```bash
# 임베딩 모델 검증 (Phase 4)
python scripts/validate_embedding_quality.py
```

검색 정확도, 관련성 점수, 성능 벤치마크를 확인할 수 있습니다.

## 다음 단계

### 개발 환경 설정
- IDE에서 프로젝트 열기
- Linter 설정 (ruff, black)
- 테스트 실행: `pytest`

### 시스템 통합
1. PostgreSQL 연결 설정 (Text-to-SQL 사용 시)
2. 실제 문서로 벡터 저장소 구축
3. 대화 메모리 데이터베이스 연결

### API 서버 구축 (예정)
- FastAPI 서버 설정
- 엔드포인트 구현
- 인증 및 권한 관리

## 문제 해결

### API Key 오류
```
AuthenticationError: Invalid Anthropic API key
```
→ `.env` 파일의 API 키가 올바른지 확인하세요.

### 모듈 Import 오류
```
ModuleNotFoundError: No module named 'anthropic'
```
→ 가상환경이 활성화되어 있는지 확인하고 `pip install -r requirements.txt` 재실행

### 임베딩 모델 다운로드 실패
```
OSError: Can't load tokenizer for 'paraphrase-multilingual-MiniLM-L12-v2'
```
→ 인터넷 연결을 확인하고 Hugging Face Hub 접근 가능한지 확인

## 추가 리소스

- **상세 문서**: [README.md](README.md)
- **API 문서**: Anthropic Claude - https://docs.anthropic.com/
- **임베딩 모델**: sentence-transformers - https://www.sbert.net/

## 지원

문제가 있으시면:
1. [README.md](README.md)의 "문제 해결" 섹션 확인
2. GitHub Issues 제출
3. 로그 파일 확인 (`logs/` 디렉토리)

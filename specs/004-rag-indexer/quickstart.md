# 빠른 시작 가이드: RAG Document Indexer

**소요 시간**: 약 15분
**목표**: Notion 문서를 벡터DB에 등록하고 검색 가능하게 만들기

---

## 사전 요구사항

- Python 3.10+
- Docker & Docker Compose
- Notion Integration (API 키)

---

## 1단계: 인프라 시작 (2분)

Qdrant 벡터 데이터베이스와 Redis를 시작합니다.

```bash
# 프로젝트 루트에서
make infra-up
```

**확인**:
```bash
# Qdrant 대시보드 접속
open http://localhost:6333/dashboard

# 상태 확인
make infra-status
```

---

## 2단계: 패키지 설치 (3분)

```bash
# 공용 모듈 설치
make install-shared

# rag-indexer 설치
make install-indexer
```

**확인**:
```bash
rag-indexer --version
# 출력: rag-indexer 0.1.0
```

---

## 3단계: Notion Integration 설정 (5분)

### 3.1 Notion Integration 생성

1. [Notion Integrations](https://www.notion.so/my-integrations) 페이지 접속
2. "새 통합 만들기" 클릭
3. 이름 입력: `RAG Indexer`
4. 기능 선택:
   - ✅ 콘텐츠 읽기
   - ✅ 댓글 읽기 (선택)
   - ❌ 콘텐츠 삽입 (불필요)
5. "저장" 클릭 후 **Internal Integration Token** 복사

### 3.2 페이지에 Integration 연결

1. 인덱싱할 Notion 페이지 열기
2. 우측 상단 `...` 메뉴 → "연결" → "RAG Indexer" 선택
3. 페이지 URL에서 **Page ID** 확인:
   ```
   https://www.notion.so/workspace/My-Page-abc123def456
                                        ↑ 이 부분이 Page ID
   ```

### 3.3 환경 변수 설정

```bash
# .env 파일 생성 또는 환경 변수 설정
export NOTION_API_KEY="secret_your_integration_token"
export QDRANT_HOST="localhost"
export QDRANT_PORT="6333"
```

또는 `~/.rag-indexer/config.yaml` 생성:
```yaml
notion:
  api_key: "secret_your_integration_token"

qdrant:
  host: localhost
  port: 6333
```

---

## 4단계: 첫 동기화 실행 (5분)

### 4.1 소스 등록

```bash
# Notion 페이지 소스 추가
rag-indexer source add \
  --name "내 문서" \
  --type notion \
  --page-id abc123def456
```

**예상 출력**:
```
✅ 소스 등록 완료
   ID: src_7f8a9b0c
   이름: 내 문서
   타입: notion
   페이지: 1개
```

### 4.2 동기화 실행

```bash
# 상세 로그와 함께 동기화
rag-indexer sync --all --verbose
```

**예상 출력**:
```
🔄 동기화 시작: 내 문서 (notion)
   ├─ 페이지 조회 중... 5개 발견
   ├─ 콘텐츠 추출 중...
   │   ├─ My Page (abc123) - 2,450자
   │   ├─ Sub Page 1 (def456) - 1,230자
   │   └─ ...
   ├─ 청킹 중... 18개 청크 생성
   ├─ 임베딩 생성 중... [████████████████] 100%
   └─ 벡터DB 저장 완료

✅ 동기화 완료
   처리: 5개 | 생성: 5개 | 청크: 18개
   소요 시간: 45초
```

---

## 5단계: 결과 확인

### CLI로 확인

```bash
rag-indexer status
```

**예상 출력**:
```
📊 RAG Indexer 상태

소스 현황:
┌──────────┬────────┬──────────┬─────────┬──────────────────────┐
│ 소스     │ 타입   │ 문서 수  │ 청크 수 │ 마지막 동기화        │
├──────────┼────────┼──────────┼─────────┼──────────────────────┤
│ 내 문서  │ notion │ 5        │ 18      │ 2025-12-05 14:30:00  │
└──────────┴────────┴──────────┴─────────┴──────────────────────┘

벡터DB:
  - 컬렉션: rag_documents
  - 포인트 수: 18
  - 상태: healthy ✅
```

### Qdrant 대시보드로 확인

1. http://localhost:6333/dashboard 접속
2. `rag_documents` 컬렉션 클릭
3. 저장된 포인트 확인

---

## 6단계: 자동 동기화 설정 (선택)

매일 오전 6시 자동 동기화를 설정합니다.

```bash
# 스케줄러 시작
rag-indexer scheduler start

# 상태 확인
rag-indexer scheduler status
```

**예상 출력**:
```
🕐 스케줄러 상태

상태: running ✅
스케줄: 0 6 * * * (매일 06:00)
다음 실행: 2025-12-06 06:00:00
```

---

## 문제 해결

### Notion API 오류

```
❌ NotionAPIError: Invalid API token
```
→ `NOTION_API_KEY` 환경 변수 확인

```
❌ NotionAPIError: Object not found
```
→ 페이지에 Integration 연결 여부 확인

### Qdrant 연결 오류

```
❌ ConnectionError: Cannot connect to Qdrant
```
→ `make infra-status`로 Docker 컨테이너 상태 확인

### 임베딩 오류

```
❌ MemoryError: Unable to allocate tensor
```
→ 배치 크기 줄이기 (`config.yaml`에서 `embedding.batch_size: 16`)

---

## 다음 단계

1. **Swagger 문서 추가**:
   ```bash
   rag-indexer source add \
     --name "API 문서" \
     --type swagger \
     --url "https://api.example.com/swagger.json"
   ```

2. **추가 Notion 페이지/데이터베이스 등록**:
   ```bash
   rag-indexer source add \
     --name "팀 위키" \
     --type notion \
     --database-id your_database_id
   ```

3. **RAG Chatbot 연동**:
   - `005-rag-chatbot` 스펙 참조
   - Slack 챗봇에서 인덱싱된 문서 검색

---

## 명령어 요약

| 작업 | 명령어 |
|------|--------|
| 소스 추가 | `rag-indexer source add --name "..." --type notion --page-id ...` |
| 소스 목록 | `rag-indexer source list` |
| 전체 동기화 | `rag-indexer sync --all` |
| 상태 확인 | `rag-indexer status` |
| 스케줄러 시작 | `rag-indexer scheduler start` |
| 스케줄러 중지 | `rag-indexer scheduler stop` |

자세한 CLI 옵션은 `rag-indexer --help` 또는 [CLI 명령어 스펙](./contracts/cli-commands.md)을 참조하세요.

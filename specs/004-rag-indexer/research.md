# Phase 0 리서치: RAG Document Indexer

**작성일**: 2025-12-05
**목적**: plan.md의 NEEDS CLARIFICATION 항목 해결

## 1. 스케줄러 라이브러리 선택

### 후보군 비교

| 라이브러리 | 장점 | 단점 | 적합성 |
|------------|------|------|--------|
| **APScheduler** | 경량, 내장 cron 지원, async 지원, 단일 프로세스 | 분산 환경 미지원 | ✅ 최적 |
| Celery | 분산 태스크, 확장성 | 복잡한 설정, Redis/RabbitMQ 필수, 과도한 규모 | ❌ 과잉 |
| 시스템 cron | 가장 단순 | Python 통합 어려움, 로깅 분리 | ❌ 부적합 |

### 결정: **APScheduler 3.10+**

**근거**:
- 프로젝트 규모 (100-500개 Notion 페이지)에 적합한 경량 솔루션
- Python 네이티브로 통합 용이
- `AsyncIOScheduler` + `CronTrigger`로 비동기 작업 지원
- Constitution의 "단순성 유지" 원칙 준수

**구현 예시**:
```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

scheduler = AsyncIOScheduler()
scheduler.add_job(
    sync_all_sources,
    CronTrigger(hour=6, minute=0),  # 매일 오전 6시
    id="daily_sync",
    replace_existing=True
)
scheduler.start()
```

---

## 2. 청킹 전략

### 후보군 비교

| 방식 | 장점 | 단점 | 적합성 |
|------|------|------|--------|
| **RecursiveCharacterTextSplitter** | 의미 단위 분할, 구조 보존 | 외부 의존성 | ✅ 최적 |
| 고정 크기 분할 | 구현 단순 | 문장 중간 절단 | ❌ 품질 저하 |
| 문장 단위 분할 | 의미 보존 | 청크 크기 불균일 | △ 차선책 |

### 결정: **langchain-text-splitters 사용**

**청킹 파라미터**:
- `chunk_size`: 1000 (한국어 기준 약 500자, 영어 기준 약 200단어)
- `chunk_overlap`: 200 (문맥 연속성 보장)
- `separators`: `["\n\n", "\n", ". ", " ", ""]` (한국어 적합 분리자)

**근거**:
- Notion 문서는 마크다운 기반으로 구조화되어 있어 `RecursiveCharacterTextSplitter`가 적합
- 오버랩 200자로 청크 간 문맥 손실 방지
- 임베딩 모델(384~768 차원)의 최적 입력 길이와 부합

**구현 예시**:
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)
chunks = splitter.split_text(document_text)
```

---

## 3. 임베딩 모델 선택

### 후보군 비교

| 모델 | 차원 | 한국어 지원 | 영어 지원 | 속도 | 품질 |
|------|------|-------------|-----------|------|------|
| all-MiniLM-L6-v2 | 384 | ❌ 미흡 | ✅ 최적 | 빠름 | 영어 전용 |
| paraphrase-multilingual-MiniLM-L12-v2 | 384 | △ 양호 | ✅ 양호 | 중간 | 다국어 균형 |
| **jhgan/ko-sbert-nli** | 768 | ✅ 최적 | △ 제한적 | 중간 | 한국어 특화 |
| snunlp/KR-SBERT-V40K-klueNLI-augSTS | 768 | ✅ 우수 | △ 제한적 | 중간 | KLUE 기반 |
| intfloat/multilingual-e5-base | 768 | ✅ 우수 | ✅ 우수 | 느림 | 다국어 최신 |

### 결정: **intfloat/multilingual-e5-large-instruct**

**근거**:
- 2024-2025년 기준 최신 다국어 임베딩 모델
- 한국어와 영어 모두 우수한 성능 (MTEB 벤치마크 상위권)
- 1024차원으로 더 풍부한 표현력 제공
- instruction-following 기능으로 검색 품질 향상
- sentence-transformers와 완벽 호환

**모델 특징**:
- 차원: 1024
- 학습 데이터: 다국어 대규모 코퍼스
- 특이점: 쿼리/문서에 instruction prefix 추가 시 성능 향상

**구현 예시**:
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")

# 문서 임베딩 (인덱싱 시)
doc_embeddings = model.encode(
    ["passage: " + chunk for chunk in chunks],
    show_progress_bar=True
)

# 쿼리 임베딩 (검색 시)
query_embedding = model.encode("query: " + user_query)
```

---

## 4. 변경 감지 방식

### 후보군 비교

| 방식 | 장점 | 단점 | 적합성 |
|------|------|------|--------|
| **content hash (SHA256)** | 정확한 변경 감지 | 전체 콘텐츠 읽기 필요 | ✅ 최적 |
| last_modified timestamp | API 호출 최소화 | Notion에서 부정확할 수 있음 | △ 보조 |
| 하이브리드 | 효율 + 정확성 | 구현 복잡 | △ 추후 고려 |

### 결정: **content hash (SHA256) 기반**

**근거**:
- Notion의 `last_edited_time`은 블록 수정 시 항상 갱신되지 않을 수 있음
- 콘텐츠 해시로 실제 변경 여부를 정확히 판단
- Constitution "정확도 우선" 원칙 준수

**최적화 전략**:
1. 1차 필터: `last_edited_time` 비교로 명백히 변경 없는 문서 스킵
2. 2차 검증: timestamp가 변경된 문서만 해시 비교
3. 해시 불일치 시에만 재인덱싱

**구현 예시**:
```python
import hashlib

def compute_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

def should_reindex(document: Document, new_content: str) -> bool:
    new_hash = compute_content_hash(new_content)
    return document.content_hash != new_hash
```

---

## 5. Notion 블록 타입 처리

### 지원 블록 타입 결정

| 우선순위 | 블록 타입 | 처리 방식 | 비고 |
|----------|-----------|-----------|------|
| P1 | `paragraph` | 텍스트 추출 | 핵심 콘텐츠 |
| P1 | `heading_1/2/3` | 텍스트 + 계층 정보 | 구조 메타데이터 |
| P1 | `bulleted_list_item` | 텍스트 + 들여쓰기 | 목록 처리 |
| P1 | `numbered_list_item` | 텍스트 + 순서 | 목록 처리 |
| P2 | `code` | 텍스트 + 언어 정보 | 기술 문서 |
| P2 | `quote` | 텍스트 + 인용 표시 | 강조 처리 |
| P2 | `callout` | 텍스트 + 이모지 | 중요 정보 |
| P2 | `toggle` | 텍스트 + 하위 블록 | 재귀 처리 |
| P3 | `table` | 행/열 텍스트화 | 테이블 평탄화 |
| P3 | `to_do` | 텍스트 + 체크 상태 | 작업 목록 |
| Skip | `image`, `video`, `file` | 메타데이터만 | 텍스트 없음 |
| Skip | `embed`, `bookmark` | URL만 저장 | 외부 링크 |

### 결정: **P1 + P2 블록 우선 지원**

**근거**:
- MVP에서는 텍스트 기반 블록 집중 (P1, P2)
- 이미지/비디오는 텍스트 검색 대상 아님
- 테이블은 복잡도 높아 후순위로

**구현 전략**:
```python
SUPPORTED_BLOCK_TYPES = {
    # P1: 핵심 텍스트 블록
    "paragraph": extract_rich_text,
    "heading_1": extract_rich_text,
    "heading_2": extract_rich_text,
    "heading_3": extract_rich_text,
    "bulleted_list_item": extract_rich_text,
    "numbered_list_item": extract_rich_text,

    # P2: 보조 텍스트 블록
    "code": extract_code_block,
    "quote": extract_rich_text,
    "callout": extract_callout,
    "toggle": extract_toggle_recursive,
}

def extract_block_content(block: dict) -> str:
    block_type = block["type"]
    if block_type in SUPPORTED_BLOCK_TYPES:
        return SUPPORTED_BLOCK_TYPES[block_type](block)
    return ""  # 미지원 블록은 빈 문자열
```

---

## 추가 리서치 결과

### Notion API Rate Limit 대응

**제한사항**:
- 평균 3 요청/초 권장
- 버스트 제한 존재

**대응 전략**:
```python
import asyncio
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(
    wait=wait_exponential(multiplier=1, min=1, max=60),
    stop=stop_after_attempt(5)
)
async def fetch_with_retry(client, page_id: str):
    await asyncio.sleep(0.35)  # Rate limit 준수
    return await client.pages.retrieve(page_id)
```

### Qdrant 컬렉션 설계

**결정사항**:
- 컬렉션 이름: `rag_documents`
- 벡터 차원: 1024 (intfloat/multilingual-e5-large-instruct 기준)
- 거리 메트릭: Cosine similarity
- 페이로드: `source_type`, `source_id`, `document_id`, `chunk_index`, `title`, `url`

```python
from qdrant_client.models import Distance, VectorParams

client.create_collection(
    collection_name="rag_documents",
    vectors_config=VectorParams(
        size=1024,
        distance=Distance.COSINE
    )
)
```

---

## 결론 요약

| 항목 | 결정 | 이유 |
|------|------|------|
| 스케줄러 | APScheduler | 경량, Python 네이티브, 프로젝트 규모 적합 |
| 청킹 | RecursiveCharacterTextSplitter (1000/200) | 의미 단위 분할, 구조 보존 |
| 임베딩 모델 | intfloat/multilingual-e5-large-instruct | 다국어 최신 모델, 1024차원, MTEB 상위권 |
| 변경 감지 | SHA256 content hash | 정확한 변경 감지, timestamp 보조 |
| Notion 블록 | P1+P2 텍스트 블록 우선 | MVP 범위, 텍스트 검색 집중 |

**Phase 1 진행 준비 완료** ✅

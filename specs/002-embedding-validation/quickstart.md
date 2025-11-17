# Quickstart Guide: Hugging Face 임베딩 통합

**Feature**: 002-embedding-validation
**Version**: 1.0.0
**Last Updated**: 2025-01-17

---

## 개요

이 가이드는 Hugging Face sentence-transformers 기반 임베딩 서비스를 빠르게 시작하는 방법을 안내합니다.

**주요 기능**:
- 🌐 다국어 지원 (한국어, 영어 등 50+ 언어)
- ⚡ 빠른 임베딩 생성 (배치 100개 ≤2초)
- 🎯 높은 검색 정확도 (Top-5 ≥90%)
- 💾 ChromaDB 벡터 스토어 통합

---

## 1. 설치 및 설정

### 1.1 의존성 설치

```bash
# 프로젝트 디렉토리로 이동
cd ai-tool

# 필수 패키지 설치
pip install sentence-transformers>=2.2.0 chromadb>=0.4.0
```

### 1.2 환경 변수 설정

`.env` 파일에 다음 설정을 추가하세요:

```bash
# Embedding Model Configuration
EMBEDDING_MODEL_NAME=paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_DEVICE=cpu  # cpu | cuda | mps
EMBEDDING_BATCH_SIZE=100
EMBEDDING_MAX_SEQUENCE_LENGTH=512

# Vector Store Configuration
CHROMA_PERSIST_DIRECTORY=./data/chroma
CHROMA_COLLECTION_NAME=documents
CHROMA_DISTANCE_FUNCTION=cosine
```

### 1.3 모델 다운로드 (선택사항)

첫 실행 시 모델이 자동으로 다운로드되지만, 사전 다운로드도 가능합니다:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"
```

**모델 크기**: ~470MB
**다운로드 위치**: `~/.cache/torch/sentence_transformers/`

---

## 2. 임베딩 서비스 사용

### 2.1 기본 사용법

```python
from src.services.embeddings import HuggingFaceEmbedding
from src.models.embedding import EmbeddingConfiguration

# 1. 설정 초기화
config = EmbeddingConfiguration()

# 2. 임베딩 서비스 생성
embedding_service = HuggingFaceEmbedding(config)

# 3. 단일 텍스트 임베딩
vector = embedding_service.embed_text("PostgreSQL 트랜잭션이란 무엇인가요?")
print(f"임베딩 차원: {len(vector)}")  # 384
print(f"벡터 샘플: {vector[:5]}")

# 4. 배치 임베딩
texts = [
    "데이터베이스 인덱스 종류",
    "SQL 쿼리 최적화 방법",
    "NoSQL과 관계형 데이터베이스 차이"
]
vectors = embedding_service.embed_texts(texts)
print(f"생성된 임베딩 수: {len(vectors)}")  # 3
```

### 2.2 GPU 사용 (옵션)

```python
from src.models.embedding import EmbeddingConfiguration, DeviceType

# CUDA GPU 사용
config_gpu = EmbeddingConfiguration(
    device=DeviceType.CUDA,
    batch_size=200  # GPU에서 더 큰 배치 크기
)
embedding_service = HuggingFaceEmbedding(config_gpu)

# Apple Silicon MPS 사용
config_mps = EmbeddingConfiguration(
    device=DeviceType.MPS,
    batch_size=150
)
embedding_service = HuggingFaceEmbedding(config_mps)
```

---

## 3. 문서 인덱싱

### 3.1 프로그래밍 방식

```python
from src.services.embeddings import HuggingFaceEmbedding
from src.services.vector_store import VectorStore
from src.models.embedding import EmbeddingConfiguration
from src.config.chroma import ChromaDBConfig

# 1. 서비스 초기화
embedding_config = EmbeddingConfiguration()
embedding_service = HuggingFaceEmbedding(embedding_config)

chroma_config = ChromaDBConfig()
vector_store = VectorStore(chroma_config, embedding_service)

# 2. 문서 준비
documents = [
    "PostgreSQL은 객체-관계형 데이터베이스 관리 시스템입니다.",
    "인덱스는 데이터베이스 검색 성능을 향상시킵니다.",
    "트랜잭션은 ACID 속성을 보장합니다."
]

metadatas = [
    {"source": "postgresql_intro.md", "category": "database"},
    {"source": "index_guide.md", "category": "performance"},
    {"source": "transaction_guide.md", "category": "database"}
]

# 3. 문서 추가 (임베딩 자동 생성)
result = vector_store.add_documents(
    documents=documents,
    metadatas=metadatas
)

print(f"추가된 문서 수: {result['count']}")
print(f"문서 ID: {result['ids']}")
```

### 3.2 스크립트 방식 (대량 처리)

```bash
# JSON 파일에서 문서 인덱싱
python scripts/index_documents.py \
    --source data/documents/ \
    --format json \
    --batch-size 100

# CSV 파일에서 문서 인덱싱
python scripts/index_documents.py \
    --source data/faq.csv \
    --format csv \
    --text-column question \
    --batch-size 50
```

**지원 형식**: JSON, CSV, TXT, Markdown

---

## 4. 검색 테스트

### 4.1 기본 검색

```python
from src.services.vector_store import VectorStore

# 쿼리 실행
query_text = "데이터베이스 성능 최적화"
results = vector_store.query(query_text, top_k=5)

# 결과 출력
for i, (doc, metadata, distance) in enumerate(
    zip(results['documents'], results['metadatas'], results['distances'])
):
    similarity = 1 - distance  # Cosine similarity
    print(f"\n[{i+1}] 유사도: {similarity:.3f}")
    print(f"문서: {doc}")
    print(f"출처: {metadata.get('source', 'N/A')}")
```

### 4.2 필터링 검색

```python
# 메타데이터 필터 적용
results = vector_store.query(
    query_text="SQL 쿼리",
    top_k=3,
    filter={"category": "database"}  # category가 'database'인 문서만
)
```

---

## 5. 성능 검증

### 5.1 임베딩 생성 속도

```python
import time
from src.services.embeddings import HuggingFaceEmbedding
from src.models.embedding import EmbeddingConfiguration

config = EmbeddingConfiguration(batch_size=100)
embedding_service = HuggingFaceEmbedding(config)

# 100개 문서 임베딩 생성 시간 측정
texts = ["테스트 문서"] * 100

start = time.time()
vectors = embedding_service.embed_texts(texts)
elapsed = time.time() - start

print(f"100개 문서 임베딩 생성: {elapsed:.2f}초")
print(f"문서당 평균: {elapsed/100*1000:.1f}ms")
# 예상 결과 (CPU): ~2초, 문서당 ~20ms
```

### 5.2 검색 정확도

```bash
# 벤치마크 테스트 실행
pytest tests/benchmarks/test_embedding_accuracy.py -v

# 예상 결과:
# ✅ Top-1 accuracy: ~75%
# ✅ Top-5 accuracy: ~92%
# ✅ Search latency P95: ~0.3초
```

---

## 6. 문제 해결

### 6.1 일반적인 문제

**Q1. 모델 다운로드가 느려요**

```bash
# 한국 미러 서버 사용 (선택사항)
export HF_ENDPOINT=https://hf-mirror.com
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"
```

**Q2. CUDA out of memory 오류**

```python
# 배치 크기 줄이기
config = EmbeddingConfiguration(
    device=DeviceType.CUDA,
    batch_size=50  # 기본값 100 → 50
)
```

**Q3. Apple Silicon에서 느려요**

```python
# MPS 디바이스 활성화
config = EmbeddingConfiguration(
    device=DeviceType.MPS,  # CPU 대신 MPS
    batch_size=100
)
```

**Q4. 검색 결과가 부정확해요**

```python
# 1. 임베딩 차원 확인
print(embedding_service.get_embedding_dimension())  # 384여야 함

# 2. 모델 검증
is_valid = embedding_service.validate_model()
print(f"모델 유효성: {is_valid}")  # True여야 함

# 3. 벡터 정규화 확인
import numpy as np
vector = embedding_service.embed_text("테스트")
magnitude = np.linalg.norm(vector)
print(f"벡터 크기: {magnitude:.6f}")  # ~1.0이어야 함 (L2 정규화)
```

### 6.2 디버깅 모드

```python
import logging

# 디버그 로그 활성화
logging.basicConfig(level=logging.DEBUG)

# 임베딩 서비스 실행
embedding_service = HuggingFaceEmbedding(config)
vector = embedding_service.embed_text("테스트")
# 상세한 실행 로그 출력
```

---

## 7. 다음 단계

### 학습 자료
- 📖 [완전한 기능 명세](./spec.md)
- 🏗️ [구현 계획](./plan.md)
- 📊 [데이터 모델](./data-model.md)
- ✅ [작업 목록](./tasks.md)

### 실습 예제
```bash
# 1. 전체 테스트 실행
pytest tests/unit/test_embeddings.py -v

# 2. 통합 테스트 실행
pytest tests/integration/test_vector_search.py -v

# 3. 성능 벤치마크
pytest tests/benchmarks/test_embedding_accuracy.py -v --benchmark
```

### 추가 최적화
- 🚀 GPU/MPS 가속 활성화
- 📦 문서 배치 크기 튜닝
- 🔍 검색 필터 활용
- 💾 ChromaDB 인덱스 최적화

---

**Version**: 1.0.0
**Last Updated**: 2025-01-17
**Status**: Ready for Implementation
**Next Step**: Implement `src/services/embeddings.py` (T042)

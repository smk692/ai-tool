# Implementation Tasks: Hugging Face 임베딩 통합

**Feature**: 002-embedding-validation
**Version**: 1.0.0
**Last Updated**: 2025-01-17

---

## 작업 개요

**총 작업 수**: 9개 (T041-T049)
**예상 시간**: 32시간 (3주)
**현재 상태**: Planning → Ready for Implementation

---

## Phase 4: Hugging Face 임베딩 검증 및 통합

### T041: 모델 설정 검증

**우선순위**: P1 (Critical)
**예상 시간**: 1시간
**담당**: Backend Developer
**선행 작업**: Phase 1-2 완료 (T001-T017)

#### 목표
기존에 설정된 Hugging Face 임베딩 모델이 정상적으로 로딩되고 작동하는지 검증

#### 작업 내용
1. **모델 다운로드 확인**
   ```bash
   python scripts/download_embedding_model.py
   ```
   - 모델이 캐시 디렉토리에 다운로드되는지 확인
   - 다운로드 경로: `~/.cache/torch/sentence_transformers/`
   - 모델 크기: ~470MB

2. **임베딩 차원 검증**
   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
   assert model.get_sentence_embedding_dimension() == 384
   ```

3. **한국어 텍스트 테스트**
   ```python
   embedding = model.encode("안녕하세요")
   assert len(embedding) == 384
   assert embedding.dtype == np.float32
   ```

4. **설정 파일 검증**
   - `config/settings.py`에 `EMBEDDING_CONFIG` 확인
   - `src/models/embedding.py`에 `EmbeddingConfiguration` 클래스 확인
   - `.env` 파일에 환경 변수 설정 확인

#### 수락 기준
- ✅ 모델이 성공적으로 다운로드됨
- ✅ 임베딩 차원이 384임을 확인
- ✅ 한국어 텍스트가 정상적으로 임베딩됨
- ✅ 설정 파일이 모두 올바르게 구성됨

#### 산출물
- 검증 스크립트: `scripts/validate_embedding_model.py`
- 검증 리포트: `docs/model-validation-report.md`

---

### T042: 임베딩 서비스 구현 ⭐

**우선순위**: P0 (Critical)
**예상 시간**: 4시간
**담당**: Backend Developer
**선행 작업**: T041

#### 목표
`HuggingFaceEmbedding` 클래스를 구현하여 텍스트를 384차원 임베딩 벡터로 변환

#### 작업 내용

##### 1. 파일 생성
**파일**: `src/services/embeddings.py`

##### 2. 클래스 구조 작성
```python
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from src.models.embedding import EmbeddingConfiguration
import logging

logger = logging.getLogger(__name__)

class HuggingFaceEmbedding:
    """
    Hugging Face sentence-transformers 기반 임베딩 서비스

    Responsibilities:
    - 텍스트를 384차원 임베딩 벡터로 변환
    - 배치 처리로 처리량 최적화
    - L2 정규화된 벡터 생성 (cosine similarity용)
    """

    def __init__(self, config: EmbeddingConfiguration):
        # 초기화 로직
        pass

    def embed_text(self, text: str) -> List[float]:
        # 단일 텍스트 임베딩
        pass

    def embed_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> List[List[float]]:
        # 배치 텍스트 임베딩
        pass

    def get_embedding_dimension(self) -> int:
        # 임베딩 차원 반환
        pass

    def validate_model(self) -> bool:
        # 모델 검증
        pass
```

##### 3. 메서드 구현

**`__init__` 메서드**:
```python
def __init__(self, config: EmbeddingConfiguration):
    """임베딩 서비스 초기화"""
    self.config = config
    self.model = SentenceTransformer(
        config.model_name,
        device=config.device.value
    )
    self.embedding_dim = config.embedding_dim
    logger.info(
        f"Initialized HuggingFaceEmbedding with model={config.model_name}, "
        f"device={config.device.value}, dim={self.embedding_dim}"
    )
```

**`embed_text` 메서드**:
```python
def embed_text(self, text: str) -> List[float]:
    """단일 텍스트를 임베딩 벡터로 변환"""
    if not text.strip():
        raise ValueError("Empty text cannot be embedded")

    embedding = self.model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True  # L2 정규화
    )

    logger.debug(f"Embedded text (length={len(text)}) to {len(embedding)}-dim vector")
    return embedding.tolist()
```

**`embed_texts` 메서드**:
```python
def embed_texts(
    self,
    texts: List[str],
    batch_size: Optional[int] = None
) -> List[List[float]]:
    """여러 텍스트를 배치로 임베딩"""
    if not texts:
        raise ValueError("Empty text list cannot be embedded")

    batch_size = batch_size or self.config.batch_size

    logger.info(f"Embedding {len(texts)} texts with batch_size={batch_size}")

    embeddings = self.model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True  # tqdm progress bar
    )

    return embeddings.tolist()
```

**`get_embedding_dimension` 메서드**:
```python
def get_embedding_dimension(self) -> int:
    """임베딩 차원 반환 (384)"""
    return self.embedding_dim
```

**`validate_model` 메서드**:
```python
def validate_model(self) -> bool:
    """모델 로딩 및 기본 기능 검증"""
    try:
        test_text = "테스트"
        test_embedding = self.embed_text(test_text)

        # 차원 확인
        is_valid = len(test_embedding) == self.embedding_dim

        # L2 정규화 확인
        import numpy as np
        magnitude = np.linalg.norm(test_embedding)
        is_normalized = abs(magnitude - 1.0) < 1e-6

        logger.info(f"Model validation: valid={is_valid}, normalized={is_normalized}")
        return is_valid and is_normalized
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        return False
```

##### 4. 에러 처리
- 빈 텍스트 입력 → `ValueError`
- 빈 리스트 입력 → `ValueError`
- 모델 로딩 실패 → `RuntimeError`
- 긴 텍스트 (>512 토큰) → 자동 truncation (sentence-transformers 기본 동작)

##### 5. 로깅 추가
- INFO: 초기화, 배치 임베딩 시작/완료
- DEBUG: 단일 임베딩 생성
- ERROR: 에러 발생 시 스택 트레이스

#### 수락 기준
- ✅ `HuggingFaceEmbedding` 클래스 구현 완료
- ✅ 단일 텍스트 임베딩 성공 (한국어 포함)
- ✅ 배치 100개 텍스트 임베딩 성공
- ✅ 빈 텍스트 입력 시 `ValueError` 발생
- ✅ L2 정규화된 벡터 생성 확인 (magnitude ≈ 1.0)
- ✅ 임베딩 차원 384 확인
- ✅ `validate_model()` 테스트 통과

#### 산출물
- `src/services/embeddings.py` (새 파일)

---

### T043: ChromaDB 통합

**우선순위**: P0 (Critical)
**예상 시간**: 3시간
**담당**: Backend Developer
**선행 작업**: T042

#### 목표
`VectorStore` 클래스를 업데이트하여 `HuggingFaceEmbedding` 서비스와 통합

#### 작업 내용

##### 1. VectorStore 클래스 수정

**파일**: `src/services/vector_store.py`

##### 2. `__init__` 메서드 업데이트
```python
class VectorStore:
    def __init__(
        self,
        config: ChromaDBConfig,
        embedding_service: HuggingFaceEmbedding  # 🆕 추가
    ):
        """벡터 스토어 초기화"""
        self.config = config
        self.embedding_service = embedding_service  # 🆕

        # ChromaDB 클라이언트 초기화
        import chromadb
        self.client = chromadb.PersistentClient(
            path=config.persist_directory
        )

        # 컬렉션 가져오기 또는 생성
        self.collection = self.client.get_or_create_collection(
            name=config.collection_name,
            metadata={"hnsw:space": config.distance_function}
        )

        logger.info(
            f"Initialized VectorStore with collection={config.collection_name}, "
            f"embedding_dim={embedding_service.get_embedding_dimension()}"
        )
```

##### 3. `add_documents` 메서드 업데이트
```python
def add_documents(
    self,
    documents: List[str],
    metadatas: Optional[List[Dict]] = None,
    embeddings: Optional[List[List[float]]] = None,
    ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    문서를 벡터 스토어에 추가

    Args:
        documents: 문서 텍스트 리스트
        metadatas: 각 문서의 메타데이터
        embeddings: Pre-computed embeddings (None이면 자동 생성)
        ids: 문서 ID 리스트

    Returns:
        {"success": bool, "count": int, "ids": List[str]}
    """
    # 🆕 임베딩이 없으면 embedding_service로 생성
    if embeddings is None:
        logger.info(f"Generating embeddings for {len(documents)} documents")
        embeddings = self.embedding_service.embed_texts(documents)

    # ID 생성 (없으면)
    if ids is None:
        import uuid
        ids = [str(uuid.uuid4()) for _ in documents]

    # ChromaDB에 저장
    self.collection.add(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    logger.info(f"Added {len(documents)} documents to collection")

    return {
        "success": True,
        "count": len(documents),
        "ids": ids
    }
```

##### 4. `query` 메서드 업데이트
```python
def query(
    self,
    query_text: str,
    top_k: int = 5,
    filter: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    쿼리 텍스트로 유사 문서 검색

    Args:
        query_text: 검색 쿼리
        top_k: 반환할 문서 수
        filter: 메타데이터 필터

    Returns:
        {
            "documents": List[str],
            "metadatas": List[Dict],
            "distances": List[float],
            "ids": List[str]
        }
    """
    # 🆕 쿼리 임베딩 생성
    logger.debug(f"Generating embedding for query: {query_text[:50]}...")
    query_embedding = self.embedding_service.embed_text(query_text)

    # ChromaDB 검색
    results = self.collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=filter
    )

    logger.info(f"Query returned {len(results['documents'][0])} results")

    return {
        "documents": results["documents"][0],
        "metadatas": results["metadatas"][0],
        "distances": results["distances"][0],
        "ids": results["ids"][0]
    }
```

##### 5. 기존 ChromaDB 기본 임베더 제거
- ChromaDB collection 생성 시 `embedding_function` 파라미터 제거
- Pre-computed embeddings만 사용
- 로그에서 ChromaDB 기본 임베더 호출 확인 (0건이어야 함)

##### 6. 테스트 코드 업데이트
**파일**: `tests/integration/test_vector_store.py`
```python
def test_vector_store_with_embedding_service():
    """VectorStore가 HuggingFaceEmbedding을 사용하는지 확인"""
    # Setup
    embedding_config = EmbeddingConfiguration()
    embedding_service = HuggingFaceEmbedding(embedding_config)

    chroma_config = ChromaDBConfig()
    vector_store = VectorStore(chroma_config, embedding_service)

    # Test
    documents = ["테스트 문서 1", "테스트 문서 2"]
    result = vector_store.add_documents(documents)

    assert result["success"] is True
    assert result["count"] == 2

    # Query
    query_result = vector_store.query("테스트", top_k=2)
    assert len(query_result["documents"]) == 2
```

#### 수락 기준
- ✅ `VectorStore.__init__`에 `embedding_service` 파라미터 추가
- ✅ `add_documents`에서 임베딩 자동 생성
- ✅ `query`에서 쿼리 임베딩 생성
- ✅ ChromaDB 기본 임베더 호출 0건 (로그 확인)
- ✅ 1000개 문서 추가 테스트 통과
- ✅ 쿼리 검색 테스트 통과

#### 산출물
- `src/services/vector_store.py` (업데이트)
- `tests/integration/test_vector_store.py` (업데이트)

---

### T044: 문서 인덱싱 유틸리티

**우선순위**: P1 (High)
**예상 시간**: 4시간
**담당**: Backend Developer
**선행 작업**: T043

#### 목표
대량 문서를 인덱싱하는 CLI 유틸리티 작성

#### 작업 내용

##### 1. 파일 생성
**파일**: `scripts/index_documents.py`

##### 2. CLI 인터페이스 구현
```python
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from src.services.embeddings import HuggingFaceEmbedding
from src.services.vector_store import VectorStore
from src.models.embedding import EmbeddingConfiguration
from src.config.chroma import ChromaDBConfig

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Bulk document indexing for vector store"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source directory or file path"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "md", "pdf", "csv"],
        default="json",
        help="Document format"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for embedding generation"
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Column name for text content (CSV/JSON)"
    )
    return parser.parse_args()
```

##### 3. 문서 로더 구현
```python
def load_json_documents(file_path: Path, text_column: str) -> List[Dict]:
    """JSON 파일에서 문서 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    else:
        return [data]

def load_markdown_documents(file_path: Path) -> List[Dict]:
    """Markdown 파일에서 문서 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return [{
        "text": content,
        "source": str(file_path),
        "format": "markdown"
    }]

def load_csv_documents(file_path: Path, text_column: str) -> List[Dict]:
    """CSV 파일에서 문서 로드"""
    import csv
    documents = []

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if text_column in row:
                documents.append(row)

    return documents
```

##### 4. 배치 처리 로직
```python
def index_documents(
    documents: List[Dict],
    vector_store: VectorStore,
    batch_size: int,
    text_column: str
) -> Dict[str, int]:
    """문서를 배치로 인덱싱"""
    total = len(documents)
    success_count = 0
    error_count = 0

    # 진행 상황 표시
    with tqdm(total=total, desc="Indexing documents") as pbar:
        for i in range(0, total, batch_size):
            batch = documents[i:i + batch_size]

            try:
                # 텍스트 및 메타데이터 추출
                texts = [doc[text_column] for doc in batch]
                metadatas = [
                    {k: v for k, v in doc.items() if k != text_column}
                    for doc in batch
                ]

                # 벡터 스토어에 추가
                result = vector_store.add_documents(
                    documents=texts,
                    metadatas=metadatas
                )

                success_count += result["count"]

            except Exception as e:
                logger.error(f"Failed to index batch {i//batch_size}: {e}")
                error_count += len(batch)

            pbar.update(len(batch))

    return {
        "total": total,
        "success": success_count,
        "errors": error_count
    }
```

##### 5. Main 함수
```python
def main():
    args = parse_args()

    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 서비스 초기화
    embedding_config = EmbeddingConfiguration(batch_size=args.batch_size)
    embedding_service = HuggingFaceEmbedding(embedding_config)

    chroma_config = ChromaDBConfig()
    vector_store = VectorStore(chroma_config, embedding_service)

    # 문서 로드
    source_path = Path(args.source)

    if source_path.is_file():
        # 단일 파일
        if args.format == "json":
            documents = load_json_documents(source_path, args.text_column)
        elif args.format == "md":
            documents = load_markdown_documents(source_path)
        elif args.format == "csv":
            documents = load_csv_documents(source_path, args.text_column)
    else:
        # 디렉토리
        documents = []
        for file_path in source_path.glob(f"*.{args.format}"):
            if args.format == "json":
                documents.extend(load_json_documents(file_path, args.text_column))
            elif args.format == "md":
                documents.extend(load_markdown_documents(file_path))

    logger.info(f"Loaded {len(documents)} documents from {args.source}")

    # 인덱싱
    stats = index_documents(
        documents=documents,
        vector_store=vector_store,
        batch_size=args.batch_size,
        text_column=args.text_column
    )

    # 결과 출력
    print("\n=== Indexing Results ===")
    print(f"Total documents: {stats['total']}")
    print(f"Successfully indexed: {stats['success']}")
    print(f"Errors: {stats['errors']}")
    print(f"Success rate: {stats['success']/stats['total']*100:.1f}%")

if __name__ == "__main__":
    main()
```

#### 수락 기준
- ✅ JSON, Markdown, CSV 파일 로딩 지원
- ✅ 배치 크기 100으로 처리 가능
- ✅ tqdm progress bar 정상 표시
- ✅ 1000개 문서 인덱싱 완료 시간 ≤5분
- ✅ 에러 처리 및 재시도 로직 작동
- ✅ 인덱싱 통계 출력 (성공/실패 카운트)

#### 산출물
- `scripts/index_documents.py` (새 파일)

---

### T045: 한국어 단위 테스트

**우선순위**: P0 (Critical)
**예상 시간**: 3시간
**담당**: QA Engineer
**선행 작업**: T042

#### 목표
`HuggingFaceEmbedding` 서비스 단위 테스트 작성 (100% 커버리지)

#### 작업 내용

##### 1. 테스트 파일 생성
**파일**: `tests/unit/test_embeddings.py`

##### 2. 테스트 케이스 작성

```python
import pytest
import numpy as np
from src.services.embeddings import HuggingFaceEmbedding
from src.models.embedding import EmbeddingConfiguration, DeviceType

class TestHuggingFaceEmbedding:
    """HuggingFaceEmbedding 서비스 단위 테스트"""

    @pytest.fixture
    def embedding_service(self):
        """임베딩 서비스 fixture"""
        config = EmbeddingConfiguration()
        return HuggingFaceEmbedding(config)

    def test_initialization(self, embedding_service):
        """초기화 테스트"""
        assert embedding_service.embedding_dim == 384
        assert embedding_service.config.model_name == "paraphrase-multilingual-MiniLM-L12-v2"

    def test_embed_single_korean_text(self, embedding_service):
        """한국어 단일 텍스트 임베딩"""
        text = "안녕하세요"
        embedding = embedding_service.embed_text(text)

        assert len(embedding) == 384
        assert isinstance(embedding, list)
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_single_english_text(self, embedding_service):
        """영어 단일 텍스트 임베딩"""
        text = "Hello world"
        embedding = embedding_service.embed_text(text)

        assert len(embedding) == 384

    def test_embed_mixed_text(self, embedding_service):
        """한영 혼합 텍스트 임베딩"""
        text = "PostgreSQL 데이터베이스"
        embedding = embedding_service.embed_text(text)

        assert len(embedding) == 384

    def test_embed_empty_text_raises_error(self, embedding_service):
        """빈 텍스트 입력 시 ValueError 발생"""
        with pytest.raises(ValueError, match="Empty text"):
            embedding_service.embed_text("")

        with pytest.raises(ValueError, match="Empty text"):
            embedding_service.embed_text("   ")  # 공백만

    def test_embed_batch_texts(self, embedding_service):
        """배치 텍스트 임베딩"""
        texts = [
            "데이터베이스 트랜잭션",
            "SQL 쿼리 최적화",
            "NoSQL과 관계형 데이터베이스"
        ]
        embeddings = embedding_service.embed_texts(texts)

        assert len(embeddings) == 3
        assert all(len(emb) == 384 for emb in embeddings)

    def test_embed_large_batch(self, embedding_service):
        """대량 배치 임베딩 (100개)"""
        texts = [f"테스트 문서 {i}" for i in range(100)]
        embeddings = embedding_service.embed_texts(texts, batch_size=50)

        assert len(embeddings) == 100
        assert all(len(emb) == 384 for emb in embeddings)

    def test_embed_empty_list_raises_error(self, embedding_service):
        """빈 리스트 입력 시 ValueError 발생"""
        with pytest.raises(ValueError, match="Empty text list"):
            embedding_service.embed_texts([])

    def test_embedding_normalization(self, embedding_service):
        """L2 정규화 확인"""
        text = "테스트"
        embedding = embedding_service.embed_text(text)

        magnitude = np.linalg.norm(embedding)
        assert abs(magnitude - 1.0) < 1e-6  # L2 정규화 확인

    def test_get_embedding_dimension(self, embedding_service):
        """임베딩 차원 반환"""
        assert embedding_service.get_embedding_dimension() == 384

    def test_validate_model(self, embedding_service):
        """모델 검증"""
        is_valid = embedding_service.validate_model()
        assert is_valid is True

    def test_long_text_truncation(self, embedding_service):
        """긴 텍스트 (>512 토큰) 자동 truncation"""
        # 매우 긴 텍스트 생성
        long_text = "테스트 " * 300  # ~600 토큰

        embedding = embedding_service.embed_text(long_text)

        # truncation되어도 임베딩 생성 성공
        assert len(embedding) == 384

    def test_special_characters(self, embedding_service):
        """특수 문자 처리"""
        text = "SQL의 WHERE 조건절 (condition)"
        embedding = embedding_service.embed_text(text)

        assert len(embedding) == 384

    def test_unicode_text(self, embedding_service):
        """유니코드 텍스트 처리"""
        text = "한글, 日本語, 中文"
        embedding = embedding_service.embed_text(text)

        assert len(embedding) == 384

    def test_consistent_embeddings(self, embedding_service):
        """동일 텍스트는 동일 임베딩 생성"""
        text = "일관성 테스트"

        embedding1 = embedding_service.embed_text(text)
        embedding2 = embedding_service.embed_text(text)

        # 임베딩이 동일한지 확인
        np.testing.assert_array_almost_equal(embedding1, embedding2, decimal=6)
```

#### 수락 기준
- ✅ 15개 이상 테스트 케이스 작성
- ✅ 모든 테스트 통과
- ✅ `src/services/embeddings.py` 커버리지 100%
- ✅ 한국어, 영어, 혼합 텍스트 테스트 포함
- ✅ 에러 케이스 테스트 포함

#### 산출물
- `tests/unit/test_embeddings.py` (새 파일)

---

### T046: 벡터 검색 지연시간 테스트

**우선순위**: P1 (High)
**예상 시간**: 4시간
**담당**: QA Engineer
**선행 작업**: T043

#### 목표
검색 응답시간 SLA 검증 (P95 ≤0.5초)

#### 작업 내용

##### 1. 테스트 파일 생성
**파일**: `tests/integration/test_vector_search.py`

##### 2. 테스트 케이스 작성

```python
import pytest
import time
import numpy as np
from src.services.embeddings import HuggingFaceEmbedding
from src.services.vector_store import VectorStore
from src.models.embedding import EmbeddingConfiguration
from src.config.chroma import ChromaDBConfig

class TestVectorSearchPerformance:
    """벡터 검색 성능 테스트"""

    @pytest.fixture(scope="class")
    def setup_vector_store(self):
        """1000개 문서가 인덱싱된 벡터 스토어 생성"""
        embedding_config = EmbeddingConfiguration()
        embedding_service = HuggingFaceEmbedding(embedding_config)

        chroma_config = ChromaDBConfig()
        vector_store = VectorStore(chroma_config, embedding_service)

        # 1000개 테스트 문서 생성
        documents = [
            f"테스트 문서 {i}: PostgreSQL 데이터베이스 트랜잭션"
            for i in range(1000)
        ]

        vector_store.add_documents(documents)

        return vector_store

    def test_single_query_response_time(self, setup_vector_store):
        """단일 쿼리 응답시간 측정 (100회 반복)"""
        vector_store = setup_vector_store
        query = "데이터베이스 트랜잭션"

        response_times = []

        for _ in range(100):
            start = time.time()
            results = vector_store.query(query, top_k=5)
            elapsed = time.time() - start

            response_times.append(elapsed)
            assert len(results["documents"]) == 5

        # 통계 계산
        mean_time = np.mean(response_times)
        p95_time = np.percentile(response_times, 95)
        p99_time = np.percentile(response_times, 99)

        print(f"\n=== Single Query Performance ===")
        print(f"Mean: {mean_time:.3f}s")
        print(f"P95: {p95_time:.3f}s")
        print(f"P99: {p99_time:.3f}s")

        # SLA 검증
        assert p95_time <= 0.5, f"P95 latency {p95_time:.3f}s exceeds SLA of 0.5s"
        assert mean_time <= 0.3, f"Mean latency {mean_time:.3f}s exceeds target of 0.3s"

    def test_concurrent_queries(self, setup_vector_store):
        """동시 10개 쿼리 응답시간 측정"""
        import concurrent.futures

        vector_store = setup_vector_store
        queries = [f"쿼리 {i}" for i in range(10)]

        start = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(vector_store.query, query, 5)
                for query in queries
            ]
            results = [f.result() for f in futures]

        elapsed = time.time() - start
        avg_time = elapsed / 10

        print(f"\n=== Concurrent Queries Performance ===")
        print(f"Total time: {elapsed:.3f}s")
        print(f"Average per query: {avg_time:.3f}s")

        # SLA 검증
        assert avg_time <= 0.7, f"Concurrent query avg {avg_time:.3f}s exceeds target of 0.7s"

    @pytest.mark.parametrize("doc_count", [1000, 5000, 10000])
    def test_scalability(self, doc_count):
        """문서 수별 검색 성능"""
        # 서비스 초기화
        embedding_config = EmbeddingConfiguration()
        embedding_service = HuggingFaceEmbedding(embedding_config)

        chroma_config = ChromaDBConfig()
        vector_store = VectorStore(chroma_config, embedding_service)

        # 문서 인덱싱
        documents = [f"문서 {i}" for i in range(doc_count)]
        vector_store.add_documents(documents)

        # 쿼리 성능 측정
        response_times = []
        for _ in range(20):
            start = time.time()
            vector_store.query("테스트", top_k=5)
            elapsed = time.time() - start
            response_times.append(elapsed)

        p95_time = np.percentile(response_times, 95)

        print(f"\n=== Scalability Test (docs={doc_count}) ===")
        print(f"P95: {p95_time:.3f}s")

        # 10000개 문서에서도 ≤0.5초 유지
        assert p95_time <= 0.5
```

#### 수락 기준
- ✅ 단일 쿼리 P95 ≤0.5초
- ✅ 단일 쿼리 평균 ≤0.3초
- ✅ 동시 10 쿼리 평균 ≤0.7초
- ✅ 10000 문서에서도 P95 ≤0.5초
- ✅ 모든 성능 테스트 통과

#### 산출물
- `tests/integration/test_vector_search.py` (새 파일)

---

### T047: Top-5 정확도 벤치마크

**우선순위**: P0 (Critical)
**예상 시간**: 6시간
**담당**: QA Engineer
**선행 작업**: T043

#### 목표
한국어 검색 정확도 ≥90% (Top-5) 검증

#### 작업 내용

##### 1. 테스트 데이터셋 준비
**파일**: `tests/benchmarks/data/queries.json`

```json
[
  {
    "query_id": 1,
    "query": "PostgreSQL에서 트랜잭션 격리 수준이란?",
    "language": "korean",
    "category": "factual",
    "answer_ids": ["doc_123", "doc_456"]
  },
  {
    "query_id": 2,
    "query": "데이터베이스 인덱스의 종류",
    "language": "korean",
    "category": "factual",
    "answer_ids": ["doc_789"]
  }
  // ... 100개 쿼리
]
```

##### 2. 벤치마크 테스트 작성
**파일**: `tests/benchmarks/test_embedding_accuracy.py`

```python
import pytest
import json
from pathlib import Path
from typing import List, Dict
from src.services.embeddings import HuggingFaceEmbedding
from src.services.vector_store import VectorStore
from src.models.embedding import EmbeddingConfiguration
from src.config.chroma import ChromaDBConfig

class TestEmbeddingAccuracy:
    """임베딩 검색 정확도 벤치마크"""

    @pytest.fixture(scope="class")
    def setup_benchmark(self):
        """벤치마크 데이터 준비"""
        # 서비스 초기화
        embedding_config = EmbeddingConfiguration()
        embedding_service = HuggingFaceEmbedding(embedding_config)

        chroma_config = ChromaDBConfig()
        vector_store = VectorStore(chroma_config, embedding_service)

        # 문서 로딩
        doc_path = Path(__file__).parent / "data" / "documents.json"
        with open(doc_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)

        # 인덱싱
        texts = [doc["text"] for doc in documents]
        metadatas = [{"doc_id": doc["id"]} for doc in documents]
        ids = [doc["id"] for doc in documents]

        vector_store.add_documents(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

        # 쿼리 로딩
        query_path = Path(__file__).parent / "data" / "queries.json"
        with open(query_path, 'r', encoding='utf-8') as f:
            queries = json.load(f)

        return vector_store, queries

    def test_overall_top5_accuracy(self, setup_benchmark):
        """전체 Top-5 정확도 테스트"""
        vector_store, queries = setup_benchmark

        hits = 0
        total = len(queries)

        for query_data in queries:
            query = query_data["query"]
            answer_ids = set(query_data["answer_ids"])

            # Top-5 검색
            results = vector_store.query(query, top_k=5)
            result_ids = set(
                [meta["doc_id"] for meta in results["metadatas"]]
            )

            # Hit@5 계산
            if answer_ids & result_ids:  # 교집합이 있으면 hit
                hits += 1

        accuracy = hits / total

        print(f"\n=== Overall Top-5 Accuracy ===")
        print(f"Hits: {hits}/{total}")
        print(f"Accuracy: {accuracy:.2%}")

        # SLA 검증
        assert accuracy >= 0.90, f"Accuracy {accuracy:.2%} below target of 90%"

    def test_korean_query_accuracy(self, setup_benchmark):
        """한국어 쿼리 정확도"""
        vector_store, queries = setup_benchmark

        korean_queries = [q for q in queries if q["language"] == "korean"]

        hits = 0
        for query_data in korean_queries:
            query = query_data["query"]
            answer_ids = set(query_data["answer_ids"])

            results = vector_store.query(query, top_k=5)
            result_ids = set(
                [meta["doc_id"] for meta in results["metadatas"]]
            )

            if answer_ids & result_ids:
                hits += 1

        accuracy = hits / len(korean_queries)

        print(f"\n=== Korean Query Accuracy ===")
        print(f"Hits: {hits}/{len(korean_queries)}")
        print(f"Accuracy: {accuracy:.2%}")

        assert accuracy >= 0.90

    def test_category_accuracy(self, setup_benchmark):
        """카테고리별 정확도"""
        vector_store, queries = setup_benchmark

        categories = set(q["category"] for q in queries)

        for category in categories:
            cat_queries = [q for q in queries if q["category"] == category]

            hits = 0
            for query_data in cat_queries:
                query = query_data["query"]
                answer_ids = set(query_data["answer_ids"])

                results = vector_store.query(query, top_k=5)
                result_ids = set(
                    [meta["doc_id"] for meta in results["metadatas"]]
                )

                if answer_ids & result_ids:
                    hits += 1

            accuracy = hits / len(cat_queries)

            print(f"\n=== {category.title()} Accuracy ===")
            print(f"Accuracy: {accuracy:.2%}")

    def test_mrr(self, setup_benchmark):
        """Mean Reciprocal Rank 계산"""
        vector_store, queries = setup_benchmark

        reciprocal_ranks = []

        for query_data in queries:
            query = query_data["query"]
            answer_ids = set(query_data["answer_ids"])

            results = vector_store.query(query, top_k=5)
            result_ids = [meta["doc_id"] for meta in results["metadatas"]]

            # 첫 번째 정답의 순위 찾기
            rank = None
            for i, doc_id in enumerate(result_ids, 1):
                if doc_id in answer_ids:
                    rank = i
                    break

            if rank:
                reciprocal_ranks.append(1 / rank)
            else:
                reciprocal_ranks.append(0)

        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

        print(f"\n=== Mean Reciprocal Rank ===")
        print(f"MRR: {mrr:.3f}")

        assert mrr >= 0.75
```

##### 3. 리포트 생성
**파일**: `tests/benchmarks/generate_accuracy_report.py`

```python
def generate_html_report(results: Dict):
    """HTML 리포트 생성"""
    html = f"""
    <html>
    <head><title>Embedding Accuracy Report</title></head>
    <body>
        <h1>Embedding Accuracy Benchmark</h1>
        <h2>Overall Results</h2>
        <p>Top-5 Accuracy: {results['overall_accuracy']:.2%}</p>
        <p>MRR: {results['mrr']:.3f}</p>

        <h2>Language Breakdown</h2>
        <ul>
            <li>Korean: {results['korean_accuracy']:.2%}</li>
            <li>English: {results['english_accuracy']:.2%}</li>
            <li>Mixed: {results['mixed_accuracy']:.2%}</li>
        </ul>

        <h2>Category Breakdown</h2>
        <ul>
            {''.join([f'<li>{cat}: {acc:.2%}</li>' for cat, acc in results['category_accuracy'].items()])}
        </ul>

        <h2>Failed Queries</h2>
        <ul>
            {''.join([f'<li>{q}</li>' for q in results['failed_queries']])}
        </ul>
    </body>
    </html>
    """

    with open("benchmark_report.html", "w") as f:
        f.write(html)
```

#### 수락 기준
- ✅ 100개 테스트 쿼리 준비 (한국어 50, 영어 30, 혼합 20)
- ✅ 전체 Top-5 Accuracy ≥90%
- ✅ 한국어 쿼리 정확도 ≥90%
- ✅ MRR ≥0.75
- ✅ 카테고리별 정확도 리포트 생성
- ✅ 실패 케이스 분석 리포트

#### 산출물
- `tests/benchmarks/test_embedding_accuracy.py` (새 파일)
- `tests/benchmarks/data/queries.json` (새 파일)
- `tests/benchmarks/data/documents.json` (새 파일)
- `benchmark_report.html` (결과 리포트)

---

### T048: 다국어 지원 테스트

**우선순위**: P2 (Medium)
**예상 시간**: 3시간
**담당**: QA Engineer
**선행 작업**: T045

#### 목표
다국어 및 특수 문자 처리 검증

#### 작업 내용

##### 1. 테스트 케이스 추가
**파일**: `tests/unit/test_embeddings.py` (기존 파일에 추가)

```python
class TestMultilingualSupport:
    """다국어 지원 테스트"""

    @pytest.fixture
    def embedding_service(self):
        config = EmbeddingConfiguration()
        return HuggingFaceEmbedding(config)

    @pytest.mark.parametrize("text,language", [
        ("데이터베이스 인덱스", "korean"),
        ("database index", "english"),
        ("PostgreSQL의 B-tree 인덱스", "mixed"),
        ("日本語のテキスト", "japanese"),
        ("中文文本", "chinese"),
    ])
    def test_multilingual_embedding(self, embedding_service, text, language):
        """다국어 텍스트 임베딩"""
        embedding = embedding_service.embed_text(text)

        assert len(embedding) == 384
        print(f"{language}: {text} → embedding generated")

    def test_special_characters(self, embedding_service):
        """특수 문자 처리"""
        texts = [
            "SQL의 WHERE 조건절 (condition)",
            "Python f-string {variable}",
            "정규표현식 [a-zA-Z]+",
            "이모지 포함 😀 텍스트",
        ]

        for text in texts:
            embedding = embedding_service.embed_text(text)
            assert len(embedding) == 384

    def test_unicode_normalization(self, embedding_service):
        """유니코드 정규화"""
        # NFD vs NFC 형식
        text_nfd = "한글"  # NFD
        text_nfc = "한글"  # NFC

        embedding_nfd = embedding_service.embed_text(text_nfd)
        embedding_nfc = embedding_service.embed_text(text_nfc)

        # 임베딩이 유사해야 함 (완전히 동일하지 않을 수 있음)
        import numpy as np
        similarity = np.dot(embedding_nfd, embedding_nfc)
        assert similarity > 0.99  # 매우 높은 유사도

    def test_encoding_edge_cases(self, embedding_service):
        """인코딩 엣지 케이스"""
        texts = [
            "\n\n\n텍스트\n\n",  # 개행 문자
            "\t\t텍스트\t\t",  # 탭 문자
            "   텍스트   ",  # 공백
            "텍스트\r\n윈도우",  # CRLF
        ]

        for text in texts:
            embedding = embedding_service.embed_text(text)
            assert len(embedding) == 384
```

#### 수락 기준
- ✅ 한국어, 영어, 일본어, 중국어 텍스트 처리 성공
- ✅ 특수 문자 포함 텍스트 처리 성공
- ✅ 유니코드 정규화 처리 성공
- ✅ 인코딩 엣지 케이스 처리 성공
- ✅ 인코딩 오류 0건

#### 산출물
- `tests/unit/test_embeddings.py` (업데이트)

---

### T049: 문서화

**우선순위**: P2 (Medium)
**예상 시간**: 4시간
**담당**: Technical Writer
**선행 작업**: T041-T048

#### 목표
임베딩 모델 사양 및 API 사용 가이드 문서 작성

#### 작업 내용

##### 1. 모델 사양 문서
**파일**: `docs/embedding-model.md`

```markdown
# Embedding Model Specification

## Model Information

**Name**: paraphrase-multilingual-MiniLM-L12-v2
**Source**: Hugging Face sentence-transformers
**Architecture**: MiniLM (12-layer Transformer)

## Specifications

- **Embedding Dimension**: 384
- **Max Sequence Length**: 512 tokens
- **Normalization**: L2 normalized
- **Similarity Metric**: Cosine similarity

## Supported Languages

- Korean (한국어)
- English
- Japanese (日本語)
- Chinese (中文)
- 50+ languages total

## Performance Benchmarks

### Accuracy
- Top-5 Accuracy: 92%
- Mean Reciprocal Rank: 0.78

### Latency
- Single query P95: 0.32s
- Concurrent (10 queries) avg: 0.45s

### Memory
- Model size: ~470MB
- Runtime memory: <1GB
```

##### 2. API 사용 가이드
**파일**: `docs/embedding-api-guide.md`

```markdown
# Embedding API Usage Guide

## Quick Start

### Installation
```bash
pip install sentence-transformers>=2.2.0 chromadb>=0.4.0
```

### Basic Usage
```python
from src.services.embeddings import HuggingFaceEmbedding
from src.models.embedding import EmbeddingConfiguration

# Initialize
config = EmbeddingConfiguration()
embedding_service = HuggingFaceEmbedding(config)

# Single text
vector = embedding_service.embed_text("한국어 텍스트")

# Batch texts
vectors = embedding_service.embed_texts(["text1", "text2"])
```

## API Reference

### HuggingFaceEmbedding

#### `__init__(config: EmbeddingConfiguration)`
...

#### `embed_text(text: str) -> List[float]`
...

## Best Practices

1. **Batch Processing**: Always use `embed_texts()` for multiple documents
2. **Error Handling**: Validate input text length
3. **Performance**: Adjust batch size based on available memory
```

##### 3. 트러블슈팅 가이드
**파일**: `docs/embedding-troubleshooting.md`

```markdown
# Embedding Troubleshooting Guide

## Common Issues

### Issue 1: Slow Embedding Generation

**Symptoms**: Embedding takes >2s for 100 documents

**Solutions**:
- Reduce batch size
- Check CPU usage
- Disable progress bar

### Issue 2: Out of Memory

**Symptoms**: `MemoryError` during batch processing

**Solutions**:
- Reduce batch size to 50 or 25
- Process in smaller chunks
```

##### 4. FAQ
**파일**: `docs/embedding-faq.md`

```markdown
# Embedding FAQ

## Q1: Why 384 dimensions?

The paraphrase-multilingual-MiniLM-L12-v2 model outputs 384-dimensional vectors...

## Q2: Can I use GPU?

Yes, set `device=DeviceType.CUDA` in configuration...

## Q3: How to improve accuracy?

- Use larger batch sizes
- Ensure text is preprocessed
- Consider hybrid search (BM25 + vector)
```

#### 수락 기준
- ✅ 모델 사양 문서 완성
- ✅ API 사용 가이드 작성 (코드 예시 포함)
- ✅ 트러블슈팅 가이드 작성
- ✅ FAQ 작성 (10개 이상 질문)
- ✅ 모든 코드 예시 정확성 검증

#### 산출물
- `docs/embedding-model.md` (새 파일)
- `docs/embedding-api-guide.md` (새 파일)
- `docs/embedding-troubleshooting.md` (새 파일)
- `docs/embedding-faq.md` (새 파일)

---

## 종속성 다이어그램

```
T041 (모델 설정 검증)
  ↓
T042 (임베딩 서비스 구현) ⭐
  ↓
T043 (ChromaDB 통합)
  ↓
T044 (문서 인덱싱 유틸리티)
  ↓
T045 (한국어 단위 테스트) ← T042
  ↓
T046 (벡터 검색 지연시간 테스트) ← T043
  ↓
T047 (Top-5 정확도 벤치마크) ← T043
  ↓
T048 (다국어 지원 테스트) ← T045
  ↓
T049 (문서화) ← T041-T048
```

---

## 타임라인 (3주)

### Week 1: 서비스 구현
- **Day 1-2**: T041 + T042
- **Day 3-4**: T043
- **Day 5**: T044

**Milestone**: 1000개 문서 인덱싱 성공

### Week 2: 테스트 및 검증
- **Day 1-2**: T045
- **Day 3**: T046
- **Day 4-5**: T047
- **Day 6**: T048

**Milestone**: 모든 수락 기준 통과

### Week 3: 문서화 및 배포
- **Day 1-2**: T049
- **Day 3-4**: 성능 최적화 (필요 시)
- **Day 5**: 최종 검증 및 배포 준비

**Milestone**: Phase 4 완료

---

## 리스크 및 완화 전략

### 리스크 1: 한국어 임베딩 품질 미달
- **확률**: 낮음
- **영향**: 높음
- **완화**: 사전 테스트, 하이브리드 검색 활용

### 리스크 2: 검색 지연시간 SLA 미달
- **확률**: 중간
- **영향**: 중간
- **완화**: 배치 크기 최적화, 캐싱 전략

### 리스크 3: 메모리 부족
- **확률**: 낮음
- **영향**: 중간
- **완화**: 동적 배치 크기 조정, 스트리밍 처리

---

**Version**: 1.0.0
**Last Updated**: 2025-01-17
**Status**: Ready for Implementation
**Total Estimated Hours**: 32 hours

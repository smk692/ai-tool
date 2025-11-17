# Feature Specification: Hugging Face 다국어 임베딩 통합 및 검증

**Feature ID**: 002-embedding-validation
**User Story**: User Story 2 - Hugging Face 다국어 임베딩 통합
**Status**: Planning
**Version**: 1.0.0
**Last Updated**: 2025-01-17

---

## 1. 개요

### 1.1 Feature 이름
**Hugging Face 다국어 임베딩 통합 및 검증 (Hugging Face Multilingual Embedding Integration & Validation)**

### 1.2 목적
OpenAI 임베딩에서 Hugging Face 오픈소스 임베딩 모델로 전환하여:
- **한국어 지원 개선**: 다국어 모델을 통한 한국어 임베딩 품질 향상
- **비용 최적화**: 무료 오픈소스 모델 사용으로 API 비용 제거
- **독립성 확보**: 외부 API 의존성 제거 및 로컬 모델 사용

### 1.3 범위

**포함 사항**:
- ✅ Hugging Face `paraphrase-multilingual-MiniLM-L12-v2` 모델 통합
- ✅ `HuggingFaceEmbedding` 서비스 구현
- ✅ ChromaDB와 임베딩 서비스 통합
- ✅ 문서 인덱싱 파이프라인 구현
- ✅ 한국어 임베딩 품질 검증 (Top-5 정확도 ≥90%)
- ✅ 검색 지연시간 검증 (≤0.5초)
- ✅ 다국어 지원 테스트
- ✅ OpenAI 임베딩 완전 제거

**제외 사항**:
- ❌ 다른 임베딩 모델 평가 (모델은 이미 선정됨)
- ❌ GPU 최적화 (CPU 전용 환경)
- ❌ 하이브리드 검색 개선 (별도 User Story)
- ❌ 프론트엔드 UI 변경

---

## 2. 배경 및 현황

### 2.1 Phase 1-2에서 완료된 사항 (T001-T017)

**✅ 완료**:
- `sentence-transformers>=2.2.0` 설치
- `paraphrase-multilingual-MiniLM-L12-v2` 모델 선정
- `config/settings.py`에 임베딩 설정 추가
- `src/models/embedding.py`에 `EmbeddingConfiguration` 엔티티 정의
- `scripts/download_embedding_model.py` 모델 캐싱 스크립트 작성
- `scripts/init_vector_store.py` ChromaDB 초기화 스크립트 작성
- ChromaDB 컬렉션 생성 (cosine similarity)

**모델 스펙**:
- **모델명**: `paraphrase-multilingual-MiniLM-L12-v2`
- **소스**: Hugging Face sentence-transformers
- **임베딩 차원**: 384
- **최대 시퀀스 길이**: 512 토큰
- **지원 언어**: 50+ 언어 (한국어 포함)
- **추론 속도**: CPU에서 빠른 추론 가능

### 2.2 현재 누락된 사항 (Phase 4: T041-T049)

**🚨 핵심 누락 구현**:
1. **`src/services/embeddings.py` 미구현**:
   - `HuggingFaceEmbedding` 클래스 없음
   - `scripts/validate_embedding_quality.py`가 이 클래스를 참조하지만 존재하지 않음

2. **ChromaDB 통합 불완전**:
   - `src/services/vector_store.py`가 ChromaDB 기본 임베더 사용
   - 설정된 Hugging Face 모델과 연결되지 않음
   - Pre-computed embeddings 전달 로직 필요

3. **문서 인덱싱 파이프라인 부재**:
   - 대량 문서 처리 유틸리티 없음
   - 배치 처리 및 진행 상황 추적 기능 부족

4. **테스트 및 검증 부족**:
   - 한국어 임베딩 품질 테스트 없음
   - Top-5 정확도 벤치마크 없음
   - 검색 지연시간 SLA 검증 없음
   - 다국어 지원 테스트 없음

5. **문서화 미완료**:
   - 임베딩 모델 사양 문서 없음
   - API 사용 가이드 없음

### 2.3 Phase 3 (Claude Code 통합) 의존성

**선행 조건**:
- ✅ Phase 3 (User Story 1) 완료 필요
- ✅ Claude Code API 통합 완료
- ✅ 모든 체인 Claude로 마이그레이션 완료
- ✅ 테스트 커버리지 ≥80% 달성

**이유**: 임베딩 검증 테스트가 Claude Code API를 사용하는 체인들과 통합되므로, Claude 마이그레이션이 먼저 완료되어야 함.

---

## 3. 기능 요구사항 (Functional Requirements)

### FR-001: HuggingFaceEmbedding 서비스 구현 ⭐ (핵심)

**파일**: `src/services/embeddings.py`

**요구사항**:
- `HuggingFaceEmbedding` 클래스 구현
- `sentence-transformers` SentenceTransformer 사용
- 단일 텍스트 및 배치 텍스트 임베딩 지원
- 한국어 텍스트 검증 기능

**인터페이스**:
```python
class HuggingFaceEmbedding:
    def __init__(self, config: EmbeddingConfiguration):
        """
        임베딩 서비스 초기화

        Args:
            config: EmbeddingConfiguration 엔티티
        """

    def embed_text(self, text: str) -> List[float]:
        """
        단일 텍스트를 임베딩 벡터로 변환

        Args:
            text: 입력 텍스트

        Returns:
            384차원 임베딩 벡터

        Raises:
            ValueError: 빈 텍스트 또는 너무 긴 텍스트
        """

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        여러 텍스트를 배치로 임베딩

        Args:
            texts: 입력 텍스트 리스트

        Returns:
            각 텍스트의 384차원 임베딩 벡터 리스트

        Raises:
            ValueError: 빈 리스트 또는 잘못된 입력
        """

    def get_embedding_dimension(self) -> int:
        """임베딩 차원 반환 (384)"""

    def validate_model(self) -> bool:
        """
        모델 로딩 및 기본 기능 검증

        Returns:
            True if 모델이 정상 작동
        """
```

**검증 기준**:
- 한국어 텍스트 "안녕하세요" → 384차원 벡터 생성
- 배치 크기 100으로 문서 처리 가능
- 빈 텍스트 입력 시 `ValueError` 발생
- 512 토큰 초과 시 자동 truncation

---

### FR-002: ChromaDB 통합 업데이트

**파일**: `src/services/vector_store.py`

**요구사항**:
- `HuggingFaceEmbedding` 인스턴스를 의존성으로 주입
- `add_documents()` 메서드: pre-computed embeddings 사용
- `query()` 메서드: 쿼리 임베딩을 HuggingFaceEmbedding으로 생성
- ChromaDB 기본 임베더 사용 중단

**수정 사항**:
```python
class VectorStore:
    def __init__(
        self,
        config: ChromaDBConfig,
        embedding_service: HuggingFaceEmbedding  # 추가
    ):
        self.embedding_service = embedding_service
        # ...

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        embeddings: Optional[List[List[float]]] = None  # 기존
    ) -> Dict[str, Any]:
        # embeddings가 None이면 embedding_service 사용
        if embeddings is None:
            embeddings = self.embedding_service.embed_texts(documents)
        # ChromaDB에 저장

    def query(
        self,
        query_text: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        # 쿼리 임베딩 생성
        query_embedding = self.embedding_service.embed_text(query_text)
        # ChromaDB 검색
```

**검증 기준**:
- 1000개 문서 추가 시 모두 설정된 모델로 임베딩됨
- 쿼리 검색 시 동일한 모델로 임베딩됨
- ChromaDB 기본 임베더 호출 없음 (로그 확인)

---

### FR-003: 문서 인덱싱 파이프라인

**파일**: `scripts/index_documents.py`

**요구사항**:
- 대량 문서 인덱싱 유틸리티
- JSON, Markdown, PDF 파일 지원
- 배치 처리 (batch_size=100)
- 진행 상황 표시 (progress bar)
- 에러 처리 및 재시도 로직
- 인덱싱 완료 후 통계 출력

**사용 예시**:
```bash
python scripts/index_documents.py \
    --source data/documents/ \
    --format json \
    --batch-size 100
```

**기능**:
1. 문서 파일 읽기 (JSON/MD/PDF)
2. 메타데이터 추출 (파일명, 날짜 등)
3. 배치 단위로 임베딩 생성
4. ChromaDB에 저장
5. 성공/실패 카운트 출력

**검증 기준**:
- 1000개 문서 인덱싱 완료 시간 ≤5분
- 인덱싱 실패율 <1%
- 진행 상황 실시간 표시

---

### FR-004: 한국어 임베딩 품질 검증

**목표**: 한국어 쿼리에 대한 검색 정확도 ≥90% (Top-5)

**검증 방법**:
1. **테스트 데이터셋**:
   - 한국어 쿼리 100개
   - 각 쿼리에 대한 정답 문서 레이블링

2. **평가 지표**:
   - **Top-5 Accuracy**: 상위 5개 결과에 정답이 포함될 확률
   - **Mean Reciprocal Rank (MRR)**: 정답 문서의 평균 순위
   - **Cosine Similarity**: 쿼리와 정답 문서 간 유사도

3. **벤치마크 테스트**:
   - 파일: `tests/benchmarks/test_embedding_accuracy.py`
   - 100 쿼리 × 5 결과 = 500 검색 수행
   - 정확도 계산 및 리포트 생성

**수락 기준**:
- Top-5 Accuracy ≥90%
- MRR ≥0.75
- 쿼리-정답 Cosine Similarity ≥0.7

---

### FR-005: 검색 지연시간 SLA

**목표**: 벡터 검색 응답시간 ≤0.5초 (95th percentile)

**측정 항목**:
- **임베딩 생성 시간**: 쿼리 텍스트 → 벡터 변환
- **벡터 검색 시간**: ChromaDB 유사도 검색
- **총 응답시간**: End-to-End 검색 파이프라인

**테스트 시나리오**:
1. **단일 쿼리 테스트**: 100회 반복 측정
2. **동시성 테스트**: 10 concurrent requests
3. **대량 문서 테스트**: 1000, 5000, 10000 문서 인덱싱 후 검색

**수락 기준**:
- 단일 쿼리 95th percentile ≤0.5초
- 동시 10 요청 시 평균 ≤0.7초
- 10000 문서 인덱싱 후에도 ≤0.5초 유지

---

### FR-006: Top-5 정확도 벤치마크

**파일**: `tests/benchmarks/test_embedding_accuracy.py`

**테스트 데이터셋 구성**:
- **한국어 쿼리**: 50개
- **영어 쿼리**: 30개
- **혼합 쿼리** (한영 혼용): 20개
- **총**: 100개 쿼리

**쿼리 카테고리**:
1. **Factual Questions**: "PostgreSQL에서 트랜잭션이란?"
2. **How-to Questions**: "Python으로 데이터베이스 연결하는 방법"
3. **Conceptual Questions**: "ACID 속성의 의미"
4. **Comparison Questions**: "NoSQL과 SQL의 차이점"

**평가 프로세스**:
1. 각 쿼리 실행 → Top-5 결과 반환
2. 정답 문서 ID와 매칭
3. Hit@5 계산 (정답이 Top-5에 있는지)
4. 전체 정확도 계산

**리포트 생성**:
- 카테고리별 정확도
- 언어별 정확도
- 실패 케이스 분석

**수락 기준**:
- 전체 Top-5 Accuracy ≥90%
- 한국어 쿼리 정확도 ≥90%
- 영어 쿼리 정확도 ≥85%

---

### FR-007: 다국어 지원 검증

**목표**: OpenAI 임베딩 대비 동등 이상의 다국어 지원 확인

**테스트 시나리오**:
1. **한국어 전용 쿼리**: "데이터베이스 인덱스"
2. **영어 전용 쿼리**: "database index"
3. **혼합 쿼리**: "PostgreSQL의 B-tree 인덱스"
4. **특수 문자**: "SQL의 WHERE 조건절"

**평가 지표**:
- 각 언어 조합별 검색 정확도
- 유니코드 처리 정확성
- 특수 문자 처리 정확성

**수락 기준**:
- 한국어 + 영어 혼합 쿼리 정확도 ≥85%
- 유니코드 텍스트 처리 오류 0건
- 특수 문자 인코딩 오류 0건

---

### FR-008: OpenAI 임베딩 완전 제거

**검증 항목**:
1. **코드 검색**:
   - `openai` 패키지 import 참조 0건
   - `OpenAIEmbeddings` 클래스 사용 0건
   - `OPENAI_API_KEY` 환경 변수 참조 0건

2. **설정 파일 확인**:
   - `requirements.txt`에 `openai` 없음
   - `config/settings.py`에 OpenAI 관련 설정 없음
   - `.env.example`에 OPENAI 변수 없음

3. **테스트 확인**:
   - 모든 테스트가 Hugging Face 임베딩만 사용
   - Mock 객체도 OpenAI 제거

**검증 스크립트**:
```bash
# 코드베이스에서 OpenAI 임베딩 참조 검색
grep -r "OpenAIEmbeddings" src/ tests/
grep -r "openai" requirements.txt pyproject.toml
grep -r "OPENAI_API_KEY" config/ .env.example
```

**수락 기준**:
- 위 검색 결과 0건 (문서 제외)
- 모든 테스트 통과 (OpenAI 없이)

---

## 4. 비기능 요구사항 (Non-Functional Requirements)

### NFR-001: 성능 (Performance)

**임베딩 생성 속도**:
- 단일 텍스트: <50ms (CPU)
- 배치 100개: <2초 (CPU)
- 1000개 문서 인덱싱: <5분

**검색 응답시간**:
- 벡터 검색: <0.5초 (95th percentile)
- E2E 검색 파이프라인: <3초

**메모리 사용량**:
- 모델 로딩: <500MB
- 배치 처리: <1GB

---

### NFR-002: 확장성 (Scalability)

**문서 규모**:
- 1000개 문서: 기본 지원
- 10000개 문서: 성능 저하 <10%
- 100000개 문서: 향후 지원 (Phase 외)

**동시성**:
- 10 concurrent queries: 평균 응답시간 <0.7초
- 50 concurrent queries: 평균 응답시간 <1.5초

---

### NFR-003: 안정성 (Reliability)

**에러 처리**:
- 빈 텍스트 입력 → `ValueError`
- 너무 긴 텍스트 → 자동 truncation (512 토큰)
- 모델 로딩 실패 → 명확한 에러 메시지
- ChromaDB 연결 실패 → 재시도 3회

**로깅**:
- 모든 임베딩 생성 로깅 (INFO 레벨)
- 에러 발생 시 상세 스택 트레이스
- 성능 메트릭 로깅 (응답시간, 처리량)

---

### NFR-004: 보안 (Security)

**데이터 검증**:
- 입력 텍스트 길이 제한 (max 512 토큰)
- SQL 인젝션 방지 (메타데이터 검증)
- XSS 방지 (HTML 이스케이프)

**모델 보안**:
- Hugging Face 모델 체크섬 검증
- 로컬 캐싱으로 외부 의존성 최소화

---

### NFR-005: 관찰성 (Observability)

**로깅**:
- 구조화된 로그 (JSON 포맷)
- 요청 ID 추적
- 성능 메트릭 (응답시간, 임베딩 차원)

**모니터링**:
- 임베딩 생성 카운트
- 검색 요청 카운트
- 평균/P95/P99 응답시간
- 에러율

---

### NFR-006: 테스트 커버리지

**목표**: ≥80% 코드 커버리지 유지

**테스트 레벨**:
- **단위 테스트**: `src/services/embeddings.py` 100% 커버리지
- **통합 테스트**: E2E 검색 파이프라인
- **벤치마크 테스트**: 정확도 및 성능

**테스트 파일**:
- `tests/unit/test_embeddings.py`
- `tests/integration/test_vector_search.py`
- `tests/benchmarks/test_embedding_accuracy.py`

---

## 5. 기술 명세 (Technical Specification)

### 5.1 임베딩 모델 상세 사양

**모델 정보**:
- **이름**: `paraphrase-multilingual-MiniLM-L12-v2`
- **소스**: [Hugging Face sentence-transformers](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **아키텍처**: MiniLM (12-layer Transformer)
- **학습 데이터**: Multilingual paraphrase pairs

**스펙**:
- **임베딩 차원**: 384
- **최대 시퀀스 길이**: 512 토큰
- **정규화**: L2 normalized vectors
- **유사도 측정**: Cosine similarity

**지원 언어** (50+ 언어):
- 한국어 (Korean)
- 영어 (English)
- 일본어 (Japanese)
- 중국어 (Chinese)
- 기타 다국어

**성능 특징**:
- CPU에서 빠른 추론 속도
- GPU 불필요
- 메모리 효율적 (<500MB)

---

### 5.2 API 인터페이스

#### HuggingFaceEmbedding 클래스

**파일**: `src/services/embeddings.py`

```python
from typing import List, Optional
from src.models.embedding import EmbeddingConfiguration
from sentence_transformers import SentenceTransformer

class HuggingFaceEmbedding:
    """Hugging Face 임베딩 서비스"""

    def __init__(self, config: EmbeddingConfiguration):
        """
        임베딩 서비스 초기화

        Args:
            config: 임베딩 설정 (모델명, 디바이스, 배치 크기 등)
        """
        self.config = config
        self.model = SentenceTransformer(
            config.model_name,
            device=config.device.value
        )
        self.embedding_dim = config.embedding_dim

    def embed_text(self, text: str) -> List[float]:
        """
        단일 텍스트를 임베딩 벡터로 변환

        Args:
            text: 입력 텍스트

        Returns:
            384차원 임베딩 벡터

        Raises:
            ValueError: 빈 텍스트 입력
        """
        if not text.strip():
            raise ValueError("Empty text cannot be embedded")

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding.tolist()

    def embed_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """
        여러 텍스트를 배치로 임베딩

        Args:
            texts: 입력 텍스트 리스트
            batch_size: 배치 크기 (기본값: config에서 가져옴)

        Returns:
            각 텍스트의 384차원 임베딩 벡터 리스트

        Raises:
            ValueError: 빈 리스트 입력
        """
        if not texts:
            raise ValueError("Empty text list cannot be embedded")

        batch_size = batch_size or self.config.batch_size

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        return embeddings.tolist()

    def get_embedding_dimension(self) -> int:
        """임베딩 차원 반환"""
        return self.embedding_dim

    def validate_model(self) -> bool:
        """
        모델 로딩 및 기본 기능 검증

        Returns:
            True if 모델이 정상 작동
        """
        try:
            test_embedding = self.embed_text("테스트")
            return len(test_embedding) == self.embedding_dim
        except Exception:
            return False
```

---

#### VectorStore 클래스 업데이트

**파일**: `src/services/vector_store.py`

```python
from typing import List, Dict, Any, Optional
from src.models.chroma import ChromaDBConfig
from src.services.embeddings import HuggingFaceEmbedding

class VectorStore:
    """ChromaDB 벡터 스토어"""

    def __init__(
        self,
        config: ChromaDBConfig,
        embedding_service: HuggingFaceEmbedding
    ):
        """
        벡터 스토어 초기화

        Args:
            config: ChromaDB 설정
            embedding_service: 임베딩 서비스
        """
        self.config = config
        self.embedding_service = embedding_service
        # ChromaDB 클라이언트 초기화 ...

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
            성공 여부 및 추가된 문서 수
        """
        # embeddings가 None이면 임베딩 서비스로 생성
        if embeddings is None:
            embeddings = self.embedding_service.embed_texts(documents)

        # ChromaDB에 저장 ...

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
            검색 결과 (documents, metadatas, distances)
        """
        # 쿼리 임베딩 생성
        query_embedding = self.embedding_service.embed_text(query_text)

        # ChromaDB 검색 ...
```

---

### 5.3 데이터 흐름

#### 문서 추가 프로세스

```
1. 사용자가 문서 추가 요청
   ↓
2. VectorStore.add_documents(documents) 호출
   ↓
3. embeddings == None 체크
   ↓
4. HuggingFaceEmbedding.embed_texts(documents)
   - 배치 크기 100으로 처리
   - L2 정규화된 384차원 벡터 생성
   ↓
5. ChromaDB.collection.add(embeddings=..., documents=...)
   ↓
6. 성공 응답 반환
```

#### 쿼리 검색 프로세스

```
1. 사용자가 검색 쿼리 입력
   ↓
2. VectorStore.query(query_text, top_k=5) 호출
   ↓
3. HuggingFaceEmbedding.embed_text(query_text)
   - 단일 쿼리 임베딩 생성
   - L2 정규화된 384차원 벡터
   ↓
4. ChromaDB.collection.query(query_embeddings=...)
   - Cosine similarity 계산
   - Top-K 문서 반환
   ↓
5. 결과 반환 (documents, metadatas, distances)
```

---

### 5.4 설정 관리

**파일**: `config/settings.py`

```python
from src.models.embedding import EmbeddingConfiguration, DeviceType

# 임베딩 설정
EMBEDDING_CONFIG = EmbeddingConfiguration(
    model_name="paraphrase-multilingual-MiniLM-L12-v2",
    embedding_dim=384,
    device=DeviceType.CPU,
    batch_size=100,
    max_sequence_length=512
)
```

**환경 변수** (`.env`):
```bash
# Embedding Model Configuration
EMBEDDING_MODEL_NAME=paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=100

# Vector Store Configuration
CHROMA_PERSIST_DIRECTORY=./data/chroma
CHROMA_COLLECTION_NAME=documents
CHROMA_DISTANCE_FUNCTION=cosine
```

---

## 6. 구현 작업 (Implementation Tasks)

### T041: 모델 설정 검증

**목표**: 기존 임베딩 모델 설정 확인

**작업**:
1. `scripts/download_embedding_model.py` 실행
2. 모델 로딩 확인 (384차원)
3. 한국어 텍스트 임베딩 테스트
4. 설정 파일 검증

**검증**:
- 모델 다운로드 성공
- 임베딩 차원 == 384
- 한국어 임베딩 생성 성공

**예상 시간**: 1시간

---

### T042: 임베딩 서비스 구현 ⭐

**목표**: `HuggingFaceEmbedding` 클래스 구현

**파일**: `src/services/embeddings.py`

**작업**:
1. 클래스 기본 구조 작성
2. `__init__` 메서드 구현
3. `embed_text` 메서드 구현
4. `embed_texts` 메서드 구현 (배치 처리)
5. `get_embedding_dimension` 메서드 구현
6. `validate_model` 메서드 구현
7. 에러 처리 추가
8. 로깅 추가

**검증**:
- 단일 텍스트 임베딩 성공
- 배치 100개 텍스트 임베딩 성공
- 빈 텍스트 입력 시 ValueError 발생
- 한국어 텍스트 처리 성공

**예상 시간**: 4시간

---

### T043: ChromaDB 통합

**목표**: VectorStore와 HuggingFaceEmbedding 연결

**파일**: `src/services/vector_store.py`

**작업**:
1. `__init__`에 `embedding_service` 파라미터 추가
2. `add_documents` 메서드 수정 (pre-computed embeddings 사용)
3. `query` 메서드 수정 (쿼리 임베딩 생성)
4. ChromaDB 기본 임베더 사용 중단
5. 테스트 코드 업데이트

**검증**:
- 문서 추가 시 HuggingFaceEmbedding 사용 확인
- 쿼리 검색 시 동일 모델 사용 확인
- ChromaDB 기본 임베더 호출 없음

**예상 시간**: 3시간

---

### T044: 문서 인덱싱 유틸리티

**목표**: 대량 문서 인덱싱 스크립트 작성

**파일**: `scripts/index_documents.py`

**작업**:
1. CLI 인터페이스 구현 (argparse)
2. 문서 로더 구현 (JSON, MD, PDF)
3. 배치 처리 로직 구현
4. 진행 상황 표시 (tqdm)
5. 에러 처리 및 재시도
6. 통계 출력

**사용 예시**:
```bash
python scripts/index_documents.py \
    --source data/documents/ \
    --format json \
    --batch-size 100
```

**검증**:
- 1000개 문서 인덱싱 성공
- 배치 처리 정상 작동
- 진행률 표시 정상

**예상 시간**: 4시간

---

### T045: 한국어 단위 테스트

**목표**: 임베딩 서비스 단위 테스트 작성

**파일**: `tests/unit/test_embeddings.py`

**작업**:
1. `HuggingFaceEmbedding` 초기화 테스트
2. 단일 텍스트 임베딩 테스트
3. 배치 텍스트 임베딩 테스트
4. 한국어 텍스트 처리 테스트
5. 에러 케이스 테스트 (빈 텍스트 등)
6. 임베딩 차원 검증 테스트

**테스트 케이스**:
- 한국어 텍스트: "안녕하세요", "데이터베이스"
- 영어 텍스트: "Hello", "Database"
- 혼합 텍스트: "PostgreSQL 데이터베이스"
- 빈 텍스트: ""
- 긴 텍스트: 512+ 토큰

**검증**:
- 모든 테스트 통과
- 커버리지 100%

**예상 시간**: 3시간

---

### T046: 벡터 검색 지연시간 테스트

**목표**: 검색 응답시간 SLA 검증

**파일**: `tests/integration/test_vector_search.py`

**작업**:
1. E2E 검색 파이프라인 테스트
2. 단일 쿼리 응답시간 측정 (100회 반복)
3. 동시성 테스트 (10 concurrent)
4. 대량 문서 테스트 (1000, 5000, 10000)
5. P95, P99 응답시간 계산
6. SLA 검증 (≤0.5초)

**테스트 시나리오**:
- 1000개 문서 인덱싱 후 100 쿼리 실행
- 각 쿼리 응답시간 측정
- P95 ≤0.5초 확인

**검증**:
- P95 응답시간 ≤0.5초
- 동시 10 요청 평균 ≤0.7초

**예상 시간**: 4시간

---

### T047: Top-5 정확도 벤치마크

**목표**: 검색 정확도 ≥90% 검증

**파일**: `tests/benchmarks/test_embedding_accuracy.py`

**작업**:
1. 테스트 데이터셋 준비 (100 쿼리)
2. 각 쿼리에 대한 정답 문서 레이블링
3. 벤치마크 실행 (100 쿼리 × Top-5)
4. Hit@5 계산
5. MRR 계산
6. 카테고리별/언어별 정확도 분석
7. 리포트 생성

**테스트 데이터**:
- 한국어 쿼리: 50개
- 영어 쿼리: 30개
- 혼합 쿼리: 20개

**검증**:
- 전체 Top-5 Accuracy ≥90%
- 한국어 정확도 ≥90%
- MRR ≥0.75

**예상 시간**: 6시간

---

### T048: 다국어 지원 테스트

**목표**: 다국어 및 특수 문자 처리 검증

**파일**: `tests/unit/test_embeddings.py` (추가)

**작업**:
1. 한국어 전용 쿼리 테스트
2. 영어 전용 쿼리 테스트
3. 한영 혼합 쿼리 테스트
4. 유니코드 텍스트 테스트
5. 특수 문자 테스트
6. 인코딩 오류 확인

**테스트 케이스**:
- 한국어: "데이터베이스 인덱스"
- 영어: "database index"
- 혼합: "PostgreSQL의 B-tree 인덱스"
- 특수문자: "SQL의 WHERE 조건절 (condition)"
- 유니코드: "한글, 日本語, 中文"

**검증**:
- 모든 언어 조합 정상 처리
- 인코딩 오류 0건
- 특수 문자 처리 성공

**예상 시간**: 3시간

---

### T049: 문서화

**목표**: 임베딩 모델 및 API 문서 작성

**파일**: `docs/embedding-model.md`

**작업**:
1. 모델 사양 문서화
2. API 사용 가이드 작성
3. 성능 벤치마크 결과 정리
4. 트러블슈팅 가이드 작성
5. FAQ 작성

**내용**:
- 모델 선정 이유
- 임베딩 서비스 사용법
- 문서 인덱싱 방법
- 성능 최적화 팁
- 일반적인 문제 해결

**검증**:
- 문서 완성도
- 코드 예시 정확성

**예상 시간**: 4시간

---

## 7. 테스트 전략 (Testing Strategy)

### 7.1 테스트 레벨

**단위 테스트 (Unit Tests)**:
- 파일: `tests/unit/test_embeddings.py`
- 대상: `HuggingFaceEmbedding` 클래스 메서드
- 커버리지 목표: 100%
- 테스트 수: 15개 이상

**통합 테스트 (Integration Tests)**:
- 파일: `tests/integration/test_vector_search.py`
- 대상: E2E 검색 파이프라인
- 시나리오: 문서 추가 → 검색 → 결과 검증
- 테스트 수: 10개 이상

**벤치마크 테스트 (Benchmark Tests)**:
- 파일: `tests/benchmarks/test_embedding_accuracy.py`
- 대상: 검색 정확도 및 성능
- 데이터: 100 쿼리 테스트셋
- 실행 시간: ~10분

---

### 7.2 테스트 데이터

**한국어 쿼리 예시**:
1. "PostgreSQL에서 트랜잭션 격리 수준이란?"
2. "데이터베이스 인덱스의 종류"
3. "SQL 쿼리 최적화 방법"
4. "NoSQL과 관계형 데이터베이스 차이"
5. "Python으로 데이터베이스 연결하는 방법"

**영어 쿼리 예시**:
1. "What is database transaction?"
2. "Types of database indexes"
3. "SQL query optimization techniques"
4. "Difference between NoSQL and SQL"
5. "How to connect database in Python"

**혼합 쿼리 예시**:
1. "PostgreSQL의 ACID 속성"
2. "MongoDB에서 aggregate 함수 사용법"
3. "Python pandas로 데이터 분석"

---

### 7.3 테스트 자동화

**CI/CD 통합**:
```yaml
# .github/workflows/test.yml
- name: Run embedding tests
  run: |
    pytest tests/unit/test_embeddings.py
    pytest tests/integration/test_vector_search.py

- name: Run benchmarks (weekly)
  run: pytest tests/benchmarks/test_embedding_accuracy.py
  schedule: cron('0 0 * * 0')  # Every Sunday
```

**로컬 테스트 스크립트**:
```bash
# 전체 테스트 실행
./scripts/run_tests.sh

# 임베딩 테스트만 실행
pytest tests/unit/test_embeddings.py -v

# 벤치마크 실행
pytest tests/benchmarks/ -v --benchmark-only
```

---

## 8. 수락 기준 (Acceptance Criteria)

### AC-001: 1000개 문서 인덱싱 성공

**시나리오**:
1. `scripts/index_documents.py` 실행
2. 1000개 JSON 문서 로딩
3. 배치 크기 100으로 임베딩 생성
4. ChromaDB에 저장

**수락 조건**:
- ✅ 인덱싱 완료 시간 ≤5분
- ✅ 실패율 <1%
- ✅ 모든 문서가 설정된 모델로 임베딩됨
- ✅ ChromaDB에 1000개 문서 저장 확인

---

### AC-002: 한국어 쿼리 Top-5 정확도 ≥90%

**시나리오**:
1. 50개 한국어 쿼리 준비
2. 각 쿼리에 대한 정답 문서 레이블링
3. 벤치마크 실행 (Top-5 검색)
4. Hit@5 계산

**수락 조건**:
- ✅ Top-5 Accuracy ≥90% (50 쿼리 중 45개 이상 정답)
- ✅ MRR ≥0.75
- ✅ 쿼리-정답 Cosine Similarity ≥0.7

---

### AC-003: 검색 지연시간 ≤0.5초 (95th percentile)

**시나리오**:
1. 1000개 문서 인덱싱
2. 100 쿼리 실행
3. 각 쿼리 응답시간 측정
4. P95 계산

**수락 조건**:
- ✅ P95 응답시간 ≤0.5초
- ✅ 평균 응답시간 ≤0.3초
- ✅ P99 응답시간 ≤0.8초

---

### AC-004: OpenAI 임베딩 참조 0건

**시나리오**:
1. 코드베이스에서 OpenAI 임베딩 검색
2. 설정 파일에서 OpenAI 키 검색
3. 테스트 코드 검증

**수락 조건**:
- ✅ `grep -r "OpenAIEmbeddings" src/ tests/` → 0건
- ✅ `grep -r "OPENAI_API_KEY" config/ .env.example` → 0건
- ✅ 모든 테스트 통과 (OpenAI 없이)

---

### AC-005: 테스트 커버리지 ≥80%

**시나리오**:
1. `pytest --cov=src/services/embeddings --cov-report=html` 실행
2. 커버리지 리포트 확인

**수락 조건**:
- ✅ `src/services/embeddings.py` 커버리지 100%
- ✅ 전체 프로젝트 커버리지 ≥80%
- ✅ 모든 단위/통합 테스트 통과

---

## 9. 종속성 및 제약사항

### 9.1 선행 조건

**Phase 3 완료 필수**:
- ✅ Claude Code API 통합 완료 (User Story 1)
- ✅ 모든 체인 Claude로 마이그레이션
- ✅ 테스트 커버리지 ≥80%
- ✅ OpenAI GPT-4o 제거 완료

**이유**: 임베딩 검증 테스트가 Claude Code API를 사용하는 체인들과 통합되므로.

---

### 9.2 기술적 제약사항

**하드웨어 제약**:
- CPU 전용 환경 (GPU 없음)
- 메모리 제한: <1GB (임베딩 모델 + 배치 처리)

**소프트웨어 제약**:
- Python 3.10+ 필수
- sentence-transformers>=2.2.0
- chromadb>=0.4.0

**성능 제약**:
- CPU에서 배치 100개 처리 시간 ≤2초
- 모델 로딩 시간 ≤5초

---

### 9.3 예산 제약

**비용 목표**: $0 (완전 무료 오픈소스)

**모델 선택**:
- Hugging Face 오픈소스 모델만 사용
- API 비용 없음
- 로컬 추론 (외부 API 호출 없음)

---

## 10. 마일스톤 및 타임라인

### Week 1: 서비스 구현 (T041-T044)

**목표**: 임베딩 서비스 및 ChromaDB 통합 완료

**작업**:
- Day 1-2: T041 모델 설정 검증 + T042 임베딩 서비스 구현
- Day 3-4: T043 ChromaDB 통합
- Day 5: T044 문서 인덱싱 유틸리티

**마일스톤**: 1000개 문서 인덱싱 성공

---

### Week 2: 테스트 및 검증 (T045-T048)

**목표**: 품질 검증 및 SLA 확인

**작업**:
- Day 1-2: T045 한국어 단위 테스트
- Day 3: T046 벡터 검색 지연시간 테스트
- Day 4-5: T047 Top-5 정확도 벤치마크
- Day 6: T048 다국어 지원 테스트

**마일스톤**: 모든 수락 기준 통과

---

### Week 3: 문서화 및 최적화 (T049)

**목표**: 문서 작성 및 성능 최적화

**작업**:
- Day 1-2: T049 문서화
- Day 3-4: 성능 최적화 (필요 시)
- Day 5: 최종 검증 및 배포 준비

**마일스톤**: Phase 4 완료

---

## 11. 리스크 및 완화 전략

### 리스크 1: 한국어 임베딩 품질 미달

**리스크**: Top-5 정확도 <90%

**완화 전략**:
- 다양한 한국어 쿼리로 사전 테스트
- 필요 시 fine-tuning 검토 (Phase 외)
- 하이브리드 검색 (BM25 + 벡터) 활용

**확률**: 낮음 (모델은 이미 한국어 지원 검증됨)

---

### 리스크 2: 검색 지연시간 SLA 미달

**리스크**: P95 응답시간 >0.5초

**완화 전략**:
- 배치 크기 최적화
- ChromaDB 인덱스 튜닝
- 캐싱 전략 도입 (자주 검색되는 쿼리)

**확률**: 중간

---

### 리스크 3: 메모리 부족

**리스크**: 대량 문서 처리 시 메모리 부족

**완화 전략**:
- 배치 크기 동적 조정
- 문서 스트리밍 처리
- 가비지 컬렉션 명시적 호출

**확률**: 낮음

---

## 12. 성공 지표 (Success Metrics)

### 기능적 지표

- ✅ Top-5 검색 정확도 ≥90%
- ✅ 검색 지연시간 P95 ≤0.5초
- ✅ 1000개 문서 인덱싱 성공률 ≥99%
- ✅ OpenAI 임베딩 참조 0건

### 품질 지표

- ✅ 테스트 커버리지 ≥80%
- ✅ 단위 테스트 통과율 100%
- ✅ 통합 테스트 통과율 100%
- ✅ 코드 리뷰 승인

### 비용 지표

- ✅ 임베딩 비용 $0 (완전 무료)
- ✅ 월 예산 절감 (OpenAI API 비용 제거)

---

## 13. 후속 작업 (Future Work)

### Phase 5 이후 개선 사항

**성능 최적화**:
- GPU 지원 추가 (선택적)
- 모델 양자화 (INT8)
- 캐싱 레이어 추가

**품질 개선**:
- Fine-tuning (도메인 특화)
- 하이브리드 검색 개선
- Reranker 모델 추가

**확장성**:
- 분산 벡터 스토어 (Qdrant, Milvus)
- 100K+ 문서 지원

---

## 14. 참고 자료 (References)

### 기술 문서

- [Hugging Face Model Card](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- [sentence-transformers Documentation](https://www.sbert.net/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [LangChain Embeddings Guide](https://python.langchain.com/docs/modules/data_connection/text_embedding/)

### 프로젝트 문서

- `specs/001-claude-huggingface-migration/spec.md` - Claude 마이그레이션 (User Story 1)
- `README.md` - 프로젝트 개요
- `tasks.md` - 전체 작업 목록
- `plan.md` - 구현 계획

---

**Document Version**: 1.0.0
**Last Updated**: 2025-01-17
**Author**: AI-Tool Development Team
**Status**: Planning → Implementation Ready

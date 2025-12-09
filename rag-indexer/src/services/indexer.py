"""벡터 데이터베이스 인덱싱 서비스.

인덱싱 파이프라인을 관리합니다: 청킹 → 임베딩 → 벡터 DB 저장.

주요 기능:
    - 단일/다중 문서 인덱싱
    - 계층적 청킹 지원 (Phase 1.3)
    - AI 메타데이터 추출 통합 (Phase 1.2)
    - 다양한 필터 옵션의 검색 기능
"""

from datetime import datetime
from typing import Optional

from shared import VectorStore, get_vector_store

from ..models import Chunk, ChunkType, Document, Source
from ..storage import Storage, get_storage
from .ai_extractor import AIExtractor, ExtractedMetadata, get_ai_extractor
from .chunker import Chunker, HierarchicalChunker, get_chunker, get_hierarchical_chunker
from .embedder import Embedder, get_embedder


class Indexer:
    """벡터 데이터베이스 인덱싱 서비스.

    전체 인덱싱 파이프라인을 오케스트레이션합니다:
    1. 문서를 작은 청크로 분할 (표준 또는 계층적)
    2. 청크에 대한 임베딩 생성
    3. Qdrant에 벡터 저장
    4. 문서 메타데이터 업데이트

    Phase 1.3 계층적 청킹 지원:
        - 부모 청크 (~4000자): 포괄적 컨텍스트를 위한 큰 청크
        - 자식 청크 (~800자): 정밀 검색을 위한 작은 청크
        - 자식 청크는 parent_id를 통해 부모를 참조

    Attributes:
        DEFAULT_COLLECTION: 기본 Qdrant 컬렉션 이름.
        collection_name: 사용 중인 컬렉션 이름.
    """

    DEFAULT_COLLECTION = "rag_documents"

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedder: Optional[Embedder] = None,
        chunker: Optional[Chunker] = None,
        hierarchical_chunker: Optional[HierarchicalChunker] = None,
        storage: Optional[Storage] = None,
        ai_extractor: Optional[AIExtractor] = None,
        collection_name: str = DEFAULT_COLLECTION,
        enable_ai_extraction: bool = False,
        enable_hierarchical: bool = False,
    ):
        """인덱서를 초기화합니다.

        Args:
            vector_store: VectorStore 인스턴스.
                None이면 기본 인스턴스를 지연 로딩합니다.
            embedder: Embedder 인스턴스.
                None이면 기본 인스턴스를 지연 로딩합니다.
            chunker: Chunker 인스턴스.
                None이면 기본 인스턴스를 지연 로딩합니다.
            hierarchical_chunker: 부모-자식 청킹을 위한 HierarchicalChunker 인스턴스.
            storage: Storage 인스턴스.
                None이면 기본 인스턴스를 지연 로딩합니다.
            ai_extractor: 메타데이터 추출을 위한 AIExtractor 인스턴스.
            collection_name: Qdrant 컬렉션 이름.
            enable_ai_extraction: AI 메타데이터 추출 사용 여부.
            enable_hierarchical: 계층적 부모-자식 청킹 사용 여부.
        """
        self._vector_store = vector_store
        self._embedder = embedder
        self._chunker = chunker
        self._hierarchical_chunker = hierarchical_chunker
        self._storage = storage
        self._ai_extractor = ai_extractor
        self.collection_name = collection_name
        self._collection_initialized = False
        self._enable_ai_extraction = enable_ai_extraction
        self._enable_hierarchical = enable_hierarchical

    @property
    def vector_store(self) -> VectorStore:
        """벡터 스토어를 지연 로딩합니다.

        Returns:
            VectorStore 인스턴스.
        """
        if self._vector_store is None:
            self._vector_store = get_vector_store()
        return self._vector_store

    @property
    def embedder(self) -> Embedder:
        """임베더를 지연 로딩합니다.

        Returns:
            Embedder 인스턴스.
        """
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    @property
    def chunker(self) -> Chunker:
        """청커를 지연 로딩합니다.

        Returns:
            Chunker 인스턴스.
        """
        if self._chunker is None:
            self._chunker = get_chunker()
        return self._chunker

    @property
    def hierarchical_chunker(self) -> HierarchicalChunker:
        """계층적 청커를 지연 로딩합니다.

        Returns:
            HierarchicalChunker 인스턴스.
        """
        if self._hierarchical_chunker is None:
            self._hierarchical_chunker = get_hierarchical_chunker()
        return self._hierarchical_chunker

    @property
    def storage(self) -> Storage:
        """스토리지를 지연 로딩합니다.

        Returns:
            Storage 인스턴스.
        """
        if self._storage is None:
            self._storage = get_storage()
        return self._storage

    @property
    def ai_extractor(self) -> Optional[AIExtractor]:
        """AI 추출기를 지연 로딩합니다 (활성화된 경우).

        AI 추출이 비활성화되어 있으면 None을 반환합니다.

        Returns:
            AIExtractor 인스턴스 또는 None.
        """
        if not self._enable_ai_extraction:
            return None
        if self._ai_extractor is None:
            self._ai_extractor = get_ai_extractor()
        return self._ai_extractor

    def _extract_ai_metadata(self, document: Document) -> dict:
        """문서에서 AI 메타데이터를 추출합니다 (활성화된 경우).

        Claude API를 사용하여 문서 내용에서 구조화된 메타데이터를
        추출합니다. 실패 시에도 인덱싱이 중단되지 않습니다.

        Args:
            document: 메타데이터를 추출할 문서.

        Returns:
            AI 추출 메타데이터 필드가 담긴 딕셔너리.
            실패 시 빈 딕셔너리 반환.
        """
        if not self.ai_extractor:
            return {}

        try:
            extracted = self.ai_extractor.extract(
                title=document.title,
                content=document.content,
            )
            return {
                "ai_content_type": extracted.content_type,
                "ai_topics": extracted.topics,
                "ai_difficulty": extracted.difficulty,
                "ai_has_code_samples": extracted.has_code_samples,
                "ai_key_entities": extracted.key_entities,
                "ai_summary": extracted.summary,
            }
        except Exception:
            # 실패 시 빈 딕셔너리 반환 - 인덱싱을 막지 않음
            return {}

    def ensure_collection(self) -> None:
        """벡터 컬렉션이 존재하는지 확인합니다.

        컬렉션이 없으면 임베딩 차원에 맞게 생성합니다.
        이미 초기화되었으면 아무 작업도 하지 않습니다.
        """
        if not self._collection_initialized:
            self.vector_store.ensure_collection(
                dimension=self.embedder.dimension,
            )
            self._collection_initialized = True

    def index_document(
        self,
        document: Document,
        source: Source,
        show_progress: bool = False,
        hierarchical: Optional[bool] = None,
    ) -> int:
        """단일 문서를 인덱싱합니다.

        문서를 청크로 분할하고, 임베딩을 생성한 후,
        벡터 DB에 저장합니다.

        인덱싱 과정:
            1. 기존 청크 삭제 (중복 방지)
            2. 문서 청킹 (표준 또는 계층적)
            3. 임베딩 생성
            4. AI 메타데이터 추출 (선택적)
            5. Qdrant에 업서트
            6. 문서 메타데이터 업데이트

        Args:
            document: 인덱싱할 문서.
            source: 부모 소스 정보.
            show_progress: 진행률 표시 여부.
            hierarchical: 계층적 청킹 사용 여부.
                None이면 인스턴스 설정을 따릅니다.

        Returns:
            인덱싱된 청크 수.
        """
        self.ensure_collection()

        # 이 문서의 기존 청크 삭제
        self.delete_document_chunks(document.id)

        # 청킹 모드 결정
        use_hierarchical = hierarchical if hierarchical is not None else self._enable_hierarchical

        # 문서 청킹 (표준 또는 계층적)
        if use_hierarchical:
            chunks = self.hierarchical_chunker.chunk_document(document, include_parents=True)
        else:
            chunks = self.chunker.chunk_document(document)

        if not chunks:
            return 0

        # 임베딩 생성
        self.embedder.embed_chunks(chunks, show_progress=show_progress)

        # AI 메타데이터 추출 (활성화된 경우)
        ai_metadata = self._extract_ai_metadata(document)

        # AI 메타데이터를 문서 메타데이터와 병합
        combined_metadata = {**(document.metadata or {}), **ai_metadata}

        # 향상된 메타데이터와 함께 Qdrant 포인트로 변환
        points = [
            chunk.to_qdrant_point(
                source_id=source.id,
                source_type=source.source_type.value,
                title=document.title,
                url=document.url,
                document_metadata=combined_metadata,
            )
            for chunk in chunks
        ]

        # 벡터 스토어에 업서트
        self.vector_store.upsert(points=points)

        # 문서 메타데이터 업데이트
        document.mark_indexed()
        self.storage.upsert_document(document)

        return len(chunks)

    def index_documents(
        self,
        documents: list[Document],
        source: Source,
        show_progress: bool = True,
    ) -> dict[str, int]:
        """여러 문서를 인덱싱합니다.

        각 문서에 대해 index_document를 호출하고
        통계를 수집합니다.

        Args:
            documents: 인덱싱할 문서 리스트.
            source: 부모 소스 정보.
            show_progress: 진행률 표시 여부.

        Returns:
            처리 통계가 담긴 딕셔너리.
            - documents_processed: 처리된 문서 수
            - chunks_created: 생성된 청크 수
            - errors: 오류 수
        """
        self.ensure_collection()

        stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "errors": 0,
        }

        for doc in documents:
            try:
                chunks_count = self.index_document(
                    document=doc,
                    source=source,
                    show_progress=show_progress,
                )
                stats["documents_processed"] += 1
                stats["chunks_created"] += chunks_count
            except Exception:
                stats["errors"] += 1
                raise

        return stats

    def delete_document_chunks(self, document_id: str) -> bool:
        """문서의 모든 청크를 삭제합니다.

        재인덱싱 전 기존 청크를 정리하거나,
        문서 삭제 시 관련 청크를 제거할 때 사용합니다.

        Args:
            document_id: 문서 ID.

        Returns:
            성공 시 True.
        """
        return self.vector_store.delete_by_filter(
            field="document_id",
            value=document_id,
        )

    def delete_source_chunks(self, source_id: str) -> bool:
        """소스의 모든 청크를 삭제합니다.

        전체 소스를 재동기화하거나 삭제할 때 사용합니다.

        Args:
            source_id: 소스 ID.

        Returns:
            성공 시 True.
        """
        return self.vector_store.delete_by_filter(
            field="source_id",
            value=source_id,
        )

    def search(
        self,
        query: str,
        limit: int = 10,
        source_id: Optional[str] = None,
        source_type: Optional[str] = None,
        score_threshold: Optional[float] = None,
        language: Optional[str] = None,
        content_type: Optional[str] = None,
        http_method: Optional[str] = None,
        chunk_type: Optional[str] = None,
    ) -> list[dict]:
        """유사한 청크를 검색합니다 (향상된 필터링 지원).

        쿼리 텍스트를 임베딩하고 벡터 유사도 검색을 수행합니다.
        다양한 메타데이터 필터로 결과를 좁힐 수 있습니다.

        Args:
            query: 검색 쿼리 텍스트.
            limit: 최대 결과 수.
            source_id: 특정 소스로 필터링.
            source_type: 소스 유형으로 필터링 (notion/swagger).
            score_threshold: 최소 유사도 점수.
            language: 언어로 필터링 (ko/en).
            content_type: 콘텐츠 유형으로 필터링 (api_doc/guide/faq).
            http_method: HTTP 메서드로 필터링 (GET/POST/PUT/DELETE).
            chunk_type: 청크 유형으로 필터링 (standard/parent/child).

        Returns:
            점수가 포함된 검색 결과 리스트.
            각 결과는 id, score, payload를 포함합니다.
        """
        self.ensure_collection()

        # 쿼리 임베딩 생성
        query_vector = self.embedder.embed_query(query)

        # 향상된 메타데이터 지원으로 필터 구성
        filter_conditions = {}
        if source_id:
            filter_conditions["source_id"] = source_id
        if source_type:
            filter_conditions["source_type"] = source_type
        # Phase 1.1: 향상된 메타데이터 필터
        if language:
            filter_conditions["language"] = language
        if content_type:
            filter_conditions["content_type"] = content_type
        if http_method:
            filter_conditions["http_method"] = http_method
        # Phase 1.3: 계층적 청크 유형 필터
        if chunk_type:
            filter_conditions["chunk_type"] = chunk_type

        # 검색 수행
        results = self.vector_store.search(
            query_vector=query_vector,
            limit=limit,
            filter_conditions=filter_conditions if filter_conditions else None,
            score_threshold=score_threshold,
        )

        return results

    def search_with_parent_context(
        self,
        query: str,
        limit: int = 10,
        source_id: Optional[str] = None,
        source_type: Optional[str] = None,
        score_threshold: Optional[float] = None,
    ) -> list[dict]:
        """자식 청크를 검색하고 부모 컨텍스트를 조회합니다.

        Phase 1.3: 계층적 검색 전략.
        - 정밀도를 위해 작은 자식 청크에서 검색
        - 풍부한 응답을 위해 부모 컨텍스트 반환

        이 전략의 장점:
            - 자식 청크는 정확한 매칭에 최적화
            - 부모 청크는 응답 생성에 충분한 컨텍스트 제공

        Args:
            query: 검색 쿼리 텍스트.
            limit: 최대 결과 수.
            source_id: 특정 소스로 필터링.
            source_type: 소스 유형으로 필터링.
            score_threshold: 최소 유사도 점수.

        Returns:
            부모 컨텍스트가 추가된 검색 결과 리스트.
            각 결과에 parent_context 필드가 추가됩니다.
        """
        # 정밀도를 위해 자식 청크만 검색
        results = self.search(
            query=query,
            limit=limit,
            source_id=source_id,
            source_type=source_type,
            score_threshold=score_threshold,
            chunk_type=ChunkType.CHILD.value,
        )

        # 부모 컨텍스트로 결과 강화
        enriched_results = []
        for result in results:
            payload = result.get("payload", {})
            parent_id = payload.get("parent_id")

            # parent_id가 있으면 부모 청크 조회 시도
            parent_context = None
            if parent_id:
                parent_point = self.vector_store.get_point(parent_id)
                if parent_point:
                    parent_context = parent_point.get("payload", {}).get("text")

            enriched_results.append({
                **result,
                "parent_context": parent_context,
            })

        return enriched_results

    def get_parent_chunk(self, parent_id: str) -> Optional[dict]:
        """ID로 부모 청크를 조회합니다.

        Args:
            parent_id: 부모 청크 ID.

        Returns:
            부모 청크 데이터 또는 찾을 수 없으면 None.
        """
        self.ensure_collection()
        return self.vector_store.get_point(parent_id)

    def get_collection_stats(self) -> dict:
        """컬렉션 통계를 조회합니다 (계층적 카운트 포함).

        전체 청크 수와 유형별 청크 수를 반환합니다.

        Returns:
            컬렉션 통계 딕셔너리.
            - collection_name: 컬렉션 이름
            - total_chunks: 전체 청크 수
            - standard_chunks: 표준 청크 수
            - parent_chunks: 부모 청크 수
            - child_chunks: 자식 청크 수
        """
        self.ensure_collection()
        total_count = self.vector_store.count()

        # 계층적 통계를 위한 청크 유형별 카운트
        stats = {
            "collection_name": self.collection_name,
            "total_chunks": total_count,
        }

        # 계층적 청킹 사용 시 청크 유형별 카운트 조회
        for chunk_type in [ChunkType.STANDARD, ChunkType.PARENT, ChunkType.CHILD]:
            type_count = self.vector_store.count(
                filter_conditions={"chunk_type": chunk_type.value},
            )
            stats[f"{chunk_type.value}_chunks"] = type_count

        return stats


# 모듈 레벨 싱글톤 인스턴스
_indexer: Optional[Indexer] = None


def get_indexer(collection_name: str = Indexer.DEFAULT_COLLECTION) -> Indexer:
    """인덱서 인스턴스를 가져옵니다.

    싱글톤 패턴을 사용하여 동일한 컬렉션에 대해
    하나의 인스턴스만 생성합니다.

    Args:
        collection_name: Qdrant 컬렉션 이름.

    Returns:
        Indexer 인스턴스.

    Note:
        첫 호출 시의 컬렉션 이름이 이후 호출에서도 유지됩니다.
        다른 컬렉션이 필요하면 Indexer 클래스를 직접 인스턴스화하세요.
    """
    global _indexer

    if _indexer is None:
        _indexer = Indexer(collection_name=collection_name)

    return _indexer

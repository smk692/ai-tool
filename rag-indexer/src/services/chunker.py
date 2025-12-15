"""텍스트 청킹 서비스.

문서를 임베딩 및 인덱싱에 적합한 작은 청크로 분할합니다.
RecursiveCharacterTextSplitter를 사용하여 의미론적 청킹을 수행합니다.

주요 기능:
    - 표준 청킹: 고정 크기 오버랩 청크 생성
    - 계층적 청킹 (Phase 1.3): 부모-자식 관계의 청크 계층 구조
    - 한국어 인식 분리자: 한글 문장 경계 고려
"""

import hashlib
from uuid import UUID

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..models import Chunk, ChunkType, Document


class Chunker:
    """텍스트 청킹 서비스.

    문서를 임베딩 및 벡터 검색에 적합한
    오버랩 청크로 분할합니다.

    특징:
        - RecursiveCharacterTextSplitter 기반
        - 한국어/일본어 문장 경계 인식
        - 결정론적 청크 ID 생성 (멱등 upsert 지원)

    Attributes:
        chunk_size: 청크당 최대 문자 수.
        chunk_overlap: 청크 간 오버랩 문자 수.
        separators: 텍스트 분할에 사용할 구분자 목록.
    """

    # 한국어 인식 분리자 - 더 나은 청크 경계를 위해
    DEFAULT_SEPARATORS = [
        "\n\n",  # 문단
        "\n",  # 줄
        "。",  # 일본어/한국어 마침표
        ".",  # 영어 마침표
        "!",
        "?",
        "！",
        "？",
        ";",
        "；",
        ",",
        "，",
        " ",  # 공백
        "",  # 문자 (폴백)
    ]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ):
        """청커를 초기화합니다.

        Args:
            chunk_size: 청크당 최대 문자 수 (기본값: 1000).
                너무 크면 검색 정밀도가 떨어지고,
                너무 작으면 문맥이 손실됩니다.
            chunk_overlap: 청크 간 오버랩 문자 수 (기본값: 200).
                문맥 연속성을 위해 인접 청크와 겹치는 부분.
            separators: 분할에 사용할 커스텀 구분자 목록.
                None이면 DEFAULT_SEPARATORS 사용.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk_text(self, text: str) -> list[str]:
        """텍스트를 청크로 분할합니다.

        Args:
            text: 분할할 텍스트.

        Returns:
            텍스트 청크 리스트.
            빈 텍스트의 경우 빈 리스트 반환.
        """
        if not text or not text.strip():
            return []

        return self._splitter.split_text(text)

    def chunk_document(self, document: Document) -> list[Chunk]:
        """문서를 Chunk 객체 리스트로 분할합니다.

        문서 내용을 청크로 분할하고, 각 청크에
        메타데이터와 결정론적 ID를 부여합니다.

        Args:
            document: 분할할 문서.

        Returns:
            메타데이터가 포함된 Chunk 객체 리스트.
        """
        text_chunks = self.chunk_text(document.content)

        chunks = []
        for idx, text in enumerate(text_chunks):
            chunk = Chunk(
                id=self._generate_chunk_id(document.id, idx),
                document_id=document.id,
                chunk_index=idx,
                text=text,
                metadata={
                    "source_id": document.source_id,
                    "document_title": document.title,
                    "document_url": document.url,
                    "total_chunks": len(text_chunks),
                },
            )
            chunk.estimate_tokens()
            chunks.append(chunk)

        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """여러 문서를 청크로 분할합니다.

        Args:
            documents: 분할할 문서 리스트.

        Returns:
            모든 Chunk 객체 리스트.
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks

    def _generate_chunk_id(self, document_id: str, chunk_index: int) -> str:
        """결정론적 청크 ID를 생성합니다.

        결정론적 ID를 사용하면 동일한 문서를
        재인덱싱할 때 멱등(idempotent) upsert가 가능합니다.

        Args:
            document_id: 부모 문서 ID.
            chunk_index: 문서 내 청크 위치.

        Returns:
            결정론적 UUID 문자열.
        """
        # 문서와 위치 기반으로 결정론적 ID 생성
        content = f"{document_id}:{chunk_index}"
        hash_bytes = hashlib.sha256(content.encode()).digest()[:16]
        return str(UUID(bytes=hash_bytes))


class HierarchicalChunker:
    """계층적 청킹 서비스 (Phase 1.3).

    부모-자식 청크 관계를 생성합니다:
        - 부모 청크 (~4000자): 포괄적인 컨텍스트를 위한 대형 청크
        - 자식 청크 (~800자): 정밀 검색을 위한 소형 청크

    장점:
        - 작은 자식 청크로 검색하여 정밀도 향상
        - 부모 컨텍스트를 반환하여 풍부한 응답 생성

    사용 사례:
        - RAG 응답 품질 개선
        - 검색 정밀도와 컨텍스트 풍부함의 균형
        - 긴 문서에서의 관련 정보 추출 최적화

    Attributes:
        parent_chunk_size: 부모 청크당 최대 문자 수.
        parent_chunk_overlap: 부모 청크 간 오버랩 문자 수.
        child_chunk_size: 자식 청크당 최대 문자 수.
        child_chunk_overlap: 자식 청크 간 오버랩 문자 수.
        separators: 텍스트 분할에 사용할 구분자 목록.
    """

    DEFAULT_PARENT_SIZE = 4000
    DEFAULT_PARENT_OVERLAP = 400
    DEFAULT_CHILD_SIZE = 800
    DEFAULT_CHILD_OVERLAP = 100

    def __init__(
        self,
        parent_chunk_size: int = DEFAULT_PARENT_SIZE,
        parent_chunk_overlap: int = DEFAULT_PARENT_OVERLAP,
        child_chunk_size: int = DEFAULT_CHILD_SIZE,
        child_chunk_overlap: int = DEFAULT_CHILD_OVERLAP,
        separators: list[str] | None = None,
    ):
        """계층적 청커를 초기화합니다.

        Args:
            parent_chunk_size: 부모 청크당 최대 문자 수 (기본값: 4000).
                포괄적인 컨텍스트를 제공할 수 있는 크기.
            parent_chunk_overlap: 부모 청크 간 오버랩 문자 수 (기본값: 400).
            child_chunk_size: 자식 청크당 최대 문자 수 (기본값: 800).
                정밀 검색에 적합한 크기.
            child_chunk_overlap: 자식 청크 간 오버랩 문자 수 (기본값: 100).
            separators: 분할에 사용할 커스텀 구분자 목록.
        """
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap
        self.separators = separators or Chunker.DEFAULT_SEPARATORS

        # 부모 청커 - 대형 청크
        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )

        # 자식 청커 - 소형 청크
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk_document(
        self,
        document: Document,
        include_parents: bool = True,
    ) -> list[Chunk]:
        """문서를 계층적 Chunk 객체로 분할합니다.

        먼저 문서를 부모 청크로 분할하고,
        각 부모 청크를 다시 자식 청크로 분할합니다.
        자식 청크는 parent_id를 통해 부모를 참조합니다.

        Args:
            document: 분할할 문서.
            include_parents: 출력에 부모 청크 포함 여부.
                False이면 자식 청크만 반환하지만,
                여전히 parent_id를 통해 부모를 참조합니다.

        Returns:
            Chunk 객체 리스트 (부모 및/또는 자식).
        """
        if not document.content or not document.content.strip():
            return []

        # 먼저 부모 청크 생성
        parent_texts = self._parent_splitter.split_text(document.content)

        all_chunks: list[Chunk] = []
        child_index = 0  # 모든 부모에 걸친 전역 자식 인덱스

        for parent_idx, parent_text in enumerate(parent_texts):
            # 결정론적 부모 청크 ID 생성
            parent_id = self._generate_chunk_id(document.id, "parent", parent_idx)

            # 부모 청크 생성
            parent_chunk = Chunk(
                id=parent_id,
                document_id=document.id,
                chunk_index=parent_idx,
                text=parent_text,
                chunk_type=ChunkType.PARENT,
                parent_id=None,
                metadata={
                    "source_id": document.source_id,
                    "document_title": document.title,
                    "document_url": document.url,
                    "total_parents": len(parent_texts),
                    "parent_index": parent_idx,
                },
            )
            parent_chunk.estimate_tokens()

            if include_parents:
                all_chunks.append(parent_chunk)

            # 이 부모로부터 자식 청크 생성
            child_texts = self._child_splitter.split_text(parent_text)

            for local_child_idx, child_text in enumerate(child_texts):
                child_id = self._generate_chunk_id(
                    document.id, "child", child_index
                )

                child_chunk = Chunk(
                    id=child_id,
                    document_id=document.id,
                    chunk_index=child_index,
                    text=child_text,
                    chunk_type=ChunkType.CHILD,
                    parent_id=parent_id,
                    metadata={
                        "source_id": document.source_id,
                        "document_title": document.title,
                        "document_url": document.url,
                        "total_chunks": sum(
                            len(self._child_splitter.split_text(pt))
                            for pt in parent_texts
                        ),
                        "parent_index": parent_idx,
                        "child_index_in_parent": local_child_idx,
                    },
                )
                child_chunk.estimate_tokens()
                all_chunks.append(child_chunk)
                child_index += 1

        return all_chunks

    def chunk_documents(
        self,
        documents: list[Document],
        include_parents: bool = True,
    ) -> list[Chunk]:
        """여러 문서를 계층적 청크로 분할합니다.

        Args:
            documents: 분할할 문서 리스트.
            include_parents: 출력에 부모 청크 포함 여부.

        Returns:
            모든 Chunk 객체 리스트.
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc, include_parents=include_parents)
            all_chunks.extend(chunks)
        return all_chunks

    def get_parent_chunk(
        self,
        child_chunk: Chunk,
        all_chunks: list[Chunk],
    ) -> Chunk | None:
        """주어진 자식 청크의 부모 청크를 찾습니다.

        Args:
            child_chunk: 부모를 찾을 자식 청크.
            all_chunks: 검색할 모든 청크 리스트.

        Returns:
            부모 청크가 있으면 반환, 없으면 None.
        """
        if child_chunk.parent_id is None:
            return None

        for chunk in all_chunks:
            if chunk.id == child_chunk.parent_id:
                return chunk
        return None

    def _generate_chunk_id(
        self,
        document_id: str,
        chunk_type: str,
        chunk_index: int,
    ) -> str:
        """결정론적 청크 ID를 생성합니다.

        부모와 자식 청크를 구분하기 위해
        chunk_type을 ID 생성에 포함합니다.

        Args:
            document_id: 부모 문서 ID.
            chunk_type: 'parent' 또는 'child'.
            chunk_index: 문서 내 청크 위치.

        Returns:
            결정론적 UUID 문자열.
        """
        content = f"{document_id}:{chunk_type}:{chunk_index}"
        hash_bytes = hashlib.sha256(content.encode()).digest()[:16]
        return str(UUID(bytes=hash_bytes))


# 모듈 레벨 싱글톤 인스턴스
_chunker: Chunker | None = None
_hierarchical_chunker: HierarchicalChunker | None = None


def get_chunker(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> Chunker:
    """표준 청커 인스턴스를 가져옵니다.

    싱글톤 패턴을 사용하여 동일한 설정에 대해
    하나의 인스턴스만 생성합니다.

    Args:
        chunk_size: 청크당 최대 문자 수.
        chunk_overlap: 청크 간 오버랩 문자 수.

    Returns:
        Chunker 인스턴스.

    Note:
        첫 호출 시의 설정값이 이후 호출에서도 유지됩니다.
    """
    global _chunker

    if _chunker is None:
        _chunker = Chunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    return _chunker


def get_hierarchical_chunker(
    parent_chunk_size: int = HierarchicalChunker.DEFAULT_PARENT_SIZE,
    parent_chunk_overlap: int = HierarchicalChunker.DEFAULT_PARENT_OVERLAP,
    child_chunk_size: int = HierarchicalChunker.DEFAULT_CHILD_SIZE,
    child_chunk_overlap: int = HierarchicalChunker.DEFAULT_CHILD_OVERLAP,
) -> HierarchicalChunker:
    """계층적 청커 인스턴스를 가져옵니다.

    싱글톤 패턴을 사용하여 동일한 설정에 대해
    하나의 인스턴스만 생성합니다.

    Args:
        parent_chunk_size: 부모 청크당 최대 문자 수.
        parent_chunk_overlap: 부모 청크 간 오버랩 문자 수.
        child_chunk_size: 자식 청크당 최대 문자 수.
        child_chunk_overlap: 자식 청크 간 오버랩 문자 수.

    Returns:
        HierarchicalChunker 인스턴스.

    Note:
        첫 호출 시의 설정값이 이후 호출에서도 유지됩니다.
    """
    global _hierarchical_chunker

    if _hierarchical_chunker is None:
        _hierarchical_chunker = HierarchicalChunker(
            parent_chunk_size=parent_chunk_size,
            parent_chunk_overlap=parent_chunk_overlap,
            child_chunk_size=child_chunk_size,
            child_chunk_overlap=child_chunk_overlap,
        )

    return _hierarchical_chunker

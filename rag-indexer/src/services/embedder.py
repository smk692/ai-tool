"""문서 청크 임베딩 서비스.

공유 EmbeddingModel을 사용하여 벡터 임베딩을 생성합니다.

주요 기능:
    - 청크 리스트에 대한 배치 임베딩 생성
    - 검색 쿼리 임베딩 생성
    - 지연 로딩을 통한 모델 초기화 최적화
"""


from shared import EmbeddingModel, get_embedding_model

from ..models import Chunk


class Embedder:
    """문서 청크 임베딩 서비스.

    공유 EmbeddingModel을 청크 전용 연산으로 래핑합니다.

    주요 특징:
        - multilingual-e5-large-instruct 모델 기본 사용
        - 배치 처리로 효율적인 임베딩 생성
        - 문서용(passage:)과 쿼리용(query:) 프리픽스 자동 적용

    Attributes:
        model_name: HuggingFace 모델 이름.
        batch_size: 배치 처리 크기.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large-instruct",
        batch_size: int = 32,
    ):
        """임베더를 초기화합니다.

        Args:
            model_name: HuggingFace 모델 이름.
                기본값은 다국어 지원 e5-large-instruct 모델.
            batch_size: 임베딩 배치 크기.
                메모리 사용량과 속도의 균형점을 고려하여 설정.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: EmbeddingModel | None = None

    @property
    def model(self) -> EmbeddingModel:
        """임베딩 모델을 지연 로딩합니다.

        첫 호출 시에만 모델을 로드하여 초기화 시간을 절약합니다.

        Returns:
            로드된 EmbeddingModel 인스턴스.
        """
        if self._model is None:
            self._model = get_embedding_model(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        """임베딩 벡터 차원을 반환합니다.

        Returns:
            벡터 차원 수 (예: 1024).
        """
        return self.model.dimension

    def embed_chunks(
        self,
        chunks: list[Chunk],
        show_progress: bool = False,
    ) -> list[Chunk]:
        """청크 리스트에 대한 임베딩을 생성합니다.

        청크 객체를 직접 수정하여 임베딩 벡터를 추가합니다.
        내부적으로 "passage:" 프리픽스를 사용하여 문서 임베딩을 생성합니다.

        Args:
            chunks: 임베딩을 생성할 청크 리스트.
            show_progress: 진행률 표시 여부.

        Returns:
            임베딩이 추가된 청크 리스트.
            입력된 청크 객체가 직접 수정됩니다.
        """
        if not chunks:
            return chunks

        # 배치 임베딩을 위해 텍스트 추출
        texts = [chunk.text for chunk in chunks]

        # 임베딩 생성 (문서용 "passage:" 프리픽스 자동 적용)
        embeddings = self.model.embed_documents(
            texts,
            batch_size=self.batch_size,
            show_progress=show_progress,
        )

        # 청크에 임베딩 연결
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        return chunks

    def embed_query(self, query: str) -> list[float]:
        """검색 쿼리에 대한 임베딩을 생성합니다.

        내부적으로 "query:" 프리픽스를 사용하여 쿼리 임베딩을 생성합니다.
        이는 비대칭 검색 모델의 특성에 맞춘 것입니다.

        Args:
            query: 검색 쿼리 텍스트.

        Returns:
            쿼리 임베딩 벡터 (float 리스트).
        """
        return self.model.embed_query(query)


# 모듈 레벨 싱글톤 인스턴스
_embedder: Embedder | None = None


def get_embedder(
    model_name: str = "intfloat/multilingual-e5-large-instruct",
    batch_size: int = 32,
) -> Embedder:
    """임베더 인스턴스를 가져옵니다.

    싱글톤 패턴을 사용하여 동일한 설정에 대해
    하나의 인스턴스만 생성합니다.

    Args:
        model_name: HuggingFace 모델 이름.
        batch_size: 임베딩 배치 크기.

    Returns:
        Embedder 인스턴스.

    Note:
        첫 호출 시의 설정값이 이후 호출에서도 유지됩니다.
        다른 설정이 필요하면 Embedder 클래스를 직접 인스턴스화하세요.
    """
    global _embedder

    if _embedder is None:
        _embedder = Embedder(
            model_name=model_name,
            batch_size=batch_size,
        )

    return _embedder

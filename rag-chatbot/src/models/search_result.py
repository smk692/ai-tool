"""SearchResult 모델 정의.

벡터DB에서 검색된 문서 청크를 나타냅니다.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """벡터DB 검색 결과 모델.

    Qdrant에서 반환된 검색 결과를 파싱하여 저장합니다.

    Attributes:
        chunk_id: 청크 고유 ID
        content: 청크 텍스트 내용
        score: 유사도 점수 (0.0 ~ 1.0)
        source_type: 문서 소스 타입 (notion, swagger)
        source_id: 원본 문서 ID
        source_title: 원본 문서 제목
        source_url: 원본 문서 URL (있는 경우)
        metadata: 추가 메타데이터
    """

    chunk_id: str = Field(..., description="청크 고유 ID")
    content: str = Field(..., min_length=1, description="청크 텍스트 내용")
    score: float = Field(..., ge=0.0, le=1.0, description="유사도 점수")
    source_type: Literal["notion", "swagger"] = Field(..., description="문서 소스 타입")
    source_id: str = Field(..., description="원본 문서 ID")
    source_title: str = Field(..., description="원본 문서 제목")
    source_url: str | None = Field(default=None, description="원본 문서 URL")
    metadata: dict[str, Any] = Field(default_factory=dict, description="추가 메타데이터")

    @property
    def is_relevant(self) -> bool:
        """유사도 임계값 충족 여부.

        기본 임계값 0.7 이상이면 관련성 있음으로 판단합니다.

        Returns:
            유사도 점수가 0.7 이상이면 True
        """
        return self.score >= 0.7

    @classmethod
    def from_qdrant_result(
        cls,
        point_id: str,
        score: float,
        payload: dict[str, Any],
    ) -> "SearchResult":
        """Qdrant 검색 결과에서 SearchResult 생성.

        Args:
            point_id: Qdrant 포인트 ID
            score: 유사도 점수
            payload: Qdrant 페이로드

        Returns:
            SearchResult 인스턴스
        """
        return cls(
            chunk_id=point_id,
            content=payload.get("content", ""),
            score=score,
            source_type=payload.get("source_type", "notion"),
            source_id=payload.get("source_id", ""),
            source_title=payload.get("title", payload.get("source_title", "Unknown")),
            source_url=payload.get("url", payload.get("source_url")),
            metadata={
                k: v
                for k, v in payload.items()
                if k not in {"content", "source_type", "source_id", "title", "url"}
            },
        )

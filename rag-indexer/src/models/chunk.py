"""청크(Chunk) 모델 정의.

벡터 인덱싱을 위한 문서 텍스트 청크를 나타냅니다.
"""

import re
from datetime import UTC, datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def detect_language(text: str) -> str:
    """텍스트 언어 감지 (한국어 또는 영어).

    문자 기반 휴리스틱 사용:
    - 한국어: 한글 문자 (가-힣)
    - 영어: 기본 폴백

    Args:
        text: 분석할 텍스트.

    Returns:
        언어 코드: 'ko' 또는 'en'.
    """
    if not text:
        return "en"

    # 한글 문자 개수 계산 (한글 음절: U+AC00-U+D7A3)
    korean_chars = len(re.findall(r"[\uac00-\ud7a3]", text))
    total_alpha = len(re.findall(r"[a-zA-Z\uac00-\ud7a3]", text))

    if total_alpha == 0:
        return "en"

    korean_ratio = korean_chars / total_alpha
    return "ko" if korean_ratio > 0.3 else "en"


class ChunkType(str, Enum):
    """계층적 청킹에서의 청크 유형.

    Attributes:
        STANDARD: 일반 청크 (비계층적)
        PARENT: 컨텍스트용 대형 청크
        CHILD: 정밀 검색용 소형 청크
    """

    STANDARD = "standard"  # 일반 청크 (비계층적)
    PARENT = "parent"  # 컨텍스트용 대형 청크
    CHILD = "child"  # 정밀 검색용 소형 청크


class Chunk(BaseModel):
    """문서에서 추출한 텍스트 청크.

    청크는 임베딩되어 벡터 데이터베이스에 저장되는 문서 세그먼트입니다.

    계층적 청킹 지원 (Phase 1.3):
    - STANDARD: 일반 청크 (기본값, 비계층적)
    - PARENT: 대형 청크 (~4000자) - 포괄적인 컨텍스트 제공
    - CHILD: 소형 청크 (~800자) - 정밀 검색용, 부모 참조

    Attributes:
        id: 고유 청크 식별자 (UUID)
        document_id: 부모 문서 ID
        chunk_index: 문서 내 청크 위치 (0부터 시작)
        text: 청크 텍스트 내용
        token_count: 대략적인 토큰 수
        embedding: 임베딩 벡터 (1024 차원)
        metadata: 청크별 메타데이터
        created_at: 생성 타임스탬프
        chunk_type: 청크 유형 (standard, parent, child)
        parent_id: 부모 청크 ID (child 청크만 해당)
    """

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="고유 청크 식별자 (UUID)",
    )
    document_id: str = Field(
        ...,
        description="부모 문서 ID",
    )
    chunk_index: int = Field(
        ...,
        description="문서 내 청크 위치 (0부터 시작)",
    )
    text: str = Field(
        ...,
        description="청크 텍스트 내용",
    )
    token_count: int = Field(
        default=0,
        description="대략적인 토큰 수",
    )
    embedding: Optional[list[float]] = Field(
        default=None,
        description="임베딩 벡터 (1024 차원)",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="청크별 메타데이터",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="생성 타임스탬프",
    )
    # Phase 1.3: 계층적 청킹 필드
    chunk_type: ChunkType = Field(
        default=ChunkType.STANDARD,
        description="청크 유형: standard, parent, child",
    )
    parent_id: Optional[str] = Field(
        default=None,
        description="부모 청크 ID (child 청크만 해당)",
    )

    def to_qdrant_point(
        self,
        source_id: str,
        source_type: str,
        title: str,
        url: Optional[str] = None,
        document_metadata: Optional[dict] = None,
    ) -> dict:
        """향상된 메타데이터와 함께 Qdrant 포인트 포맷으로 변환.

        Args:
            source_id: 페이로드용 소스 ID.
            source_type: 페이로드용 소스 유형.
            title: 페이로드용 문서 제목.
            url: 페이로드용 문서 URL.
            document_metadata: 병합할 문서 수준 메타데이터 (선택적).

        Returns:
            Qdrant upsert용 딕셔너리.

        향상된 메타데이터 필드 (2025):
            - language: 감지된 언어 (ko/en)
            - token_count: 추정 토큰 수
            - total_chunks: 부모 문서의 총 청크 수
            - created_at: 청크 생성 타임스탬프 (ISO 포맷)
            - Notion 전용: notion_page_id, notion_parent_id, notion_database_id
            - Swagger 전용: api_endpoint, http_method, api_version
            - AI 추출 (Phase 1.2): ai_content_type, ai_topics, ai_difficulty,
              ai_has_code_samples, ai_key_entities, ai_summary
            - 계층적 (Phase 1.3): chunk_type, parent_id

        Raises:
            ValueError: 청크에 임베딩이 없는 경우.
        """
        if self.embedding is None:
            raise ValueError("청크에 임베딩이 없습니다")

        # 핵심 필드가 포함된 기본 페이로드
        payload = {
            "chunk_id": self.id,
            "document_id": self.document_id,
            "source_id": source_id,
            "source_type": source_type,
            "title": title,
            "url": url,
            "chunk_index": self.chunk_index,
            "text": self.text,
            # 향상된 필드 (Phase 1.1)
            "language": detect_language(self.text),
            "token_count": self.token_count,
            "total_chunks": self.metadata.get("total_chunks", 1),
            "created_at": self.created_at.isoformat(),
            # 계층적 청킹 필드 (Phase 1.3)
            "chunk_type": self.chunk_type.value,
            "parent_id": self.parent_id,
        }

        # 청크 수준 메타데이터 병합 (중복 제외)
        for key, value in self.metadata.items():
            if key not in payload and value is not None:
                payload[key] = value

        # 문서 수준 메타데이터 병합 (소스별 필드)
        if document_metadata:
            # Notion 전용 필드
            notion_fields = [
                "notion_page_id",
                "notion_parent_id",
                "notion_database_id",
                "notion_last_edited_time",
            ]
            for field in notion_fields:
                if field in document_metadata:
                    payload[field] = document_metadata[field]

            # Swagger 전용 필드
            swagger_fields = [
                "api_endpoint",
                "http_method",
                "api_version",
                "operation_id",
                "tags",
            ]
            for field in swagger_fields:
                if field in document_metadata:
                    payload[field] = document_metadata[field]

            # 일반 메타데이터 필드
            generic_fields = ["content_type", "author", "category"]
            for field in generic_fields:
                if field in document_metadata:
                    payload[field] = document_metadata[field]

            # AI 추출 메타데이터 필드 (Phase 1.2)
            ai_fields = [
                "ai_content_type",
                "ai_topics",
                "ai_difficulty",
                "ai_has_code_samples",
                "ai_key_entities",
                "ai_summary",
            ]
            for field in ai_fields:
                if field in document_metadata:
                    payload[field] = document_metadata[field]

        return {
            "id": self.id,
            "vector": self.embedding,
            "payload": payload,
        }

    def estimate_tokens(self) -> int:
        """텍스트 길이 기반 토큰 수 추정.

        대략적인 근사값 사용:
        - 영어: ~4자/토큰
        - 한국어/CJK: ~2자/토큰
        - 평균: ~3자/토큰

        Returns:
            추정 토큰 수.
        """
        # 단순 휴리스틱: 평균 ~3자/토큰 가정
        self.token_count = len(self.text) // 3
        return self.token_count

    def model_dump_json_safe(self) -> dict:
        """JSON 직렬화 가능한 딕셔너리로 변환 (임베딩 제외).

        임베딩은 크기가 크므로 제외하고,
        datetime을 ISO 포맷 문자열로 변환합니다.

        Returns:
            JSON 직렬화 가능한 딕셔너리.
        """
        data = self.model_dump(exclude={"embedding"})
        data["created_at"] = self.created_at.isoformat()
        return data

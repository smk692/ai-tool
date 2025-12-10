"""문서(Document) 모델 정의.

데이터 소스에서 추출된 문서를 나타냅니다.
"""

from datetime import UTC, datetime
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class Document(BaseModel):
    """데이터 소스에서 추출된 문서.

    청킹 및 인덱싱이 가능한 단일 문서 단위(페이지, 엔드포인트 등)를 나타냅니다.

    Attributes:
        id: 고유 문서 식별자 (UUID)
        source_id: 부모 소스 ID
        external_id: 외부 시스템 ID (예: Notion page_id)
        title: 문서 제목
        content: 전체 문서 내용 텍스트
        url: 원본 문서 URL
        content_hash: 변경 감지를 위한 콘텐츠 SHA256 해시
        metadata: 소스별 추가 메타데이터
        created_at: 생성 타임스탬프
        updated_at: 마지막 업데이트 타임스탬프
        indexed_at: 벡터DB에 인덱싱된 시간
    """

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="고유 문서 식별자 (UUID)",
    )
    source_id: str = Field(
        ...,
        description="부모 소스 ID",
    )
    external_id: str = Field(
        ...,
        description="외부 시스템 ID (예: Notion page_id)",
    )
    title: str = Field(
        ...,
        description="문서 제목",
    )
    content: str = Field(
        default="",
        description="전체 문서 내용 텍스트",
    )
    url: Optional[str] = Field(
        default=None,
        description="원본 문서 URL",
    )
    content_hash: str = Field(
        default="",
        description="변경 감지를 위한 콘텐츠 SHA256 해시",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="소스별 추가 메타데이터",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="생성 타임스탬프",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="마지막 업데이트 타임스탬프",
    )
    indexed_at: Optional[datetime] = Field(
        default=None,
        description="벡터DB에 인덱싱된 시간",
    )

    def mark_indexed(self) -> None:
        """문서를 인덱싱 완료로 표시.

        indexed_at과 updated_at을 현재 시간으로 업데이트합니다.
        """
        self.indexed_at = datetime.now(UTC)
        self.updated_at = datetime.now(UTC)

    def needs_reindex(self, new_hash: str) -> bool:
        """재인덱싱 필요 여부 확인.

        콘텐츠 해시를 비교하여 문서 내용이 변경되었는지 확인합니다.

        Args:
            new_hash: 비교할 새 콘텐츠 해시.

        Returns:
            콘텐츠가 변경되어 재인덱싱이 필요하면 True.
        """
        return self.content_hash != new_hash

    def model_dump_json_safe(self) -> dict:
        """JSON 직렬화 가능한 딕셔너리로 변환.

        datetime 필드를 ISO 포맷 문자열로 변환합니다.

        Returns:
            JSON 직렬화 가능한 딕셔너리.
        """
        data = self.model_dump()
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        if self.indexed_at:
            data["indexed_at"] = self.indexed_at.isoformat()
        return data

    @classmethod
    def from_json_safe(cls, data: dict) -> "Document":
        """JSON-safe 딕셔너리에서 Document 생성.

        ISO 포맷 문자열을 datetime 객체로 파싱합니다.

        Args:
            data: JSON-safe 딕셔너리 데이터.

        Returns:
            Document 인스턴스.
        """
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        if data.get("indexed_at") and isinstance(data["indexed_at"], str):
            data["indexed_at"] = datetime.fromisoformat(data["indexed_at"])
        return cls(**data)

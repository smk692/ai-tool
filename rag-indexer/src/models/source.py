"""소스(Source) 모델 정의.

인덱싱을 위한 데이터 소스 유형 및 설정을 정의합니다.
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Annotated, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """지원되는 데이터 소스 유형.

    Attributes:
        NOTION: Notion 페이지/데이터베이스 소스
        SWAGGER: Swagger/OpenAPI 명세 소스
    """

    NOTION = "notion"
    SWAGGER = "swagger"


class NotionSourceConfig(BaseModel):
    """Notion 데이터 소스 설정.

    Notion에서 동기화할 페이지 및 데이터베이스를 지정합니다.

    Attributes:
        page_ids: 동기화할 Notion 페이지 ID 목록
        database_ids: 동기화할 Notion 데이터베이스 ID 목록
        include_children: 하위 페이지 포함 여부
    """

    page_ids: list[str] = Field(
        default_factory=list,
        description="동기화할 Notion 페이지 ID 목록",
    )
    database_ids: list[str] = Field(
        default_factory=list,
        description="동기화할 Notion 데이터베이스 ID 목록",
    )
    include_children: bool = Field(
        default=True,
        description="하위 페이지 포함 여부",
    )


class SwaggerSourceConfig(BaseModel):
    """Swagger/OpenAPI 데이터 소스 설정.

    OpenAPI 명세 문서를 가져올 URL과 인증 정보를 지정합니다.

    Attributes:
        url: Swagger/OpenAPI JSON 명세 URL
        auth_header: 선택적 인증 헤더 값
    """

    url: str = Field(
        ...,
        description="Swagger/OpenAPI JSON 명세 URL",
    )
    auth_header: Optional[str] = Field(
        default=None,
        description="선택적 인증 헤더 값",
    )


# 소스 설정 유니온 타입 (Notion 또는 Swagger)
SourceConfig = Annotated[
    Union[NotionSourceConfig, SwaggerSourceConfig],
    Field(discriminator=None),
]


class Source(BaseModel):
    """데이터 소스 정의.

    문서 인덱싱을 위한 설정된 소스를 나타냅니다.

    Attributes:
        id: 고유 소스 식별자 (UUID)
        name: 사람이 읽을 수 있는 소스 이름
        source_type: 데이터 소스 유형
        config: 유형별 설정
        enabled: 동기화 활성화 여부
        created_at: 생성 타임스탬프
        updated_at: 마지막 업데이트 타임스탬프
        last_synced_at: 마지막 성공적인 동기화 타임스탬프
    """

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="고유 소스 식별자 (UUID)",
    )
    name: str = Field(
        ...,
        description="사람이 읽을 수 있는 소스 이름",
    )
    source_type: SourceType = Field(
        ...,
        description="데이터 소스 유형",
    )
    config: SourceConfig = Field(
        ...,
        description="유형별 설정",
    )
    enabled: bool = Field(
        default=True,
        description="동기화 활성화 여부",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="생성 타임스탬프",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="마지막 업데이트 타임스탬프",
    )
    last_synced_at: Optional[datetime] = Field(
        default=None,
        description="마지막 성공적인 동기화 타임스탬프",
    )

    def update_sync_time(self) -> None:
        """마지막 동기화 타임스탬프를 현재 시간으로 업데이트.

        last_synced_at과 updated_at을 현재 시간으로 설정합니다.
        """
        self.last_synced_at = datetime.now(UTC)
        self.updated_at = datetime.now(UTC)

    def model_dump_json_safe(self) -> dict:
        """JSON 직렬화 가능한 딕셔너리로 변환.

        datetime 필드를 ISO 포맷 문자열로 변환합니다.

        Returns:
            JSON 직렬화 가능한 딕셔너리.
        """
        data = self.model_dump()
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        if self.last_synced_at:
            data["last_synced_at"] = self.last_synced_at.isoformat()
        return data

    @classmethod
    def from_json_safe(cls, data: dict) -> "Source":
        """JSON-safe 딕셔너리에서 Source 생성.

        ISO 포맷 문자열을 datetime 객체로 파싱하고,
        source_type에 따라 적절한 config 객체를 생성합니다.

        Args:
            data: JSON-safe 딕셔너리 데이터.

        Returns:
            Source 인스턴스.
        """
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        if data.get("last_synced_at") and isinstance(data["last_synced_at"], str):
            data["last_synced_at"] = datetime.fromisoformat(data["last_synced_at"])

        # source_type에 따라 config 파싱
        source_type = SourceType(data["source_type"])
        if source_type == SourceType.NOTION:
            data["config"] = NotionSourceConfig(**data["config"])
        elif source_type == SourceType.SWAGGER:
            data["config"] = SwaggerSourceConfig(**data["config"])

        return cls(**data)

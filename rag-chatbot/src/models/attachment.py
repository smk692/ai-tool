"""첨부 파일 모델.

Slack 파일 및 Claude Vision API용 이미지 콘텐츠 모델을 정의합니다.
"""

from pydantic import BaseModel, Field


class ImageContent(BaseModel):
    """Claude Vision API용 이미지 콘텐츠.

    Base64 인코딩된 이미지 데이터와 메타데이터를 담습니다.

    Attributes:
        media_type: MIME 타입 (image/jpeg, image/png 등)
        data: Base64 인코딩된 이미지 데이터
        filename: 원본 파일명 (선택)
        file_size: 파일 크기 (bytes)
    """

    media_type: str = Field(..., description="이미지 MIME 타입")
    data: str = Field(..., description="Base64 인코딩된 이미지 데이터")
    filename: str | None = Field(default=None, description="원본 파일명")
    file_size: int = Field(default=0, ge=0, description="파일 크기 (bytes)")


class SlackFileInfo(BaseModel):
    """Slack 파일 메타데이터.

    Slack API에서 반환하는 파일 정보를 담습니다.

    Attributes:
        id: Slack 파일 ID
        name: 파일명
        mimetype: MIME 타입
        size: 파일 크기 (bytes)
        url_private_download: 다운로드 URL (Bot Token 필요)
    """

    id: str = Field(..., description="Slack 파일 ID")
    name: str = Field(..., description="파일명")
    mimetype: str = Field(..., description="MIME 타입")
    size: int = Field(..., ge=0, description="파일 크기 (bytes)")
    url_private_download: str = Field(..., description="다운로드 URL")

    @property
    def is_image(self) -> bool:
        """이미지 파일 여부 확인.

        Returns:
            이미지 파일이면 True
        """
        return self.mimetype.startswith("image/")

    @property
    def is_supported_image(self) -> bool:
        """지원되는 이미지 형식 여부 확인.

        Claude Vision API에서 지원하는 형식인지 확인합니다.
        지원 형식: JPEG, PNG, GIF, WebP

        Returns:
            지원되는 이미지 형식이면 True
        """
        supported_types = {
            "image/jpeg",
            "image/png",
            "image/gif",
            "image/webp",
        }
        return self.mimetype in supported_types

    @classmethod
    def from_slack_file(cls, file_dict: dict) -> "SlackFileInfo":
        """Slack 파일 딕셔너리에서 생성.

        Args:
            file_dict: Slack API 파일 객체

        Returns:
            SlackFileInfo 인스턴스
        """
        return cls(
            id=file_dict.get("id", ""),
            name=file_dict.get("name", ""),
            mimetype=file_dict.get("mimetype", ""),
            size=file_dict.get("size", 0),
            url_private_download=file_dict.get("url_private_download", ""),
        )

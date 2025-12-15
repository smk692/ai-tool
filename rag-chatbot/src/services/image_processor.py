"""이미지 처리 서비스.

Slack 파일을 다운로드하여 Claude Vision API 형식으로 변환합니다.
"""

import base64
import logging

import httpx

from ..config import get_settings
from ..models.attachment import ImageContent, SlackFileInfo

logger = logging.getLogger(__name__)


class ImageProcessor:
    """이미지 처리기.

    Slack 파일을 다운로드하고 Claude Vision API에서 사용 가능한 형식으로 변환합니다.
    """

    # 지원하는 이미지 MIME 타입
    SUPPORTED_MIMETYPES = {
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/webp",
    }

    def __init__(self) -> None:
        """이미지 처리기 초기화."""
        self.settings = get_settings()

    @property
    def max_image_size(self) -> int:
        """최대 이미지 크기 (bytes)."""
        return self.settings.image_max_size_mb * 1024 * 1024

    @property
    def max_images_per_request(self) -> int:
        """요청당 최대 이미지 수."""
        return self.settings.image_max_count

    @property
    def download_timeout(self) -> int:
        """다운로드 타임아웃 (초)."""
        return self.settings.image_download_timeout

    async def process_slack_files(
        self,
        files: list[dict],
        bot_token: str | None = None,
    ) -> list[ImageContent]:
        """Slack 파일을 Claude Vision 형식으로 변환.

        Args:
            files: Slack 파일 딕셔너리 배열
            bot_token: Slack Bot 토큰 (None이면 설정에서 가져옴)

        Returns:
            ImageContent 리스트
        """
        if not self.settings.image_processing_enabled:
            logger.debug("이미지 처리 비활성화됨")
            return []

        token = bot_token or self.settings.slack_bot_token

        # 이미지 파일만 필터링
        image_files = self._filter_image_files(files)
        if not image_files:
            logger.debug("처리할 이미지 파일 없음")
            return []

        # 개수 제한 적용
        if len(image_files) > self.max_images_per_request:
            logger.warning(
                f"이미지 개수 제한 초과: {len(image_files)}개 → "
                f"{self.max_images_per_request}개로 제한"
            )
            image_files = image_files[: self.max_images_per_request]

        # 각 이미지 처리
        results: list[ImageContent] = []
        for file_info in image_files:
            try:
                image_content = await self._process_single_file(file_info, token)
                if image_content:
                    results.append(image_content)
            except Exception as e:
                logger.warning(f"이미지 처리 실패 ({file_info.name}): {e}")
                continue

        logger.info(f"이미지 처리 완료: {len(results)}개 성공")
        return results

    def _filter_image_files(self, files: list[dict]) -> list[SlackFileInfo]:
        """지원되는 이미지 파일만 필터링.

        Args:
            files: Slack 파일 딕셔너리 배열

        Returns:
            SlackFileInfo 리스트
        """
        result = []
        for file_dict in files:
            try:
                file_info = SlackFileInfo.from_slack_file(file_dict)
                if not file_info.is_supported_image:
                    logger.debug(f"지원하지 않는 형식 건너뜀: {file_info.mimetype}")
                    continue
                if file_info.size > self.max_image_size:
                    logger.warning(
                        f"파일 크기 초과 건너뜀: {file_info.name} "
                        f"({file_info.size / 1024 / 1024:.1f}MB > "
                        f"{self.settings.image_max_size_mb}MB)"
                    )
                    continue
                result.append(file_info)
            except Exception as e:
                logger.warning(f"파일 정보 파싱 실패: {e}")
                continue
        return result

    async def _process_single_file(
        self,
        file_info: SlackFileInfo,
        bot_token: str,
    ) -> ImageContent | None:
        """단일 파일 처리.

        Args:
            file_info: Slack 파일 정보
            bot_token: Slack Bot 토큰

        Returns:
            ImageContent 또는 None
        """
        # 파일 다운로드
        image_data = await self._download_file(
            url=file_info.url_private_download,
            bot_token=bot_token,
        )
        if not image_data:
            return None

        # Base64 인코딩
        encoded_data = base64.b64encode(image_data).decode("utf-8")

        return ImageContent(
            media_type=file_info.mimetype,
            data=encoded_data,
            filename=file_info.name,
            file_size=len(image_data),
        )

    async def _download_file(
        self,
        url: str,
        bot_token: str,
    ) -> bytes | None:
        """Slack에서 파일 다운로드.

        Args:
            url: 다운로드 URL
            bot_token: Slack Bot 토큰

        Returns:
            파일 바이너리 데이터 또는 None
        """
        try:
            async with httpx.AsyncClient(timeout=self.download_timeout) as client:
                response = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {bot_token}"},
                    follow_redirects=True,
                )
                response.raise_for_status()
                return response.content
        except httpx.TimeoutException:
            logger.error(f"파일 다운로드 타임아웃: {url}")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"파일 다운로드 HTTP 에러: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"파일 다운로드 실패: {e}")
            return None


# 기본 인스턴스 (지연 로딩)
_default_processor: ImageProcessor | None = None


def get_image_processor() -> ImageProcessor:
    """이미지 처리기 싱글톤 인스턴스 반환.

    Returns:
        ImageProcessor 인스턴스
    """
    global _default_processor
    if _default_processor is None:
        _default_processor = ImageProcessor()
    return _default_processor

"""Query 모델 정의.

사용자가 Slack에서 보낸 질문을 나타냅니다.
"""

import re
from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator


class Query(BaseModel):
    """사용자 질문 모델.

    Slack에서 수신한 메시지를 파싱하여 질문 정보를 저장합니다.

    Attributes:
        text: 질문 텍스트 (멘션 태그 제거됨)
        user_id: Slack 사용자 ID (e.g., U1234567890)
        channel_id: Slack 채널 ID (e.g., C1234567890 또는 D1234567890)
        thread_ts: 스레드 타임스탬프 (대화 식별자)
        message_ts: 메시지 타임스탬프 (고유 식별자)
        is_dm: DM 여부
        created_at: 질문 수신 시간 (UTC)
    """

    text: str = Field(..., min_length=1, max_length=4000, description="질문 텍스트")
    user_id: str = Field(..., pattern=r"^U[A-Z0-9]{10}$", description="Slack 사용자 ID")
    channel_id: str = Field(
        ..., pattern=r"^[CD][A-Z0-9]{10}$", description="Slack 채널 ID"
    )
    thread_ts: str = Field(..., pattern=r"^\d+\.\d+$", description="스레드 타임스탬프")
    message_ts: str = Field(..., pattern=r"^\d+\.\d+$", description="메시지 타임스탬프")
    is_dm: bool = Field(default=False, description="DM 여부")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="수신 시간"
    )

    @field_validator("text", mode="before")
    @classmethod
    def strip_mention(cls, v: str) -> str:
        """멘션 태그 제거.

        Args:
            v: 원본 텍스트

        Returns:
            멘션 태그가 제거된 텍스트
        """
        if not isinstance(v, str):
            return v
        # <@U1234567890> 형식의 멘션 제거
        cleaned = re.sub(r"<@[A-Z0-9]+>", "", v).strip()
        return cleaned if cleaned else v.strip()

    @classmethod
    def from_slack_event(
        cls,
        text: str,
        user: str,
        channel: str,
        ts: str,
        thread_ts: str | None = None,
        channel_type: str | None = None,
    ) -> "Query":
        """Slack 이벤트에서 Query 생성.

        Args:
            text: 메시지 텍스트
            user: 사용자 ID
            channel: 채널 ID
            ts: 메시지 타임스탬프
            thread_ts: 스레드 타임스탬프 (없으면 ts 사용)
            channel_type: 채널 타입 (im = DM)

        Returns:
            Query 인스턴스
        """
        return cls(
            text=text,
            user_id=user,
            channel_id=channel,
            message_ts=ts,
            thread_ts=thread_ts or ts,
            is_dm=channel_type == "im" or channel.startswith("D"),
        )

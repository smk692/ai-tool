"""Feedback 모델 정의.

사용자 피드백을 나타냅니다. Redis에 저장 후 JSON으로 백업됩니다.
"""

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class Feedback(BaseModel):
    """사용자 피드백 모델.

    답변에 대한 사용자 피드백 (리액션)을 저장합니다.

    Attributes:
        message_ts: 답변 메시지 타임스탬프 (Primary Key)
        thread_ts: 스레드 타임스탬프
        channel_id: 채널 ID
        user_id: 피드백 제공자 ID
        question: 원본 질문
        answer: 챗봇 답변
        rating: 평가 (positive, negative)
        reaction: Slack 리액션 이름
        created_at: 피드백 시간
    """

    message_ts: str = Field(..., pattern=r"^\d+\.\d+$", description="메시지 타임스탬프")
    thread_ts: str = Field(..., pattern=r"^\d+\.\d+$", description="스레드 타임스탬프")
    channel_id: str = Field(
        ..., pattern=r"^[CD][A-Z0-9]{10}$", description="채널 ID"
    )
    user_id: str = Field(..., pattern=r"^U[A-Z0-9]{10}$", description="사용자 ID")
    question: str = Field(..., description="원본 질문")
    answer: str = Field(..., description="챗봇 답변")
    rating: Literal["positive", "negative"] = Field(..., description="평가")
    reaction: str = Field(..., description="Slack 리액션")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="피드백 시간"
    )

    # 긍정 리액션 목록
    POSITIVE_REACTIONS: set[str] = {
        "+1",
        "thumbsup",
        "white_check_mark",
        "heavy_check_mark",
        "heart",
        "star",
        "clap",
        "tada",
        "100",
    }

    # 부정 리액션 목록
    NEGATIVE_REACTIONS: set[str] = {
        "-1",
        "thumbsdown",
        "x",
        "no_entry",
        "no_entry_sign",
        "disappointed",
    }

    @classmethod
    def rating_from_reaction(cls, reaction: str) -> Literal["positive", "negative"] | None:
        """리액션 이름을 rating으로 변환.

        Args:
            reaction: Slack 리액션 이름

        Returns:
            rating 또는 None (매핑되지 않는 리액션)
        """
        if reaction in cls.POSITIVE_REACTIONS:
            return "positive"
        elif reaction in cls.NEGATIVE_REACTIONS:
            return "negative"
        return None

    @classmethod
    def redis_key(cls, message_ts: str) -> str:
        """Redis 키 생성.

        Args:
            message_ts: 메시지 타임스탬프

        Returns:
            Redis 키 문자열
        """
        return f"feedback:{message_ts}"

    def to_export_dict(self) -> dict:
        """JSON 내보내기용 딕셔너리 변환.

        Returns:
            직렬화 가능한 딕셔너리
        """
        return {
            "message_ts": self.message_ts,
            "thread_ts": self.thread_ts,
            "channel_id": self.channel_id,
            "user_id": self.user_id,
            "question": self.question,
            "answer": self.answer,
            "rating": self.rating,
            "reaction": self.reaction,
            "created_at": self.created_at.isoformat(),
        }

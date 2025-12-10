"""Conversation 모델 정의.

스레드 내 대화 기록을 나타냅니다. Redis에 저장됩니다.
"""

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class ConversationMessage(BaseModel):
    """대화 메시지.

    Attributes:
        role: 역할 (user, assistant)
        content: 메시지 내용
        ts: 메시지 타임스탬프
    """

    role: Literal["user", "assistant"] = Field(..., description="역할")
    content: str = Field(..., description="메시지 내용")
    ts: str = Field(..., description="메시지 타임스탬프")


class Conversation(BaseModel):
    """대화 컨텍스트 모델.

    Redis에 저장되어 스레드 내 대화 기록을 유지합니다.

    Attributes:
        thread_ts: 스레드 타임스탬프 (Primary Key)
        channel_id: 채널 ID
        messages: 대화 메시지 목록 (최대 N개)
        created_at: 대화 시작 시간
        updated_at: 마지막 업데이트 시간
    """

    thread_ts: str = Field(
        ...,
        pattern=r"^(\d+\.\d+|[CD][A-Z0-9]{10})$",
        description="스레드 타임스탬프 또는 DM 채널 ID",
    )
    channel_id: str = Field(
        ..., pattern=r"^[CD][A-Z0-9]{10}$", description="채널 ID"
    )
    messages: list[ConversationMessage] = Field(
        default_factory=list, description="대화 메시지 목록"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="생성 시간"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="업데이트 시간"
    )

    def add_message(
        self,
        role: Literal["user", "assistant"],
        content: str,
        ts: str,
        max_messages: int = 10,
    ) -> None:
        """메시지 추가.

        최대 메시지 수를 초과하면 가장 오래된 메시지부터 제거합니다.

        Args:
            role: 역할 (user, assistant)
            content: 메시지 내용
            ts: 메시지 타임스탬프
            max_messages: 최대 메시지 수 (기본값: 10)
        """
        self.messages.append(ConversationMessage(role=role, content=content, ts=ts))

        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

        self.updated_at = datetime.now(timezone.utc)

    def to_claude_messages(self) -> list[dict[str, str]]:
        """Claude API 형식으로 변환.

        Returns:
            Claude API messages 형식의 리스트
        """
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def get_context_summary(self, max_chars: int = 2000) -> str:
        """대화 컨텍스트 요약 생성.

        Args:
            max_chars: 최대 문자 수

        Returns:
            대화 컨텍스트 요약 문자열
        """
        summary_parts = []
        total_chars = 0

        for msg in reversed(self.messages):
            prefix = "사용자: " if msg.role == "user" else "어시스턴트: "
            line = f"{prefix}{msg.content}"

            if total_chars + len(line) > max_chars:
                break

            summary_parts.insert(0, line)
            total_chars += len(line) + 1

        return "\n".join(summary_parts)

    @classmethod
    def redis_key(cls, thread_ts: str) -> str:
        """Redis 키 생성.

        Args:
            thread_ts: 스레드 타임스탬프

        Returns:
            Redis 키 문자열
        """
        return f"conversation:{thread_ts}"

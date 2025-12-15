"""대화 컨텍스트 관리 서비스.

Redis를 사용하여 스레드별 대화 기록을 관리합니다.
"""

import logging
from typing import Literal

import redis

from ..config import get_settings
from ..models import Conversation

logger = logging.getLogger(__name__)


class ConversationService:
    """대화 컨텍스트 관리 서비스.

    Redis에 스레드별 대화 기록을 저장하고 관리합니다.
    """

    def __init__(
        self,
        redis_client: redis.Redis | None = None,
    ) -> None:
        """ConversationService 초기화.

        Args:
            redis_client: Redis 클라이언트 (None이면 새로 생성)
        """
        self.settings = get_settings()

        if redis_client:
            self._redis = redis_client
        else:
            self._redis = redis.Redis(
                host=self.settings.redis_host,
                port=self.settings.redis_port,
                db=self.settings.redis_db,
                decode_responses=True,
            )

        logger.info("ConversationService 초기화 완료")

    def get_conversation(
        self,
        thread_ts: str,
        channel_id: str,
    ) -> Conversation | None:
        """기존의 대화 기록 조회.

        Args:
            thread_ts: 스레드 타임스탬프
            channel_id: 채널 ID

        Returns:
            대화 기록 (없으면 None)
        """
        key = Conversation.redis_key(thread_ts)

        try:
            data = self._redis.get(key)
            if data:
                return Conversation.model_validate_json(data)
            return None
        except Exception as e:
            logger.error(f"대화 조회 실패 (thread_ts={thread_ts}): {e}")
            return None

    def get_or_create_conversation(
        self,
        thread_ts: str,
        channel_id: str,
    ) -> Conversation:
        """기존의 대화 기록 조회 또는 생성.

        Args:
            thread_ts: 스레드 타임스탬프
            channel_id: 채널 ID

        Returns:
            대화 기록
        """
        conversation = self.get_conversation(thread_ts, channel_id)
        if conversation:
            return conversation

        return Conversation(
            thread_ts=thread_ts,
            channel_id=channel_id,
        )

    def add_message(
        self,
        thread_ts: str,
        channel_id: str,
        role: Literal["user", "assistant"],
        content: str,
        message_ts: str,
        max_messages: int | None = None,
    ) -> Conversation:
        """대화에 메시지 추가.

        Args:
            thread_ts: 스레드 타임스탬프
            channel_id: 채널 ID
            role: 메시지 역할 (user/assistant)
            content: 메시지 내용
            message_ts: 메시지 타임스탬프
            max_messages: 최대 메시지 수 (None이면 기본값 사용)

        Returns:
            업데이트된 대화 기록
        """
        conversation = self.get_or_create_conversation(thread_ts, channel_id)

        conversation.add_message(
            role=role,
            content=content,
            ts=message_ts,
            max_messages=max_messages or self.settings.conversation_max_messages,
        )

        self.save_conversation(conversation)
        return conversation

    def save_conversation(self, conversation: Conversation) -> bool:
        """대화 기록 저장.

        TTL이 설정된 상태로 저장됩니다.

        Args:
            conversation: 저장할 대화 기록

        Returns:
            저장 성공 여부
        """
        key = Conversation.redis_key(conversation.thread_ts)

        try:
            data = conversation.model_dump_json()
            self._redis.setex(
                name=key,
                time=self.settings.conversation_ttl_seconds,
                value=data,
            )
            logger.debug(
                f"대화 저장 완료 (thread_ts={conversation.thread_ts}, "
                f"messages={len(conversation.messages)})"
            )
            return True
        except Exception as e:
            logger.error(f"대화 저장 실패 (thread_ts={conversation.thread_ts}): {e}")
            return False

    def delete_conversation(self, thread_ts: str) -> bool:
        """대화 기록 삭제.

        Args:
            thread_ts: 스레드 타임스탬프

        Returns:
            삭제 성공 여부
        """
        key = Conversation.redis_key(thread_ts)

        try:
            self._redis.delete(key)
            logger.debug(f"대화 삭제 완료 (thread_ts={thread_ts})")
            return True
        except Exception as e:
            logger.error(f"대화 삭제 실패 (thread_ts={thread_ts}): {e}")
            return False

    def get_context_summary(
        self,
        thread_ts: str,
        channel_id: str,
        max_chars: int = 2000,
    ) -> str | None:
        """대화 컨텍스트 요약 조회.

        RAG 프롬프트에 포함할 이전 대화 요약을 생성합니다.

        Args:
            thread_ts: 스레드 타임스탬프
            channel_id: 채널 ID
            max_chars: 최대 문자 수

        Returns:
            대화 요약 문자열 (없으면 None)
        """
        conversation = self.get_conversation(thread_ts, channel_id)
        if not conversation or not conversation.messages:
            return None

        return conversation.get_context_summary(max_chars=max_chars)

    def health_check(self) -> dict:
        """서비스 상태 확인.

        Returns:
            상태 정보 딕셔너리
        """
        try:
            self._redis.ping()
            return {
                "status": "healthy",
                "redis": {
                    "connected": True,
                    "host": self.settings.redis_host,
                    "port": self.settings.redis_port,
                },
            }
        except Exception as e:
            logger.error(f"Redis 연결 실패: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
            }


# 기본 인스턴스 (싱글톤 패턴)
_default_service: ConversationService | None = None


def get_conversation_service() -> ConversationService:
    """ConversationService 싱글톤 인스턴스 반환.

    Returns:
        ConversationService 인스턴스
    """
    global _default_service
    if _default_service is None:
        _default_service = ConversationService()
    return _default_service

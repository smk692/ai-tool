"""DM 이벤트 핸들러.

사용자가 봇에게 직접 메시지를 보낼 때 질문을 처리합니다.
스레드 기반 컨텍스트와 리액션 피드백을 지원합니다.
"""

import logging
from typing import Any

from slack_bolt import App

from ..utils import add_reaction_safe, remove_reaction_safe
from .base import HandlerContext, MessageProcessor, build_handler_context

logger = logging.getLogger(__name__)


class DMProcessor(MessageProcessor):
    """DM 메시지 처리기.

    사용자가 봇에게 DM을 보내면 RAG 파이프라인을 통해 답변을 생성합니다.
    리액션으로 처리 상태를 표시합니다.
    """

    def _get_channel_type(self) -> str:
        """채널 타입 반환.

        Returns:
            "im" (DM은 항상 im)
        """
        return "im"

    def _get_max_messages(self) -> int:
        """최대 대화 메시지 수 반환.

        Returns:
            DM 대화 최대 메시지 수
        """
        return self.settings.dm_conversation_max_messages

    def _get_log_prefix(self) -> str:
        """로그 접두사 반환.

        Returns:
            "DM"
        """
        return "DM"

    def _get_response_thread_ts(self, ctx: HandlerContext) -> str:
        """응답을 보낼 스레드 타임스탬프 반환.

        DM에서는 message_ts를 사용하여 스레드 응답합니다.

        Args:
            ctx: 핸들러 컨텍스트

        Returns:
            메시지 타임스탬프
        """
        return ctx.message_ts

    def _on_start(self, ctx: HandlerContext) -> None:
        """처리 시작 훅 - 눈 리액션 추가.

        Args:
            ctx: 핸들러 컨텍스트
        """
        add_reaction_safe(
            client=ctx.client,
            channel=ctx.channel_id,
            timestamp=ctx.message_ts,
            name=self.settings.reaction_processing,
        )

    def _on_success(self, ctx: HandlerContext) -> None:
        """처리 성공 훅 - 리액션 교체 (눈 → 체크마크).

        Args:
            ctx: 핸들러 컨텍스트
        """
        remove_reaction_safe(
            client=ctx.client,
            channel=ctx.channel_id,
            timestamp=ctx.message_ts,
            name=self.settings.reaction_processing,
        )
        add_reaction_safe(
            client=ctx.client,
            channel=ctx.channel_id,
            timestamp=ctx.message_ts,
            name=self.settings.reaction_done,
        )

    def _on_error(self, ctx: HandlerContext, error: Exception) -> None:
        """처리 실패 훅 - 처리 중 리액션 제거 후 에러 메시지 전송.

        Args:
            ctx: 핸들러 컨텍스트
            error: 발생한 예외
        """
        remove_reaction_safe(
            client=ctx.client,
            channel=ctx.channel_id,
            timestamp=ctx.message_ts,
            name=self.settings.reaction_processing,
        )
        super()._on_error(ctx, error)

    def _on_empty_message(self, ctx: HandlerContext) -> None:
        """빈 메시지 처리 훅 - 리액션 제거 후 안내 메시지.

        Args:
            ctx: 핸들러 컨텍스트
        """
        remove_reaction_safe(
            client=ctx.client,
            channel=ctx.channel_id,
            timestamp=ctx.message_ts,
            name=self.settings.reaction_processing,
        )
        super()._on_empty_message(ctx)


def register_dm_handlers(app: App) -> None:
    """DM 핸들러 등록.

    Args:
        app: Slack Bolt App 인스턴스
    """
    processor = DMProcessor()

    @app.event("message")
    def handle_dm(body: dict[str, Any], say, client) -> None:
        """DM 이벤트 처리.

        사용자가 봇에게 DM을 보내면 RAG 파이프라인을 통해 답변을 생성합니다.
        스레드 단위로 컨텍스트를 유지하고, 리액션으로 처리 상태를 표시합니다.

        Args:
            body: 이벤트 페이로드
            say: Slack 메시지 전송 함수
            client: Slack 클라이언트
        """
        event = body.get("event", {})

        # DM 채널만 처리 (channel_type == "im")
        if event.get("channel_type") != "im":
            return

        # 봇 자신의 메시지 무시
        if event.get("bot_id"):
            return

        # 서브타입이 있는 메시지 무시 (메시지 수정, 삭제 등)
        if event.get("subtype"):
            return

        ctx = build_handler_context(body, say, client)
        processor.process(ctx)

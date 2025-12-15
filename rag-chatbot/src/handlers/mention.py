"""앱 멘션 이벤트 핸들러.

채널에서 봇을 멘션할 때 질문을 처리합니다.
"""

import logging

from slack_bolt import App

from .base import MessageProcessor, build_handler_context

logger = logging.getLogger(__name__)


class MentionProcessor(MessageProcessor):
    """멘션 메시지 처리기.

    채널에서 봇을 멘션할 때 RAG 파이프라인을 통해 답변을 생성합니다.
    """

    def _get_channel_type(self) -> str:
        """채널 타입 반환.

        Returns:
            "channel" (멘션은 항상 채널)
        """
        return "channel"

    def _get_max_messages(self) -> int:
        """최대 대화 메시지 수 반환.

        Returns:
            채널 대화 최대 메시지 수
        """
        return self.settings.conversation_max_messages

    def _get_log_prefix(self) -> str:
        """로그 접두사 반환.

        Returns:
            "멘션"
        """
        return "멘션"

    def _get_empty_message_help(self) -> str:
        """빈 메시지 안내 텍스트 반환.

        Returns:
            멘션용 안내 메시지
        """
        return "질문 내용을 입력해 주세요. 예: @봇이름 API 문서 어디서 볼 수 있나요?"


def register_mention_handlers(app: App) -> None:
    """앱 멘션 핸들러 등록.

    Args:
        app: Slack Bolt App 인스턴스
    """
    processor = MentionProcessor()

    @app.event("app_mention")
    def handle_app_mention(body, say, client) -> None:
        """앱 멘션 이벤트 처리.

        사용자가 봇을 멘션하면 RAG 파이프라인을 통해 답변을 생성합니다.

        Args:
            body: 이벤트 페이로드
            say: Slack 메시지 전송 함수
            client: Slack 클라이언트
        """
        ctx = build_handler_context(body, say, client)
        processor.process(ctx)

"""리액션 피드백 이벤트 핸들러.

사용자가 봇 답변에 리액션을 추가할 때 피드백을 수집합니다.
"""

import logging
from typing import Any

from slack_bolt import App

from ..config import get_settings
from ..models import Feedback
from ..services import get_feedback_service

logger = logging.getLogger(__name__)


def register_feedback_handlers(app: App) -> None:
    """피드백 핸들러 등록.

    Args:
        app: Slack Bolt App 인스턴스
    """

    @app.event("reaction_added")
    def handle_reaction_added(
        body: dict[str, Any],
        client: Any,
    ) -> None:
        """리액션 추가 이벤트 처리.

        봇 메시지에 리액션이 추가되면 피드백으로 저장합니다.

        Args:
            body: 이벤트 페이로드
            client: Slack 클라이언트
        """
        event = body.get("event", {})
        settings = get_settings()

        # 봇 자신의 메시지에 대한 리액션만 처리
        item_user = event.get("item_user", "")
        if not item_user:
            return

        # 봇 사용자 ID 확인 (설정에서)
        bot_user_id = settings.slack_bot_user_id
        if bot_user_id and item_user != bot_user_id:
            # 봇 ID가 설정되어 있고, 대상이 봇이 아니면 무시
            return

        # 리액션에서 rating 결정
        reaction = event.get("reaction", "")
        rating = Feedback.rating_from_reaction(reaction)

        if rating is None:
            # 지원되지 않는 리액션
            logger.debug(f"지원되지 않는 리액션 무시: {reaction}")
            return

        try:
            item = event.get("item", {})
            channel_id = item.get("channel", "")
            message_ts = item.get("ts", "")
            user_id = event.get("user", "")

            # 원본 메시지 조회 (답변 텍스트)
            result = client.conversations_history(
                channel=channel_id,
                latest=message_ts,
                inclusive=True,
                limit=1,
            )

            if not result.get("messages"):
                logger.warning(f"메시지를 찾을 수 없음: {message_ts}")
                return

            message = result["messages"][0]
            answer = message.get("text", "")
            thread_ts = message.get("thread_ts", message_ts)

            # 스레드에서 원본 질문 조회 시도
            question = _get_original_question(client, channel_id, thread_ts, message_ts)

            # Feedback 객체 생성 및 저장
            feedback = Feedback(
                message_ts=message_ts,
                thread_ts=thread_ts,
                channel_id=channel_id,
                user_id=user_id,
                question=question,
                answer=answer,
                rating=rating,
                reaction=reaction,
            )

            feedback_service = get_feedback_service()
            feedback_service.save(feedback)

            logger.info(
                f"피드백 저장 완료 - 채널: {channel_id}, "
                f"평가: {rating}, 리액션: {reaction}"
            )

        except Exception as e:
            logger.error(f"리액션 처리 실패: {e}", exc_info=True)


def _get_original_question(
    client: Any,
    channel_id: str,
    thread_ts: str,
    answer_ts: str,
) -> str:
    """스레드에서 원본 질문 조회.

    답변 직전의 사용자 메시지를 질문으로 간주합니다.

    Args:
        client: Slack 클라이언트
        channel_id: 채널 ID
        thread_ts: 스레드 타임스탬프
        answer_ts: 답변 메시지 타임스탬프

    Returns:
        원본 질문 텍스트 (찾지 못하면 빈 문자열)
    """
    try:
        # 스레드 내 메시지 조회
        result = client.conversations_replies(
            channel=channel_id,
            ts=thread_ts,
            limit=20,
        )

        messages = result.get("messages", [])
        if not messages:
            return ""

        # 답변 메시지 직전의 사용자 메시지 찾기
        for i, msg in enumerate(messages):
            if msg.get("ts") == answer_ts:
                # 이전 메시지가 있으면 반환
                if i > 0:
                    prev_msg = messages[i - 1]
                    # 봇 메시지가 아닌지 확인
                    if not prev_msg.get("bot_id"):
                        return prev_msg.get("text", "")
                break

        # 스레드 첫 메시지 반환 (멘션의 경우)
        if messages and not messages[0].get("bot_id"):
            return messages[0].get("text", "")

        return ""
    except Exception as e:
        logger.warning(f"원본 질문 조회 실패: {e}")
        return ""

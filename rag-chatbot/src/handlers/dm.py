"""DM 이벤트 핸들러.

사용자가 봇에게 직접 메시지를 보낼 때 질문을 처리합니다.
"""

import asyncio
import logging
from typing import Any

from slack_bolt import App, Say
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..guardrails import mask_sensitive_info
from ..models import Query
from ..services import get_conversation_service, get_rag_service

logger = logging.getLogger(__name__)


def register_dm_handlers(app: App) -> None:
    """DM 핸들러 등록.

    Args:
        app: Slack Bolt App 인스턴스
    """

    @app.event("message")
    def handle_dm(
        body: dict[str, Any],
        say: Say,
        client: Any,
    ) -> None:
        """DM 이벤트 처리.

        사용자가 봇에게 DM을 보내면 RAG 파이프라인을 통해 답변을 생성합니다.

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

        try:
            # DM에서는 채널 ID를 스레드 식별자로 사용
            dm_thread_ts = event.get("channel", "")
            message_ts = event.get("ts", "")

            # Query 객체 생성
            query = Query.from_slack_event(
                text=event.get("text", ""),
                user=event.get("user", ""),
                channel=event.get("channel", ""),
                ts=message_ts,
                thread_ts=None,  # DM은 스레드 없음
                channel_type="im",
            )

            # 민감 정보 확인 로깅
            masked_text = mask_sensitive_info(query.text)
            logger.info(
                f"DM 수신 - 채널: {query.channel_id}, "
                f"질문: {masked_text[:100]}..."
            )

            # 빈 질문 체크
            if not query.text.strip():
                say(text="질문 내용을 입력해 주세요. 예: API 문서 어디서 볼 수 있나요?")
                return

            # 대화 컨텍스트 로드 (DM은 채널 ID를 스레드 ID로 사용)
            conversation_service = get_conversation_service()
            conversation_context = conversation_service.get_context_summary(
                thread_ts=dm_thread_ts,
                channel_id=query.channel_id,
            )

            # 사용자 질문 저장
            conversation_service.add_message(
                thread_ts=dm_thread_ts,
                channel_id=query.channel_id,
                role="user",
                content=query.text,
                message_ts=message_ts,
            )

            # RAG 서비스 호출 (비동기 → 동기 변환)
            rag_service = get_rag_service()
            response = asyncio.run(
                rag_service.answer(query, conversation_context=conversation_context)
            )

            # 답변 전송 (DM은 스레드 없이 직접 전송)
            _send_dm_response(
                say=say,
                text=response.format_for_slack(),
            )

            # 어시스턴트 답변 저장
            conversation_service.add_message(
                thread_ts=dm_thread_ts,
                channel_id=query.channel_id,
                role="assistant",
                content=response.text,
                message_ts=message_ts,
            )

            logger.info(
                f"DM 답변 완료 - 채널: {query.channel_id}, "
                f"토큰: {response.tokens_used}, "
                f"시간: {response.generation_time_ms}ms"
            )

        except Exception as e:
            logger.error(f"DM 처리 실패: {e}", exc_info=True)
            _send_dm_error_response(say)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception),
)
def _send_dm_response(
    say: Say,
    text: str,
) -> None:
    """DM 응답 전송 (재시도 로직 포함).

    Args:
        say: Slack 메시지 전송 함수
        text: 전송할 텍스트
    """
    say(text=text)


def _send_dm_error_response(say: Say) -> None:
    """DM 에러 응답 전송.

    Args:
        say: Slack 메시지 전송 함수
    """
    error_message = (
        "죄송합니다, 답변 생성 중 오류가 발생했습니다.\n"
        "잠시 후 다시 시도해 주세요."
    )
    try:
        say(text=error_message)
    except Exception as e:
        logger.error(f"DM 에러 응답 전송 실패: {e}")

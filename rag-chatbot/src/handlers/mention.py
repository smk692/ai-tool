"""앱 멘션 이벤트 핸들러.

채널에서 봇을 멘션할 때 질문을 처리합니다.
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


def register_mention_handlers(app: App) -> None:
    """앱 멘션 핸들러 등록.

    Args:
        app: Slack Bolt App 인스턴스
    """

    @app.event("app_mention")
    def handle_app_mention(
        body: dict[str, Any],
        say: Say,
        client: Any,
    ) -> None:
        """앱 멘션 이벤트 처리.

        사용자가 봇을 멘션하면 RAG 파이프라인을 통해 답변을 생성합니다.

        Args:
            body: 이벤트 페이로드
            say: Slack 메시지 전송 함수
            client: Slack 클라이언트
        """
        event = body.get("event", {})

        try:
            # Query 객체 생성
            query = Query.from_slack_event(
                text=event.get("text", ""),
                user=event.get("user", ""),
                channel=event.get("channel", ""),
                ts=event.get("ts", ""),
                thread_ts=event.get("thread_ts"),
                channel_type="channel",  # 채널 멘션
            )

            # 민감 정보 확인 로깅
            masked_text = mask_sensitive_info(query.text)
            logger.info(
                f"멘션 수신 - 채널: {query.channel_id}, "
                f"스레드: {query.thread_ts}, "
                f"질문: {masked_text[:100]}..."
            )

            # 빈 질문 체크
            if not query.text.strip():
                say(
                    text="질문 내용을 입력해 주세요. 예: @봇이름 API 문서 어디서 볼 수 있나요?",
                    thread_ts=query.thread_ts,
                )
                return

            # 대화 컨텍스트 로드
            conversation_service = get_conversation_service()
            conversation_context = conversation_service.get_context_summary(
                thread_ts=query.thread_ts,
                channel_id=query.channel_id,
            )

            # 사용자 질문 저장
            conversation_service.add_message(
                thread_ts=query.thread_ts,
                channel_id=query.channel_id,
                role="user",
                content=query.text,
                message_ts=event.get("ts", ""),
            )

            # RAG 서비스 호출 (비동기 → 동기 변환)
            rag_service = get_rag_service()
            response = asyncio.run(
                rag_service.answer(query, conversation_context=conversation_context)
            )

            # 답변 전송
            _send_response(
                say=say,
                text=response.format_for_slack(),
                thread_ts=query.thread_ts,
            )

            # 어시스턴트 답변 저장
            conversation_service.add_message(
                thread_ts=query.thread_ts,
                channel_id=query.channel_id,
                role="assistant",
                content=response.text,
                message_ts=query.thread_ts,  # 답변 ts는 스레드 ts 사용
            )

            logger.info(
                f"답변 완료 - 스레드: {query.thread_ts}, "
                f"토큰: {response.tokens_used}, "
                f"시간: {response.generation_time_ms}ms"
            )

        except Exception as e:
            logger.error(f"멘션 처리 실패: {e}", exc_info=True)
            thread_ts = event.get("thread_ts") or event.get("ts", "")
            _send_error_response(say, thread_ts)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception),
)
def _send_response(
    say: Say,
    text: str,
    thread_ts: str,
) -> None:
    """Slack에 응답 전송 (재시도 로직 포함).

    Args:
        say: Slack 메시지 전송 함수
        text: 전송할 텍스트
        thread_ts: 스레드 타임스탬프
    """
    say(text=text, thread_ts=thread_ts)


def _send_error_response(
    say: Say,
    thread_ts: str,
) -> None:
    """에러 응답 전송.

    Args:
        say: Slack 메시지 전송 함수
        thread_ts: 스레드 타임스탬프
    """
    error_message = (
        "⚠️ 죄송합니다, 답변 생성 중 오류가 발생했습니다.\n"
        "잠시 후 다시 시도해 주세요."
    )
    try:
        say(text=error_message, thread_ts=thread_ts)
    except Exception as e:
        logger.error(f"에러 응답 전송 실패: {e}")

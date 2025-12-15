"""메시지 처리 기본 클래스.

핸들러 공통 로직을 추상화한 MessageProcessor 기반 클래스를 제공합니다.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from slack_bolt import Say
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..config import get_settings
from ..guardrails import mask_sensitive_info
from ..models import ImageContent, Query
from ..services import get_conversation_service, get_image_processor, get_rag_service
from ..utils import split_message_for_slack

logger = logging.getLogger(__name__)


@dataclass
class HandlerContext:
    """핸들러 컨텍스트.

    이벤트 처리에 필요한 모든 정보를 담는 데이터 클래스입니다.

    Attributes:
        event: Slack 이벤트 페이로드
        client: Slack 클라이언트
        say: Slack 메시지 전송 함수
        channel_id: 채널 ID
        message_ts: 메시지 타임스탬프
        thread_ts: 스레드 타임스탬프
        user_id: 사용자 ID
        text: 메시지 텍스트
        files: Slack 파일 배열
        channel_type: 채널 타입 (im, channel)
    """

    event: dict[str, Any]
    client: Any
    say: Say
    channel_id: str
    message_ts: str
    thread_ts: str
    user_id: str
    text: str
    files: list[dict[str, Any]] = field(default_factory=list)
    channel_type: str = "channel"


class MessageProcessor(ABC):
    """메시지 처리기 기반 클래스.

    DM과 멘션 핸들러의 공통 로직을 추상화합니다.
    Template Method 패턴을 사용하여 핸들러별 커스텀 동작을 지원합니다.
    """

    def __init__(self) -> None:
        """메시지 처리기 초기화."""
        self.settings = get_settings()

    def process(self, ctx: HandlerContext) -> None:
        """메시지 처리 메인 로직.

        Template Method 패턴으로 구현되어 있으며,
        _on_start, _on_success, _on_error 훅을 통해 커스텀 동작을 지원합니다.

        Args:
            ctx: 핸들러 컨텍스트
        """
        try:
            # 1. 처리 시작 훅 (DM: 리액션 추가)
            self._on_start(ctx)

            # 2. Query 객체 생성
            query = Query.from_slack_event(
                text=ctx.text,
                user=ctx.user_id,
                channel=ctx.channel_id,
                ts=ctx.message_ts,
                thread_ts=ctx.thread_ts,
                channel_type=self._get_channel_type(),
                files=ctx.files,
            )

            # 3. 빈 질문 + 이미지 없음 체크
            if not query.text.strip() and not query.has_images:
                self._on_empty_message(ctx)
                return

            # 4. 민감 정보 확인 로깅
            masked_text = mask_sensitive_info(query.text)
            image_count = len(
                [f for f in ctx.files if f.get("mimetype", "").startswith("image/")]
            )
            logger.info(
                f"{self._get_log_prefix()} 수신 - 채널: {ctx.channel_id}, "
                f"스레드: {ctx.thread_ts}, "
                f"질문: {masked_text[:100]}..., "
                f"이미지: {image_count}개"
            )

            # 5. 이미지 처리 (있는 경우)
            images = None
            if query.has_images:
                images = asyncio.run(self._process_images(ctx))

            # 6. 대화 컨텍스트 로드
            conversation_service = get_conversation_service()
            conversation_context = conversation_service.get_context_summary(
                thread_ts=query.thread_ts,
                channel_id=query.channel_id,
            )

            # 7. 사용자 질문 저장
            conversation_service.add_message(
                thread_ts=query.thread_ts,
                channel_id=query.channel_id,
                role="user",
                content=query.text,
                message_ts=ctx.message_ts,
                max_messages=self._get_max_messages(),
            )

            # 8. RAG 서비스 호출 (이미지 포함)
            rag_service = get_rag_service()
            response = asyncio.run(
                rag_service.answer(
                    query,
                    conversation_context=conversation_context,
                    images=images,
                )
            )

            # 9. 답변 전송
            self._send_response(
                say=ctx.say,
                text=response.format_for_slack(),
                thread_ts=self._get_response_thread_ts(ctx),
            )

            # 10. 어시스턴트 답변 저장
            conversation_service.add_message(
                thread_ts=query.thread_ts,
                channel_id=query.channel_id,
                role="assistant",
                content=response.text,
                message_ts=query.thread_ts,
                max_messages=self._get_max_messages(),
            )

            # 11. 처리 완료 훅 (DM: 완료 리액션)
            self._on_success(ctx)

            # 12. 완료 로깅
            logger.info(
                f"{self._get_log_prefix()} 답변 완료 - "
                f"스레드: {ctx.thread_ts}, "
                f"토큰: {response.tokens_used}, "
                f"시간: {response.generation_time_ms}ms"
            )

        except Exception as e:
            logger.error(f"{self._get_log_prefix()} 처리 실패: {e}", exc_info=True)
            self._on_error(ctx, e)

    @abstractmethod
    def _get_channel_type(self) -> str:
        """채널 타입 반환.

        Returns:
            채널 타입 (im 또는 channel)
        """

    @abstractmethod
    def _get_max_messages(self) -> int:
        """최대 대화 메시지 수 반환.

        Returns:
            최대 메시지 수
        """

    @abstractmethod
    def _get_log_prefix(self) -> str:
        """로그 접두사 반환.

        Returns:
            로그 접두사 (예: "DM", "멘션")
        """

    def _get_response_thread_ts(self, ctx: HandlerContext) -> str:
        """응답을 보낼 스레드 타임스탬프 반환.

        기본적으로 thread_ts를 사용하지만, 하위 클래스에서 오버라이드 가능.

        Args:
            ctx: 핸들러 컨텍스트

        Returns:
            스레드 타임스탬프
        """
        return ctx.thread_ts

    def _on_start(self, ctx: HandlerContext) -> None:
        """처리 시작 훅.

        하위 클래스에서 오버라이드하여 처리 시작 시 동작을 정의합니다.
        예: DM에서 눈 리액션 추가

        Args:
            ctx: 핸들러 컨텍스트
        """

    def _on_success(self, ctx: HandlerContext) -> None:
        """처리 성공 훅.

        하위 클래스에서 오버라이드하여 처리 성공 시 동작을 정의합니다.
        예: DM에서 체크마크 리액션 추가

        Args:
            ctx: 핸들러 컨텍스트
        """

    def _on_error(self, ctx: HandlerContext, error: Exception) -> None:
        """처리 실패 훅.

        하위 클래스에서 오버라이드하여 처리 실패 시 동작을 정의합니다.

        Args:
            ctx: 핸들러 컨텍스트
            error: 발생한 예외
        """
        thread_ts = ctx.thread_ts or ctx.message_ts
        self._send_error_response(ctx.say, thread_ts)

    def _on_empty_message(self, ctx: HandlerContext) -> None:
        """빈 메시지 처리 훅.

        하위 클래스에서 오버라이드하여 빈 메시지 처리 동작을 정의합니다.

        Args:
            ctx: 핸들러 컨텍스트
        """
        help_message = self._get_empty_message_help()
        ctx.say(text=help_message, thread_ts=self._get_response_thread_ts(ctx))

    def _get_empty_message_help(self) -> str:
        """빈 메시지 안내 텍스트 반환.

        Returns:
            안내 메시지
        """
        return "질문 내용을 입력해 주세요. 예: API 문서 어디서 볼 수 있나요?"

    async def _process_images(self, ctx: HandlerContext) -> list[ImageContent] | None:
        """이미지 처리.

        Slack 파일을 다운로드하여 Claude Vision 형식으로 변환합니다.

        Args:
            ctx: 핸들러 컨텍스트

        Returns:
            ImageContent 리스트 또는 None
        """
        if not ctx.files or not self.settings.image_processing_enabled:
            return None

        try:
            processor = get_image_processor()
            images = await processor.process_slack_files(
                files=ctx.files,
                bot_token=self.settings.slack_bot_token,
            )
            return images if images else None
        except Exception as e:
            logger.warning(f"이미지 처리 중 오류 발생: {e}")
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def _send_response(
        self,
        say: Say,
        text: str,
        thread_ts: str,
    ) -> None:
        """Slack에 응답 전송 (재시도 로직 포함).

        긴 메시지는 여러 청크로 분할하여 전송합니다.
        코드 블록 경계를 안전하게 처리합니다.

        Args:
            say: Slack 메시지 전송 함수
            text: 전송할 텍스트
            thread_ts: 스레드 타임스탬프
        """
        chunks = split_message_for_slack(text)

        for chunk in chunks:
            say(text=chunk, thread_ts=thread_ts)

    def _send_error_response(
        self,
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


def build_handler_context(
    body: dict[str, Any],
    say: Say,
    client: Any,
) -> HandlerContext:
    """핸들러 컨텍스트 빌드.

    Slack 이벤트 페이로드에서 HandlerContext를 생성합니다.

    Args:
        body: 이벤트 페이로드
        say: Slack 메시지 전송 함수
        client: Slack 클라이언트

    Returns:
        HandlerContext 인스턴스
    """
    event = body.get("event", {})
    message_ts = event.get("ts", "")
    channel_id = event.get("channel", "")
    thread_ts = event.get("thread_ts") or message_ts

    return HandlerContext(
        event=event,
        client=client,
        say=say,
        channel_id=channel_id,
        message_ts=message_ts,
        thread_ts=thread_ts,
        user_id=event.get("user", ""),
        text=event.get("text", ""),
        files=event.get("files", []),
        channel_type=event.get("channel_type", "channel"),
    )

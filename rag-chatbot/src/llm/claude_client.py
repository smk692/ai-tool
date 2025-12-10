"""Claude Agent SDK 클라이언트 래퍼.

Claude Agent SDK를 사용하여 LLM 응답을 생성합니다.
"""

import time

import anyio
import structlog
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import get_settings
from ..models import Response, SearchResult, SourceReference
from .prompts import build_rag_prompt

logger = structlog.get_logger(__name__)


class ClaudeClient:
    """Claude Agent SDK 클라이언트.

    RAG 컨텍스트를 포함한 질문에 대해 Claude API를 통해 답변을 생성합니다.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4000,
    ):
        """ClaudeClient 초기화.

        Args:
            model: 사용할 Claude 모델
            max_tokens: 최대 응답 토큰 수
        """
        self.model = model
        self.max_tokens = max_tokens
        self._settings = get_settings()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def generate_response(
        self,
        question: str,
        search_results: list[SearchResult],
        conversation_context: str | None = None,
    ) -> Response:
        """RAG 컨텍스트 기반 답변 생성.

        Args:
            question: 사용자 질문
            search_results: 벡터DB 검색 결과 목록
            conversation_context: 이전 대화 컨텍스트 (선택)

        Returns:
            Response 객체
        """
        start_time = time.perf_counter()

        # 프롬프트 생성
        prompt = build_rag_prompt(
            question=question,
            search_results=search_results,
            conversation_context=conversation_context,
        )

        logger.info(
            "Claude API 호출 시작",
            question_length=len(question),
            context_count=len(search_results),
            has_conversation=conversation_context is not None,
        )

        try:
            # Claude Agent SDK 옵션 설정
            options = ClaudeAgentOptions(
                system_prompt=self._get_system_prompt(),
                max_turns=1,
                allowed_tools=[],  # RAG 응답에는 도구 사용 안함
            )

            # 응답 수집
            response_text = ""
            tokens_used = 0

            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            response_text += block.text
                elif isinstance(message, ResultMessage):
                    tokens_used = getattr(message, "input_tokens", 0) + getattr(
                        message, "output_tokens", 0
                    )

            # 응답 텍스트 정리 (Slack 메시지 제한 4000자)
            if len(response_text) > 4000:
                response_text = response_text[:3950] + "\n\n...(답변이 잘렸습니다)"

            generation_time_ms = int((time.perf_counter() - start_time) * 1000)

            # 소스 참조 생성
            sources = self._extract_sources(search_results)

            logger.info(
                "Claude API 호출 완료",
                response_length=len(response_text),
                tokens_used=tokens_used,
                generation_time_ms=generation_time_ms,
            )

            return Response(
                text=response_text,
                sources=sources,
                model=self.model,
                tokens_used=tokens_used,
                generation_time_ms=generation_time_ms,
            )

        except Exception as e:
            logger.error("Claude API 호출 실패", error=str(e))
            raise

    def _get_system_prompt(self) -> str:
        """시스템 프롬프트 반환."""
        return (
            "당신은 회사 내부 문서와 API 스펙을 기반으로 "
            "질문에 답변하는 도움이 되는 어시스턴트입니다.\n\n"
            "핵심 원칙:\n"
            "1. 제공된 컨텍스트 정보만을 기반으로 정확하게 답변합니다.\n"
            "2. 컨텍스트에 없는 정보는 추측하지 않고, 모른다고 솔직하게 말합니다.\n"
            "3. 기술적 질문에는 구체적이고 실용적인 답변을 제공합니다.\n"
            "4. 한국어로 자연스럽게 답변합니다.\n"
            "5. 답변은 간결하면서도 필요한 정보를 빠뜨리지 않습니다.\n\n"
            "응답 형식:\n"
            "- Slack 메시지로 표시되므로 마크다운 형식을 사용합니다.\n"
            "- 코드가 있으면 코드 블록으로 감싸줍니다.\n"
            "- 중요한 정보는 *굵게* 표시합니다."
        )

    def _extract_sources(
        self, search_results: list[SearchResult]
    ) -> list[SourceReference]:
        """검색 결과에서 소스 참조 추출.

        중복 제거하여 고유한 소스만 반환합니다.

        Args:
            search_results: 검색 결과 목록

        Returns:
            SourceReference 목록
        """
        seen_ids: set[str] = set()
        sources: list[SourceReference] = []

        for result in search_results:
            if result.source_id not in seen_ids:
                seen_ids.add(result.source_id)
                sources.append(
                    SourceReference(
                        title=result.source_title,
                        url=result.source_url,
                        source_type=result.source_type,
                    )
                )

        return sources


# 동기 래퍼 함수 (Slack 핸들러에서 사용)
def generate_response_sync(
    client: ClaudeClient,
    question: str,
    search_results: list[SearchResult],
    conversation_context: str | None = None,
) -> Response:
    """동기 방식으로 응답 생성.

    Args:
        client: ClaudeClient 인스턴스
        question: 사용자 질문
        search_results: 검색 결과 목록
        conversation_context: 대화 컨텍스트

    Returns:
        Response 객체
    """
    return anyio.from_thread.run(
        client.generate_response,
        question,
        search_results,
        conversation_context,
    )

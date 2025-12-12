"""Response 모델 정의.

Claude LLM이 생성한 답변을 나타냅니다.
"""

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class SourceReference(BaseModel):
    """참조 문서 정보.

    답변 생성에 사용된 소스 문서 참조 정보입니다.

    Attributes:
        title: 문서 제목
        url: 문서 URL (있는 경우)
        source_type: 소스 타입 (notion, swagger)
    """

    title: str = Field(..., description="문서 제목")
    url: str | None = Field(default=None, description="문서 URL")
    source_type: Literal["notion", "swagger"] = Field(..., description="소스 타입")


class Response(BaseModel):
    """LLM 생성 답변 모델.

    Claude API를 통해 생성된 답변과 메타데이터를 저장합니다.

    Attributes:
        text: 답변 텍스트
        sources: 참조 문서 목록
        model: 사용된 LLM 모델
        tokens_used: 사용된 토큰 수
        generation_time_ms: 생성 시간 (밀리초)
        created_at: 생성 시간 (UTC)
        is_fallback: 폴백 응답 여부
    """

    text: str = Field(..., min_length=1, max_length=4000, description="답변 텍스트")
    sources: list[SourceReference] = Field(
        default_factory=list, description="참조 문서 목록"
    )
    model: str = Field(default="claude-sonnet-4-20250514", description="사용된 모델")
    tokens_used: int = Field(default=0, ge=0, description="사용된 토큰 수")
    generation_time_ms: int = Field(default=0, ge=0, description="생성 시간 (ms)")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="생성 시간"
    )
    is_fallback: bool = Field(default=False, description="폴백 응답 여부")

    def format_for_slack(self, max_length: int = 3900) -> str:
        """Slack 메시지 형식으로 포맷팅.

        답변 텍스트와 참조 문서 목록을 Slack mrkdwn 형식으로 변환합니다.
        Slack 메시지 제한(4000자)을 고려하여 자동으로 자릅니다.

        Args:
            max_length: 최대 메시지 길이 (기본 3900, 안전 마진 포함)

        Returns:
            Slack 메시지 형식의 문자열
        """
        # 소스 참조 섹션 먼저 생성
        sources_section = ""
        if self.sources:
            sources_section = "\n\n📚 *참조 문서:*\n"
            for src in self.sources:
                if src.url:
                    # Notion URL을 클릭 가능한 Slack 링크로 포맷
                    sources_section += f"• <{src.url}|{src.title}>\n"
                else:
                    sources_section += f"• {src.title}\n"

        # 텍스트와 소스 합쳐서 길이 확인
        sources_length = len(sources_section)
        available_for_text = max_length - sources_length

        # 텍스트 자르기 (필요 시)
        text = self.text
        if len(text) > available_for_text:
            # 말줄임표와 함께 자르기
            truncation_notice = "\n\n_(답변이 길어 일부가 생략되었습니다)_"
            text = text[: available_for_text - len(truncation_notice)] + truncation_notice

        return text + sources_section

    def format_for_slack_split(self, max_length: int = 3900) -> list[str]:
        """긴 답변을 여러 Slack 메시지로 분할.

        4000자를 초과하는 긴 답변을 여러 메시지로 분할합니다.
        마지막 메시지에 소스 참조를 포함합니다.

        Args:
            max_length: 각 메시지의 최대 길이

        Returns:
            분할된 메시지 목록
        """
        # 소스 참조 섹션 생성
        sources_section = ""
        if self.sources:
            sources_section = "\n\n📚 *참조 문서:*\n"
            for src in self.sources:
                if src.url:
                    sources_section += f"• <{src.url}|{src.title}>\n"
                else:
                    sources_section += f"• {src.title}\n"

        messages: list[str] = []
        text = self.text

        # 텍스트가 단일 메시지에 들어가는 경우
        if len(text) + len(sources_section) <= max_length:
            return [text + sources_section]

        # 분할 필요
        # 마지막 메시지에 sources_section을 위한 공간 확보
        continuation_notice = "\n\n_(계속...)_"

        while text:
            if len(text) + len(sources_section) <= max_length:
                # 마지막 조각
                messages.append(text + sources_section)
                break
            else:
                # 중간 조각
                chunk_size = max_length - len(continuation_notice)
                chunk = text[:chunk_size] + continuation_notice
                messages.append(chunk)
                text = text[chunk_size:]

        return messages

    @classmethod
    def fallback_response(cls, reason: str = "검색 결과가 없습니다.") -> "Response":
        """폴백 응답 생성.

        검색 결과가 없거나 오류 발생 시 사용되는 기본 응답입니다.

        Args:
            reason: 폴백 사유

        Returns:
            폴백 Response 인스턴스
        """
        return cls(
            text=f"죄송합니다. {reason} 다른 질문을 시도해 주세요.",
            sources=[],
            is_fallback=True,
        )

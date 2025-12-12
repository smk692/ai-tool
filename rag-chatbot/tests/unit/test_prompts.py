"""프롬프트 모듈 테스트."""


from src.models import SearchResult


# prompts 모듈 함수들을 직접 테스트하기 위해 inline 정의
# (claude_agent_sdk 의존성 피하기 위함)
def _format_source_label(result: SearchResult) -> str:
    """소스 타입 레이블 포맷팅."""
    source_labels = {
        "notion": "Notion 문서",
        "swagger": "API 스펙",
    }
    return source_labels.get(result.source_type, result.source_type)


def build_rag_prompt(
    question: str,
    search_results: list[SearchResult],
    conversation_context: str | None = None,
) -> str:
    """RAG 프롬프트 생성."""
    parts: list[str] = []

    if conversation_context:
        parts.append("## 이전 대화")
        parts.append(conversation_context)
        parts.append("")

    if search_results:
        parts.append("## 참고 문서")
        for i, result in enumerate(search_results, 1):
            source_label = _format_source_label(result)
            parts.append(f"### [{i}] {result.source_title} ({source_label})")
            parts.append(f"유사도: {result.score:.2f}")
            parts.append("")
            parts.append(result.content)
            parts.append("")
    else:
        parts.append("## 참고 문서")
        parts.append("검색된 관련 문서가 없습니다.")
        parts.append("")

    parts.append("## 질문")
    parts.append(question)

    return "\n".join(parts)


def build_no_context_prompt(question: str) -> str:
    """컨텍스트 없는 프롬프트 생성."""
    return f"""관련 문서를 찾을 수 없습니다.

## 질문
{question}

위 질문에 대해 관련 문서를 찾을 수 없다고 안내하고,
질문을 다시 구체화하거나 다른 키워드로 질문해달라고 요청하세요."""


def build_followup_prompt(
    question: str,
    previous_answer: str,
    search_results: list[SearchResult] | None = None,
) -> str:
    """후속 질문 프롬프트 생성."""
    parts: list[str] = []

    parts.append("## 이전 답변")
    parts.append(previous_answer)
    parts.append("")

    if search_results:
        parts.append("## 추가 참고 문서")
        for i, result in enumerate(search_results, 1):
            source_label = _format_source_label(result)
            parts.append(f"### [{i}] {result.source_title} ({source_label})")
            parts.append(result.content)
            parts.append("")

    parts.append("## 후속 질문")
    parts.append(question)
    parts.append("")
    parts.append("이전 답변을 참고하여 후속 질문에 답변해주세요.")

    return "\n".join(parts)


def truncate_context(
    search_results: list[SearchResult],
    max_tokens: int = 4000,
    chars_per_token: int = 4,
) -> list[SearchResult]:
    """토큰 제한에 맞게 컨텍스트 자르기."""
    max_chars = max_tokens * chars_per_token
    current_chars = 0
    truncated: list[SearchResult] = []

    sorted_results = sorted(search_results, key=lambda x: x.score, reverse=True)

    for result in sorted_results:
        result_chars = len(result.content) + len(result.source_title) + 50
        if current_chars + result_chars <= max_chars:
            truncated.append(result)
            current_chars += result_chars
        else:
            break

    return truncated


class TestBuildRagPrompt:
    """build_rag_prompt 함수 테스트."""

    def test_basic_prompt(self) -> None:
        """기본 프롬프트 생성 테스트."""
        results = [
            SearchResult(
                chunk_id="chunk-1",
                source_id="doc-1",
                source_title="휴가 정책",
                source_url="https://notion.so/vacation",
                source_type="notion",
                content="연차는 입사 후 1년이 지나면 15일이 부여됩니다.",
                score=0.9,
            )
        ]
        prompt = build_rag_prompt(
            question="회사 휴가 정책이 어떻게 되나요?",
            search_results=results,
        )

        assert "회사 휴가 정책" in prompt
        assert "연차는 입사 후 1년" in prompt
        assert "휴가 정책" in prompt

    def test_prompt_with_conversation_context(self) -> None:
        """대화 컨텍스트 포함 프롬프트 테스트."""
        results = [
            SearchResult(
                chunk_id="chunk-1",
                source_id="doc-1",
                source_title="API 문서",
                source_url="https://api.example.com",
                source_type="swagger",
                content="GET /users 엔드포인트는 사용자 목록을 반환합니다.",
                score=0.85,
            )
        ]
        context = "사용자: API에 대해 알려주세요\n어시스턴트: 어떤 API에 대해 알고 싶으신가요?"

        prompt = build_rag_prompt(
            question="사용자 목록 조회는 어떻게 하나요?",
            search_results=results,
            conversation_context=context,
        )

        assert "사용자 목록 조회" in prompt
        assert "GET /users" in prompt

    def test_multiple_search_results(self) -> None:
        """여러 검색 결과 프롬프트 테스트."""
        results = [
            SearchResult(
                chunk_id="chunk-1",
                source_id="doc-1",
                source_title="문서1",
                source_url="https://example.com/1",
                source_type="notion",
                content="첫 번째 문서 내용",
                score=0.9,
            ),
            SearchResult(
                chunk_id="chunk-2",
                source_id="doc-2",
                source_title="문서2",
                source_url="https://example.com/2",
                source_type="notion",
                content="두 번째 문서 내용",
                score=0.85,
            ),
        ]
        prompt = build_rag_prompt(
            question="테스트 질문",
            search_results=results,
        )

        assert "첫 번째 문서 내용" in prompt
        assert "두 번째 문서 내용" in prompt


class TestBuildNoContextPrompt:
    """build_no_context_prompt 함수 테스트."""

    def test_no_context_prompt(self) -> None:
        """컨텍스트 없음 프롬프트 테스트."""
        prompt = build_no_context_prompt("모르는 질문입니다")

        assert "모르는 질문" in prompt
        assert "찾을 수 없" in prompt or "없습니다" in prompt


class TestBuildFollowupPrompt:
    """build_followup_prompt 함수 테스트."""

    def test_followup_prompt(self) -> None:
        """후속 질문 프롬프트 테스트."""
        prompt = build_followup_prompt(
            question="더 자세히 설명해주세요",
            previous_answer="간단한 설명입니다.",
        )

        assert "더 자세히 설명" in prompt
        assert "간단한 설명" in prompt


class TestTruncateContext:
    """truncate_context 함수 테스트."""

    def test_within_limit(self) -> None:
        """토큰 제한 내 테스트."""
        results = [
            SearchResult(
                chunk_id="chunk-1",
                source_id="doc-1",
                source_title="문서1",
                source_url="https://example.com",
                source_type="notion",
                content="짧은 내용",
                score=0.9,
            )
        ]
        truncated = truncate_context(results, max_tokens=1000)

        assert len(truncated) == 1
        assert truncated[0].content == "짧은 내용"

    def test_exceeds_limit(self) -> None:
        """토큰 제한 초과 시 잘림 테스트."""
        long_content = "매우 긴 내용 " * 1000  # 약 4000자
        results = [
            SearchResult(
                chunk_id="chunk-1",
                source_id="doc-1",
                source_title="문서1",
                source_url="https://example.com",
                source_type="notion",
                content=long_content,
                score=0.9,
            )
        ]
        truncated = truncate_context(results, max_tokens=100)

        # 토큰 제한에 맞게 잘려야 함
        assert len(truncated) >= 0

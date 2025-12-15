"""RAG 프롬프트 템플릿.

Claude API 호출에 사용되는 프롬프트를 생성합니다.
"""

from ..models import SearchResult


def build_rag_prompt(
    question: str,
    search_results: list[SearchResult],
    conversation_context: str | None = None,
) -> str:
    """RAG 프롬프트 생성.

    검색 결과와 대화 컨텍스트를 포함한 프롬프트를 구성합니다.

    Args:
        question: 사용자 질문
        search_results: 벡터DB 검색 결과 목록
        conversation_context: 이전 대화 컨텍스트 (선택)

    Returns:
        완성된 프롬프트 문자열
    """
    parts: list[str] = []

    # 대화 컨텍스트 추가
    if conversation_context:
        parts.append("## 이전 대화")
        parts.append(conversation_context)
        parts.append("")

    # 검색 결과 추가
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
        parts.append("벡터DB에서 검색된 관련 문서가 없습니다.")
        parts.append("")
        parts.append("**MCP 도구를 사용하여 실시간으로 정보를 조회하세요.**")
        parts.append("- API 관련 질문: Swagger MCP 도구 사용")
        parts.append("- 모니터링/메트릭 질문: Grafana MCP 도구 사용")
        parts.append("- 에러/이슈 질문: Sentry MCP 도구 사용")
        parts.append("- AWS 인프라 질문: AWS MCP 도구 사용")
        parts.append("")

    # 질문 추가
    parts.append("## 질문")
    parts.append(question)

    return "\n".join(parts)


def build_no_context_prompt(question: str) -> str:
    """컨텍스트 없는 프롬프트 생성.

    검색 결과가 없을 때 사용되는 프롬프트입니다.

    Args:
        question: 사용자 질문

    Returns:
        프롬프트 문자열
    """
    return f"""관련 문서를 찾을 수 없습니다.

## 질문
{question}

위 질문에 대해 관련 문서를 찾을 수 없다고 안내하고,
질문을 다시 구체화하거나 다른 키워드로 질문해달라고 요청하세요."""


def build_followup_prompt(
    question: str,
    previous_answer: str,
    search_results: list[SearchResult],
) -> str:
    """후속 질문 프롬프트 생성.

    이전 답변을 참조하는 후속 질문에 대한 프롬프트입니다.

    Args:
        question: 후속 질문
        previous_answer: 이전 답변
        search_results: 새로운 검색 결과

    Returns:
        프롬프트 문자열
    """
    parts: list[str] = []

    # 이전 답변
    parts.append("## 이전 답변")
    parts.append(previous_answer)
    parts.append("")

    # 새 검색 결과
    if search_results:
        parts.append("## 추가 참고 문서")
        for i, result in enumerate(search_results, 1):
            source_label = _format_source_label(result)
            parts.append(f"### [{i}] {result.source_title} ({source_label})")
            parts.append(result.content)
            parts.append("")

    # 후속 질문
    parts.append("## 후속 질문")
    parts.append(question)
    parts.append("")
    parts.append("이전 답변을 참고하여 후속 질문에 답변해주세요.")

    return "\n".join(parts)


def _format_source_label(result: SearchResult) -> str:
    """소스 타입 레이블 포맷팅.

    Args:
        result: 검색 결과

    Returns:
        포맷팅된 소스 레이블
    """
    source_labels = {
        "notion": "Notion 문서",
        "swagger": "API 스펙",
    }
    return source_labels.get(result.source_type, result.source_type)


def truncate_context(
    search_results: list[SearchResult],
    max_tokens: int = 4000,
    chars_per_token: int = 4,
) -> list[SearchResult]:
    """토큰 제한에 맞게 컨텍스트 자르기.

    검색 결과를 유사도 순으로 정렬 후, 토큰 제한 내에서 최대한 포함합니다.

    Args:
        search_results: 검색 결과 목록
        max_tokens: 최대 토큰 수
        chars_per_token: 토큰당 예상 문자 수

    Returns:
        잘린 검색 결과 목록
    """
    max_chars = max_tokens * chars_per_token
    current_chars = 0
    truncated: list[SearchResult] = []

    # 유사도 높은 순으로 정렬
    sorted_results = sorted(search_results, key=lambda x: x.score, reverse=True)

    for result in sorted_results:
        result_chars = len(result.content) + len(result.source_title) + 50  # 메타데이터
        if current_chars + result_chars <= max_chars:
            truncated.append(result)
            current_chars += result_chars
        else:
            break

    return truncated

"""성능 관련 테스트.

응답 시간 10초 이내 요구사항을 검증합니다.
"""

import time

from src.llm.prompts import build_rag_prompt, truncate_context
from src.models import Response, SearchResult, SourceReference


class TestPromptBuildingPerformance:
    """프롬프트 생성 성능 테스트."""

    def test_build_rag_prompt_performance(self) -> None:
        """프롬프트 생성이 100ms 이내에 완료되어야 함."""
        # 다수의 검색 결과로 테스트
        results = [
            SearchResult(
                chunk_id=f"chunk-{i}",
                source_id=f"doc-{i}",
                source_title=f"문서 제목 {i}",
                source_url=f"https://example.com/doc-{i}",
                source_type="notion",
                content=f"이것은 테스트 문서 {i}의 내용입니다. " * 50,  # 약 1KB
                score=max(0.1, 0.9 - (i * 0.04)),  # 항상 0 이상 유지
            )
            for i in range(20)  # 20개 검색 결과
        ]

        start = time.perf_counter()
        prompt = build_rag_prompt(
            question="회사 정책에 대해 알려주세요",
            search_results=results,
            conversation_context="이전 대화 내용입니다.",
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100, f"프롬프트 생성이 {elapsed_ms:.2f}ms 걸림 (100ms 이내여야 함)"
        assert len(prompt) > 0


class TestContextTruncationPerformance:
    """컨텍스트 자르기 성능 테스트."""

    def test_truncate_context_performance(self) -> None:
        """컨텍스트 자르기가 50ms 이내에 완료되어야 함."""
        # 대용량 컨텍스트로 테스트
        results = [
            SearchResult(
                chunk_id=f"chunk-{i}",
                source_id=f"doc-{i}",
                source_title=f"문서 {i}",
                source_url=f"https://example.com/{i}",
                source_type="notion",
                content="테스트 내용 " * 500,  # 약 5KB 각각
                score=0.9 - (i * 0.01),
            )
            for i in range(50)  # 50개 결과
        ]

        start = time.perf_counter()
        truncated = truncate_context(results, max_tokens=4000)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50, f"컨텍스트 자르기가 {elapsed_ms:.2f}ms 걸림 (50ms 이내여야 함)"
        # 결과가 토큰 제한 내로 잘렸는지 확인
        total_chars = sum(len(r.content) + len(r.source_title) + 50 for r in truncated)
        assert total_chars <= 4000 * 4  # chars_per_token = 4


class TestResponseFormattingPerformance:
    """응답 포맷팅 성능 테스트."""

    def test_format_for_slack_performance(self) -> None:
        """Slack 포맷팅이 20ms 이내에 완료되어야 함."""
        # 긴 응답과 다수의 소스로 테스트
        response = Response(
            text="답변 내용입니다. " * 200,  # 약 4KB
            sources=[
                SourceReference(
                    title=f"문서 {i}",
                    url=f"https://example.com/doc-{i}",
                    source_type="notion",
                )
                for i in range(10)
            ],
            model="claude-sonnet-4-20250514",
            tokens_used=500,
            generation_time_ms=1000,
        )

        start = time.perf_counter()
        formatted = response.format_for_slack()
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 20, f"Slack 포맷팅이 {elapsed_ms:.2f}ms 걸림 (20ms 이내여야 함)"
        assert len(formatted) <= 3900  # Slack 제한 이내

    def test_format_for_slack_split_performance(self) -> None:
        """Slack 분할 포맷팅이 30ms 이내에 완료되어야 함."""
        # 최대 길이(4000자) 텍스트로 테스트 - 분할 필요
        response = Response(
            text="답변 " * 998,  # 약 3990자 (4000자 이내)
            sources=[
                SourceReference(
                    title=f"문서 {i}",
                    url=f"https://example.com/doc-{i}",
                    source_type="notion",
                )
                for i in range(5)
            ],
        )

        start = time.perf_counter()
        parts = response.format_for_slack_split()
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 30, f"Slack 분할 포맷팅이 {elapsed_ms:.2f}ms 걸림 (30ms 이내여야 함)"
        # 모든 파트가 제한 이내인지 확인
        for part in parts:
            assert len(part) <= 3900


class TestGuardrailsPerformance:
    """가드레일 성능 테스트."""

    def test_sensitive_info_detection_performance(self) -> None:
        """민감 정보 탐지가 50ms 이내에 완료되어야 함."""
        from src.guardrails import SensitiveInfoDetector

        detector = SensitiveInfoDetector()

        # 여러 민감 정보가 포함된 큰 텍스트
        text = """
        사용자 정보:
        - 이메일: user1@example.com, user2@test.org
        - 전화번호: 010-1234-5678, 010-9876-5432
        - 주민번호: 850315-2876543
        - 카드번호: 1234-5678-9012-3456
        - API 키: sk-abcdefghijklmnopqrstuvwxyz123456
        - 서버 IP: 192.168.1.100, 10.0.0.1
        - password: secretPass123
        """ * 10  # 10배로 늘림

        start = time.perf_counter()
        result = detector.detect(text)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50, f"민감 정보 탐지가 {elapsed_ms:.2f}ms 걸림 (50ms 이내여야 함)"
        assert result.found is True
        assert len(result.patterns_matched) > 0


class TestEndToEndPerformanceBudget:
    """E2E 성능 예산 테스트.

    전체 응답 시간 10초 예산 분배:
    - 벡터 검색: ~500ms (외부 서비스)
    - 프롬프트 생성: ~100ms
    - LLM 호출: ~8000ms (외부 서비스, 대부분 시간 소요)
    - 응답 포맷팅: ~50ms
    - 오버헤드: ~350ms
    """

    def test_local_processing_under_budget(self) -> None:
        """로컬 처리 시간이 전체 예산의 10% (1초) 이내여야 함.

        외부 서비스(Qdrant, Claude API) 제외한 로컬 처리만 측정.
        """
        # 1. 프롬프트 생성
        results = [
            SearchResult(
                chunk_id=f"chunk-{i}",
                source_id=f"doc-{i}",
                source_title=f"문서 {i}",
                source_url=f"https://example.com/{i}",
                source_type="notion",
                content="테스트 내용 " * 100,
                score=0.85,
            )
            for i in range(10)
        ]

        start = time.perf_counter()

        # 컨텍스트 자르기
        truncated = truncate_context(results, max_tokens=4000)

        # 프롬프트 생성
        prompt = build_rag_prompt(
            question="테스트 질문입니다",
            search_results=truncated,
            conversation_context="이전 대화 내용",
        )

        # 응답 생성 (LLM 호출 제외, 응답 객체 생성만)
        response = Response(
            text="LLM 응답 내용입니다. " * 100,
            sources=[
                SourceReference(
                    title=r.source_title,
                    url=r.source_url,
                    source_type=r.source_type,
                )
                for r in truncated[:5]
            ],
            model="claude-sonnet-4-20250514",
            tokens_used=500,
            generation_time_ms=5000,
        )

        # Slack 포맷팅
        formatted = response.format_for_slack()

        elapsed_ms = (time.perf_counter() - start) * 1000

        # 로컬 처리는 200ms 이내여야 함 (전체 10초의 2%)
        assert elapsed_ms < 200, (
            f"로컬 처리가 {elapsed_ms:.2f}ms 걸림 (200ms 이내여야 함)"
        )

        # 결과 검증
        assert len(prompt) > 0
        assert len(formatted) > 0

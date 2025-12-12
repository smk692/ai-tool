"""모델 모듈 테스트."""


from src.models import (
    Conversation,
    ConversationMessage,
    Feedback,
    Query,
    Response,
    SearchResult,
    SourceReference,
)


class TestQuery:
    """Query 모델 테스트."""

    def test_create_query(self) -> None:
        """쿼리 생성 테스트."""
        query = Query(
            text="테스트 질문입니다",
            user_id="U1234567890",
            channel_id="C1234567890",
            thread_ts="1234567890.123456",
            message_ts="1234567890.123457",
        )
        assert query.text == "테스트 질문입니다"
        assert query.user_id == "U1234567890"
        assert query.channel_id == "C1234567890"
        assert query.is_dm is False

    def test_query_strip_mention(self) -> None:
        """멘션 태그 제거 테스트."""
        query = Query(
            text="<@U1234567890> 테스트 질문입니다",
            user_id="U1234567890",
            channel_id="C1234567890",
            thread_ts="1234567890.123456",
            message_ts="1234567890.123457",
        )
        assert query.text == "테스트 질문입니다"
        assert "<@" not in query.text

    def test_query_is_dm(self) -> None:
        """DM 여부 테스트."""
        query = Query(
            text="질문",
            user_id="U1234567890",
            channel_id="D1234567890",  # D로 시작하면 DM
            thread_ts="1234567890.123456",
            message_ts="1234567890.123457",
            is_dm=True,
        )
        assert query.is_dm is True

    def test_query_from_slack_event(self) -> None:
        """Slack 이벤트에서 Query 생성 테스트."""
        query = Query.from_slack_event(
            text="<@U1234567890> 질문입니다",
            user="U1234567890",
            channel="C1234567890",
            ts="1234567890.123456",
            thread_ts="1234567890.123455",
            channel_type="channel",
        )
        assert query.text == "질문입니다"
        assert query.is_dm is False


class TestSearchResult:
    """SearchResult 모델 테스트."""

    def test_create_search_result(self) -> None:
        """검색 결과 생성 테스트."""
        result = SearchResult(
            chunk_id="chunk-123",
            source_id="doc-123",
            source_title="테스트 문서",
            source_url="https://example.com/doc",
            source_type="notion",
            content="문서 내용입니다",
            score=0.85,
        )
        assert result.chunk_id == "chunk-123"
        assert result.source_id == "doc-123"
        assert result.score == 0.85
        assert result.is_relevant is True  # 기본 threshold 0.7 이상

    def test_search_result_relevance(self) -> None:
        """관련성 판단 테스트."""
        high_score = SearchResult(
            chunk_id="1",
            source_id="1",
            source_title="문서1",
            source_url="https://example.com",
            source_type="notion",
            content="내용",
            score=0.9,
        )
        low_score = SearchResult(
            chunk_id="2",
            source_id="2",
            source_title="문서2",
            source_url="https://example.com",
            source_type="notion",
            content="내용",
            score=0.3,
        )
        assert high_score.is_relevant is True
        assert low_score.is_relevant is False

    def test_from_qdrant_result(self) -> None:
        """Qdrant 결과에서 SearchResult 생성 테스트."""
        result = SearchResult.from_qdrant_result(
            point_id="point-123",
            score=0.88,
            payload={
                "content": "문서 내용",
                "source_type": "swagger",
                "source_id": "api-doc-1",
                "title": "API 문서",
                "url": "https://api.example.com/docs",
            },
        )
        assert result.chunk_id == "point-123"
        assert result.score == 0.88
        assert result.source_type == "swagger"
        assert result.source_title == "API 문서"


class TestSourceReference:
    """SourceReference 모델 테스트."""

    def test_create_source_reference(self) -> None:
        """소스 참조 생성 테스트."""
        source = SourceReference(
            title="API 문서",
            url="https://api.example.com/docs",
            source_type="swagger",
        )
        assert source.title == "API 문서"
        assert source.source_type == "swagger"

    def test_source_reference_without_url(self) -> None:
        """URL 없는 소스 참조 테스트."""
        source = SourceReference(
            title="내부 문서",
            source_type="notion",
        )
        assert source.url is None


class TestResponse:
    """Response 모델 테스트."""

    def test_create_response(self) -> None:
        """응답 생성 테스트."""
        response = Response(
            text="답변 내용입니다",
            sources=[
                SourceReference(
                    title="문서1",
                    url="https://example.com",
                    source_type="notion",
                )
            ],
            model="claude-sonnet-4-20250514",
            tokens_used=150,
            generation_time_ms=500,
        )
        assert response.text == "답변 내용입니다"
        assert len(response.sources) == 1
        assert response.tokens_used == 150

    def test_fallback_response(self) -> None:
        """폴백 응답 생성 테스트."""
        response = Response.fallback_response("문서를 찾을 수 없습니다.")
        assert "문서를 찾을 수 없습니다" in response.text
        assert len(response.sources) == 0
        assert response.is_fallback is True

    def test_format_for_slack(self) -> None:
        """Slack 포맷 테스트."""
        response = Response(
            text="답변입니다",
            sources=[
                SourceReference(
                    title="문서1",
                    url="https://example.com/doc1",
                    source_type="notion",
                )
            ],
        )
        formatted = response.format_for_slack()
        assert "답변입니다" in formatted
        assert "참조 문서" in formatted
        assert "<https://example.com/doc1|문서1>" in formatted


class TestConversationMessage:
    """ConversationMessage 모델 테스트."""

    def test_create_message(self) -> None:
        """메시지 생성 테스트."""
        msg = ConversationMessage(
            role="user",
            content="안녕하세요",
            ts="1234567890.123456",
        )
        assert msg.role == "user"
        assert msg.content == "안녕하세요"
        assert msg.ts == "1234567890.123456"


class TestConversation:
    """Conversation 모델 테스트."""

    def test_create_conversation(self) -> None:
        """대화 생성 테스트."""
        conv = Conversation(
            thread_ts="1234567890.123456",
            channel_id="C1234567890",
        )
        assert conv.thread_ts == "1234567890.123456"
        assert len(conv.messages) == 0

    def test_add_message(self) -> None:
        """메시지 추가 테스트."""
        conv = Conversation(
            thread_ts="1234567890.123456",
            channel_id="C1234567890",
        )
        conv.add_message(role="user", content="질문", ts="1234567890.123457")
        conv.add_message(role="assistant", content="답변", ts="1234567890.123458")
        assert len(conv.messages) == 2


class TestFeedback:
    """Feedback 모델 테스트."""

    def test_create_positive_feedback(self) -> None:
        """긍정 피드백 생성 테스트."""
        feedback = Feedback(
            message_ts="1234567890.123456",
            thread_ts="1234567890.123455",
            user_id="U1234567890",
            channel_id="C1234567890",
            question="테스트 질문",
            answer="테스트 답변",
            rating="positive",
            reaction="+1",
        )
        assert feedback.rating == "positive"
        assert feedback.reaction == "+1"

    def test_create_negative_feedback(self) -> None:
        """부정 피드백 생성 테스트."""
        feedback = Feedback(
            message_ts="1234567890.123456",
            thread_ts="1234567890.123455",
            user_id="U1234567890",
            channel_id="C1234567890",
            question="테스트 질문",
            answer="테스트 답변",
            rating="negative",
            reaction="-1",
        )
        assert feedback.rating == "negative"

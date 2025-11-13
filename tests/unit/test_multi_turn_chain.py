"""Unit tests for Multi-turn Conversation Chain."""

import pytest
from decimal import Decimal
from unittest.mock import Mock
from uuid import uuid4

from src.chains.multi_turn import MultiTurnChain
from src.db.sqlite import SQLiteConversationMemory
from src.models.query_request import QueryRequest, QueryType
from src.models.query_response import ResponseType
from src.services.llm_client import LLMClient


class TestMultiTurnChain:
    """Test suite for MultiTurnChain class."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        return Mock(spec=LLMClient)

    @pytest.fixture
    def mock_memory(self):
        """Create mock conversation memory."""
        return Mock(spec=SQLiteConversationMemory)

    @pytest.fixture
    def multi_turn_chain(self, mock_llm_client, mock_memory):
        """Create MultiTurnChain instance with mocks."""
        return MultiTurnChain(
            llm_client=mock_llm_client,
            memory=mock_memory,
        )

    @pytest.fixture
    def sample_query_request(self):
        """Create sample query request with session."""
        return QueryRequest(
            user_id="test_user",
            query_text="안녕하세요",
            query_type=QueryType.ASSISTANT,
            session_id=uuid4(),
        )

    def test_initialization(self, mock_llm_client, mock_memory):
        """Test chain initialization."""
        chain = MultiTurnChain(
            llm_client=mock_llm_client,
            memory=mock_memory,
        )

        assert chain.llm_client == mock_llm_client
        assert chain.memory == mock_memory
        assert chain.prompts is not None

    def test_chat_without_history(
        self,
        multi_turn_chain,
        mock_llm_client,
        mock_memory,
        sample_query_request,
    ):
        """Test chat without conversation history."""
        # Mock empty history
        mock_memory.get_conversation_history.return_value = []

        # Mock LLM response
        mock_llm_client.format_messages.return_value = []
        mock_llm_client.invoke.return_value = {
            "content": "안녕하세요! 무엇을 도와드릴까요?",
            "execution_time": 0.8,
            "token_usage": Mock(input_tokens=10, output_tokens=15, total_tokens=25),
        }

        result = multi_turn_chain.chat(sample_query_request)

        # Verify response
        assert result.query_id == sample_query_request.query_id
        assert result.response_type == ResponseType.ASSISTANT_MESSAGE
        assert "안녕하세요" in result.response_text
        assert result.confidence_score > Decimal("0.0")

        # Verify conversation was saved
        assert mock_memory.save_conversation.called

    def test_chat_with_history(
        self,
        multi_turn_chain,
        mock_llm_client,
        mock_memory,
        sample_query_request,
        sample_conversation_history,
    ):
        """Test chat with conversation history."""
        # Mock conversation history
        history_messages = [
            {
                "user_message": "안녕하세요",
                "assistant_message": "안녕하세요! 무엇을 도와드릴까요?",
            },
            {
                "user_message": "주문 내역을 확인하고 싶어요",
                "assistant_message": "주문 내역 확인을 도와드리겠습니다.",
            },
        ]
        mock_memory.get_conversation_history.return_value = history_messages

        # Mock LLM response
        mock_llm_client.format_messages.return_value = []
        mock_llm_client.invoke.return_value = {
            "content": "주문 번호를 알려주시겠어요?",
            "execution_time": 1.0,
            "token_usage": Mock(input_tokens=50, output_tokens=20, total_tokens=70),
        }

        # Update query text to follow conversation
        sample_query_request.query_text = "어디서 확인하나요?"

        result = multi_turn_chain.chat(sample_query_request)

        # Verify history was retrieved
        assert mock_memory.get_conversation_history.called

        # Verify result
        assert result.response_type == ResponseType.ASSISTANT_MESSAGE
        assert result.response_text is not None

    def test_chat_without_session(
        self,
        multi_turn_chain,
        mock_llm_client,
        mock_memory,
    ):
        """Test chat without session ID (no history)."""
        query_request = QueryRequest(
            user_id="test_user",
            query_text="안녕하세요",
            session_id=None,  # No session
        )

        # Mock LLM response
        mock_llm_client.format_messages.return_value = []
        mock_llm_client.invoke.return_value = {
            "content": "안녕하세요!",
            "execution_time": 0.5,
            "token_usage": Mock(input_tokens=5, output_tokens=5, total_tokens=10),
        }

        result = multi_turn_chain.chat(query_request)

        # History should not be retrieved
        assert not mock_memory.get_conversation_history.called

        # Conversation should not be saved
        assert not mock_memory.save_conversation.called

    def test_chat_history_retrieval_error(
        self,
        multi_turn_chain,
        mock_llm_client,
        mock_memory,
        sample_query_request,
    ):
        """Test handling of history retrieval errors."""
        # Mock history error
        mock_memory.get_conversation_history.side_effect = Exception("DB error")

        # Mock LLM response
        mock_llm_client.format_messages.return_value = []
        mock_llm_client.invoke.return_value = {
            "content": "안녕하세요!",
            "execution_time": 0.5,
            "token_usage": Mock(input_tokens=5, output_tokens=5, total_tokens=10),
        }

        # Should not raise error, continue without history
        result = multi_turn_chain.chat(sample_query_request)

        assert result.response_type == ResponseType.ASSISTANT_MESSAGE

    def test_chat_save_error(
        self,
        multi_turn_chain,
        mock_llm_client,
        mock_memory,
        sample_query_request,
    ):
        """Test handling of conversation save errors."""
        # Mock empty history
        mock_memory.get_conversation_history.return_value = []

        # Mock save error
        mock_memory.save_conversation.side_effect = Exception("Save failed")

        # Mock LLM response
        mock_llm_client.format_messages.return_value = []
        mock_llm_client.invoke.return_value = {
            "content": "안녕하세요!",
            "execution_time": 0.5,
            "token_usage": Mock(input_tokens=5, output_tokens=5, total_tokens=10),
        }

        # Should not raise error, continue anyway
        result = multi_turn_chain.chat(sample_query_request)

        assert result.response_type == ResponseType.ASSISTANT_MESSAGE

    def test_chat_llm_error(
        self,
        multi_turn_chain,
        mock_llm_client,
        mock_memory,
        sample_query_request,
    ):
        """Test handling of LLM invocation errors."""
        # Mock empty history
        mock_memory.get_conversation_history.return_value = []

        # Mock LLM error
        mock_llm_client.format_messages.return_value = []
        mock_llm_client.invoke.side_effect = Exception("LLM failed")

        result = multi_turn_chain.chat(sample_query_request)

        # Should return error response
        assert result.response_type == ResponseType.ERROR
        assert "대화 생성" in result.error_message

    def test_format_history(self, multi_turn_chain):
        """Test conversation history formatting."""
        history_messages = [
            {
                "user_message": "첫 번째 질문",
                "assistant_message": "첫 번째 답변",
            },
            {
                "user_message": "두 번째 질문",
                "assistant_message": "두 번째 답변",
            },
        ]

        formatted = multi_turn_chain._format_history(history_messages)

        # Should have 4 messages (2 user + 2 assistant)
        assert len(formatted) == 4

        # Check order and roles
        assert formatted[0]["role"] == "user"
        assert formatted[0]["content"] == "첫 번째 질문"
        assert formatted[1]["role"] == "assistant"
        assert formatted[1]["content"] == "첫 번째 답변"
        assert formatted[2]["role"] == "user"
        assert formatted[3]["role"] == "assistant"

    def test_format_history_missing_messages(self, multi_turn_chain):
        """Test history formatting with missing messages."""
        history_messages = [
            {"user_message": "질문만 있음"},  # No assistant message
            {"assistant_message": "답변만 있음"},  # No user message
        ]

        formatted = multi_turn_chain._format_history(history_messages)

        # Should handle missing messages gracefully
        assert len(formatted) == 2

    def test_calculate_confidence_normal_response(self, multi_turn_chain):
        """Test confidence calculation for normal response."""
        response = "안녕하세요! 무엇을 도와드릴까요?"

        confidence = multi_turn_chain._calculate_confidence(response)

        assert confidence == Decimal("0.8")  # Base confidence

    def test_calculate_confidence_short_response(self, multi_turn_chain):
        """Test confidence calculation for short response."""
        response = "네"  # Very short

        confidence = multi_turn_chain._calculate_confidence(response)

        # Should be reduced
        assert confidence < Decimal("0.8")

    def test_calculate_confidence_uncertain_response(self, multi_turn_chain):
        """Test confidence calculation with uncertainty words."""
        response = "잘 모르겠습니다만, 아마도 이렇게 하시면 될 것 같습니다."

        confidence = multi_turn_chain._calculate_confidence(response)

        # Should be significantly reduced
        assert confidence < Decimal("0.7")

    def test_calculate_confidence_bounds(self, multi_turn_chain):
        """Test confidence is always within valid range."""
        test_cases = [
            "네",  # Short
            "잘 모르겠습니다",  # Uncertain
            "안녕하세요! 무엇을 도와드릴까요?",  # Normal
            "A" * 1000,  # Very long
        ]

        for response in test_cases:
            confidence = multi_turn_chain._calculate_confidence(response)
            assert Decimal("0.3") <= confidence <= Decimal("1.0")

    def test_clear_session_success(
        self,
        multi_turn_chain,
        mock_memory,
    ):
        """Test successful session clearing."""
        session_id = str(uuid4())

        result = multi_turn_chain.clear_session(session_id)

        assert result is True
        assert mock_memory.clear_session.called_with(session_id)

    def test_clear_session_error(
        self,
        multi_turn_chain,
        mock_memory,
    ):
        """Test session clearing error handling."""
        session_id = str(uuid4())
        mock_memory.clear_session.side_effect = Exception("Clear failed")

        result = multi_turn_chain.clear_session(session_id)

        assert result is False

    def test_error_response_creation(
        self,
        multi_turn_chain,
        sample_query_request,
    ):
        """Test error response creation."""
        error_msg = "Test error"

        response = multi_turn_chain._error_response(
            sample_query_request,
            error_msg,
        )

        assert response.query_id == sample_query_request.query_id
        assert response.response_type == ResponseType.ERROR
        assert response.error_message == error_msg
        assert response.confidence_score == Decimal("0.0")

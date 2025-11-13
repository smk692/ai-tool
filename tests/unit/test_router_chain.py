"""Unit tests for Router Chain."""

import pytest
from unittest.mock import Mock, patch
from uuid import uuid4

from src.chains.router import RouterChain
from src.models.query_request import QueryRequest, QueryType
from src.services.llm_client import LLMClient


class TestRouterChain:
    """Test suite for RouterChain class."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        return Mock(spec=LLMClient)

    @pytest.fixture
    def router_chain(self, mock_llm_client):
        """Create RouterChain instance with mock LLM."""
        return RouterChain(llm_client=mock_llm_client)

    def test_initialization(self, mock_llm_client):
        """Test router chain initialization."""
        chain = RouterChain(llm_client=mock_llm_client)

        assert chain.llm_client == mock_llm_client
        assert chain.prompts is not None

    def test_classify_text_to_sql(self, router_chain, mock_llm_client, korean_text_to_sql_queries):
        """Test classification of text-to-sql queries."""
        # Mock LLM response
        mock_llm_client.format_messages.return_value = []
        mock_llm_client.invoke.return_value = {
            "content": "text_to_sql",
            "token_usage": Mock(total_tokens=20),
        }

        query_request = QueryRequest(
            user_id="test_user",
            query_text=korean_text_to_sql_queries[0]["query"],
        )

        result = router_chain.classify(query_request)

        assert result == QueryType.TEXT_TO_SQL
        assert query_request.query_type == QueryType.TEXT_TO_SQL
        assert mock_llm_client.invoke.called

    def test_classify_knowledge(self, router_chain, mock_llm_client, korean_knowledge_queries):
        """Test classification of knowledge queries."""
        mock_llm_client.format_messages.return_value = []
        mock_llm_client.invoke.return_value = {
            "content": "knowledge discovery",
            "token_usage": Mock(total_tokens=20),
        }

        query_request = QueryRequest(
            user_id="test_user",
            query_text=korean_knowledge_queries[0]["query"],
        )

        result = router_chain.classify(query_request)

        assert result == QueryType.KNOWLEDGE
        assert query_request.query_type == QueryType.KNOWLEDGE

    def test_classify_assistant(self, router_chain, mock_llm_client, korean_assistant_queries):
        """Test classification of general assistant queries."""
        mock_llm_client.format_messages.return_value = []
        mock_llm_client.invoke.return_value = {
            "content": "general conversation",
            "token_usage": Mock(total_tokens=20),
        }

        query_request = QueryRequest(
            user_id="test_user",
            query_text=korean_assistant_queries[0]["query"],
        )

        result = router_chain.classify(query_request)

        assert result == QueryType.ASSISTANT
        assert query_request.query_type == QueryType.ASSISTANT

    def test_classify_with_sql_keyword(self, router_chain, mock_llm_client):
        """Test classification with 'sql' keyword in response."""
        mock_llm_client.format_messages.return_value = []
        mock_llm_client.invoke.return_value = {
            "content": "This is a SQL query",
            "token_usage": Mock(total_tokens=20),
        }

        query_request = QueryRequest(
            user_id="test_user",
            query_text="test query",
        )

        result = router_chain.classify(query_request)

        assert result == QueryType.TEXT_TO_SQL

    def test_classify_with_document_keyword(self, router_chain, mock_llm_client):
        """Test classification with 'document' keyword in response."""
        mock_llm_client.format_messages.return_value = []
        mock_llm_client.invoke.return_value = {
            "content": "This is about document retrieval",
            "token_usage": Mock(total_tokens=20),
        }

        query_request = QueryRequest(
            user_id="test_user",
            query_text="test query",
        )

        result = router_chain.classify(query_request)

        assert result == QueryType.KNOWLEDGE

    def test_route_to_text_to_sql_chain(self, router_chain, mock_llm_client):
        """Test routing to text-to-sql chain."""
        mock_llm_client.format_messages.return_value = []
        mock_llm_client.invoke.return_value = {
            "content": "text_to_sql",
            "token_usage": Mock(total_tokens=20),
        }

        query_request = QueryRequest(
            user_id="test_user",
            query_text="지난달 매출은?",
        )

        result = router_chain.route(query_request)

        assert result["query_type"] == QueryType.TEXT_TO_SQL.value
        assert result["next_chain"] == "text_to_sql_chain"

    def test_route_to_knowledge_chain(self, router_chain, mock_llm_client):
        """Test routing to knowledge chain."""
        mock_llm_client.format_messages.return_value = []
        mock_llm_client.invoke.return_value = {
            "content": "knowledge",
            "token_usage": Mock(total_tokens=20),
        }

        query_request = QueryRequest(
            user_id="test_user",
            query_text="환불 정책은?",
        )

        result = router_chain.route(query_request)

        assert result["query_type"] == QueryType.KNOWLEDGE.value
        assert result["next_chain"] == "knowledge_chain"

    def test_route_to_multi_turn_chain(self, router_chain, mock_llm_client):
        """Test routing to multi-turn chain."""
        mock_llm_client.format_messages.return_value = []
        mock_llm_client.invoke.return_value = {
            "content": "assistant",
            "token_usage": Mock(total_tokens=20),
        }

        query_request = QueryRequest(
            user_id="test_user",
            query_text="안녕하세요",
        )

        result = router_chain.route(query_request)

        assert result["query_type"] == QueryType.ASSISTANT.value
        assert result["next_chain"] == "multi_turn_chain"

    def test_classify_updates_query_request(self, router_chain, mock_llm_client):
        """Test that classify updates the query_request object."""
        mock_llm_client.format_messages.return_value = []
        mock_llm_client.invoke.return_value = {
            "content": "text_to_sql",
            "token_usage": Mock(total_tokens=20),
        }

        query_request = QueryRequest(
            user_id="test_user",
            query_text="test",
        )

        # Initially query_type is None
        assert query_request.query_type is None

        router_chain.classify(query_request)

        # After classification, query_type should be set
        assert query_request.query_type == QueryType.TEXT_TO_SQL

    def test_classify_case_insensitive(self, router_chain, mock_llm_client):
        """Test that classification is case-insensitive."""
        mock_llm_client.format_messages.return_value = []
        mock_llm_client.invoke.return_value = {
            "content": "TEXT_TO_SQL",  # Uppercase
            "token_usage": Mock(total_tokens=20),
        }

        query_request = QueryRequest(
            user_id="test_user",
            query_text="test",
        )

        result = router_chain.classify(query_request)

        assert result == QueryType.TEXT_TO_SQL

    def test_classify_with_extra_whitespace(self, router_chain, mock_llm_client):
        """Test classification with extra whitespace in response."""
        mock_llm_client.format_messages.return_value = []
        mock_llm_client.invoke.return_value = {
            "content": "  text_to_sql  ",  # Extra whitespace
            "token_usage": Mock(total_tokens=20),
        }

        query_request = QueryRequest(
            user_id="test_user",
            query_text="test",
        )

        result = router_chain.classify(query_request)

        assert result == QueryType.TEXT_TO_SQL

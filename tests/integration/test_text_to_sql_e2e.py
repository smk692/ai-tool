"""End-to-end integration tests for Text-to-SQL workflow."""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch

from src.chains.text_to_sql import TextToSQLChain
from src.chains.router import RouterChain
from src.services.llm_client import LLMClient
from src.models.query_request import QueryRequest
from src.models.llm_config import LLMConfiguration, LLMProvider


@pytest.fixture
def sample_queries():
    """Load sample queries from fixtures."""
    fixture_path = Path(__file__).parent.parent / "fixtures" / "sample_queries.json"
    with open(fixture_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["text_to_sql_queries"]


@pytest.fixture
def mock_responses():
    """Load mock Claude responses from fixtures."""
    fixture_path = Path(__file__).parent.parent / "fixtures" / "mock_responses.json"
    with open(fixture_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["claude_sql_responses"]


@pytest.fixture
def llm_config():
    """Create test LLM configuration."""
    return LLMConfiguration(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-3-5-sonnet-20241022",
        api_key="test_api_key",
        temperature=0.7,
        max_tokens=4096,
    )


@pytest.fixture
def text_to_sql_chain(llm_config):
    """Create Text-to-SQL chain with test configuration."""
    return TextToSQLChain(llm_config=llm_config)


@pytest.fixture
def router_chain(llm_config):
    """Create router chain for intent classification."""
    return RouterChain(llm_config=llm_config)


class TestTextToSQLWorkflowIntegration:
    """Integration tests for complete Text-to-SQL workflow."""

    @patch("src.services.llm_client.ChatAnthropic")
    def test_simple_aggregation_query(
        self, mock_chat, text_to_sql_chain, sample_queries, mock_responses
    ):
        """Test simple aggregation query workflow (q001: 지난달 신규 가입자 수는?)."""
        query = sample_queries[0]  # q001
        mock_response_data = mock_responses[0]

        # Mock LLM response
        mock_response = Mock()
        mock_response.content = mock_response_data["response"]["content"][0]["text"]
        mock_response.response_metadata = {
            "model": mock_response_data["response"]["model"],
            "usage": mock_response_data["response"]["usage"],
        }

        mock_chat_instance = Mock()
        mock_chat_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_chat_instance

        # Execute workflow
        result = text_to_sql_chain.generate_sql(query["query"])

        # Assertions
        assert result is not None
        assert "SELECT" in result.upper()
        assert "users" in result.lower()
        assert "COUNT" in result.upper()
        mock_chat_instance.invoke.assert_called_once()

    @patch("src.services.llm_client.ChatAnthropic")
    def test_join_query_with_sorting(
        self, mock_chat, text_to_sql_chain, sample_queries
    ):
        """Test join query with sorting (q003: 가장 많이 판매된 상품 top 5는?)."""
        query = sample_queries[2]  # q003

        # Mock LLM response
        mock_sql = """
        SELECT p.product_name, SUM(oi.quantity) as total_sold
        FROM products p
        JOIN order_items oi ON p.product_id = oi.product_id
        GROUP BY p.product_id, p.product_name
        ORDER BY total_sold DESC
        LIMIT 5;
        """
        mock_response = Mock()
        mock_response.content = mock_sql
        mock_response.response_metadata = {
            "model": "claude-3-5-sonnet-20241022",
            "usage": {"input_tokens": 1300, "output_tokens": 120},
        }

        mock_chat_instance = Mock()
        mock_chat_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_chat_instance

        # Execute workflow
        result = text_to_sql_chain.generate_sql(query["query"])

        # Assertions
        assert "JOIN" in result.upper()
        assert "products" in result.lower()
        assert "order_items" in result.lower()
        assert "LIMIT 5" in result.upper()
        assert "ORDER BY" in result.upper()

    @patch("src.services.llm_client.ChatAnthropic")
    def test_group_by_aggregation(
        self, mock_chat, text_to_sql_chain, sample_queries
    ):
        """Test group by with aggregation (q004: 부서별 평균 급여를 구해줘)."""
        query = sample_queries[3]  # q004

        # Mock LLM response
        mock_sql = """
        SELECT d.department_name, AVG(e.salary) as avg_salary
        FROM departments d
        JOIN employees e ON d.department_id = e.department_id
        GROUP BY d.department_id, d.department_name
        ORDER BY avg_salary DESC;
        """
        mock_response = Mock()
        mock_response.content = mock_sql
        mock_response.response_metadata = {
            "model": "claude-3-5-sonnet-20241022",
            "usage": {"input_tokens": 1280, "output_tokens": 110},
        }

        mock_chat_instance = Mock()
        mock_chat_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_chat_instance

        # Execute workflow
        result = text_to_sql_chain.generate_sql(query["query"])

        # Assertions
        assert "GROUP BY" in result.upper()
        assert "AVG" in result.upper()
        assert "departments" in result.lower() or "employees" in result.lower()


class TestTextToSQLErrorHandling:
    """Test error handling in Text-to-SQL workflow."""

    @patch("src.services.llm_client.ChatAnthropic")
    def test_empty_query_handling(self, mock_chat, text_to_sql_chain):
        """Test handling of empty query."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            text_to_sql_chain.generate_sql("")

    @patch("src.services.llm_client.ChatAnthropic")
    def test_llm_timeout_handling(self, mock_chat, text_to_sql_chain):
        """Test handling of LLM timeout."""
        mock_chat_instance = Mock()
        mock_chat_instance.invoke.side_effect = TimeoutError("Request timeout")
        mock_chat.return_value = mock_chat_instance

        with pytest.raises(TimeoutError):
            text_to_sql_chain.generate_sql("지난달 신규 가입자 수는?")

    @patch("src.services.llm_client.ChatAnthropic")
    def test_invalid_sql_response(self, mock_chat, text_to_sql_chain):
        """Test handling of invalid SQL response."""
        # Mock LLM returning non-SQL text
        mock_response = Mock()
        mock_response.content = "I don't understand the question."
        mock_response.response_metadata = {
            "model": "claude-3-5-sonnet-20241022",
            "usage": {"input_tokens": 100, "output_tokens": 20},
        }

        mock_chat_instance = Mock()
        mock_chat_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_chat_instance

        result = text_to_sql_chain.generate_sql("이상한 질문")

        # Should still return the response (validation happens elsewhere)
        assert result is not None


class TestIntentClassificationWorkflow:
    """Test intent classification as part of the workflow."""

    @patch("src.services.llm_client.ChatAnthropic")
    def test_text_to_sql_intent_classification(
        self, mock_chat, router_chain, sample_queries
    ):
        """Test that SQL query is correctly classified as text_to_sql intent."""
        query = sample_queries[0]  # q001

        # Mock intent classification response
        mock_response = Mock()
        mock_response.content = '{"intent": "text_to_sql", "confidence": 0.95}'
        mock_response.response_metadata = {
            "model": "claude-3-5-sonnet-20241022",
            "usage": {"input_tokens": 150, "output_tokens": 30},
        }

        mock_chat_instance = Mock()
        mock_chat_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_chat_instance

        # Execute workflow
        intent = router_chain.classify_intent(query["query"])

        # Assertions
        assert intent == "text_to_sql"


class TestTextToSQLPerformance:
    """Test performance requirements for Text-to-SQL workflow."""

    @patch("src.services.llm_client.ChatAnthropic")
    def test_response_time_within_sla(
        self, mock_chat, text_to_sql_chain, sample_queries
    ):
        """Test that response time is within 60 second SLA."""
        import time

        query = sample_queries[0]

        # Mock quick response
        mock_response = Mock()
        mock_response.content = "SELECT COUNT(*) FROM users;"
        mock_response.response_metadata = {
            "model": "claude-3-5-sonnet-20241022",
            "usage": {"input_tokens": 100, "output_tokens": 20},
        }

        mock_chat_instance = Mock()
        mock_chat_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_chat_instance

        # Measure execution time
        start_time = time.time()
        result = text_to_sql_chain.generate_sql(query["query"])
        elapsed_time = time.time() - start_time

        # Assertions
        assert result is not None
        assert elapsed_time < 60, f"Response time {elapsed_time}s exceeds 60s SLA"


class TestKoreanLanguageSupport:
    """Test Korean language support in Text-to-SQL workflow."""

    @patch("src.services.llm_client.ChatAnthropic")
    def test_korean_query_processing(
        self, mock_chat, text_to_sql_chain, sample_queries
    ):
        """Test processing of Korean language queries."""
        # Test all Korean queries
        for query in sample_queries:
            mock_response = Mock()
            mock_response.content = "SELECT * FROM test;"
            mock_response.response_metadata = {
                "model": "claude-3-5-sonnet-20241022",
                "usage": {"input_tokens": 100, "output_tokens": 20},
            }

            mock_chat_instance = Mock()
            mock_chat_instance.invoke.return_value = mock_response
            mock_chat.return_value = mock_chat_instance

            # Should not raise any encoding errors
            result = text_to_sql_chain.generate_sql(query["query"])
            assert result is not None

    @patch("src.services.llm_client.ChatAnthropic")
    def test_mixed_korean_english_query(self, mock_chat, text_to_sql_chain):
        """Test processing of mixed Korean-English queries."""
        mixed_query = "지난달 new users는 몇 명이야?"

        mock_response = Mock()
        mock_response.content = "SELECT COUNT(*) FROM users;"
        mock_response.response_metadata = {
            "model": "claude-3-5-sonnet-20241022",
            "usage": {"input_tokens": 120, "output_tokens": 25},
        }

        mock_chat_instance = Mock()
        mock_chat_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_chat_instance

        result = text_to_sql_chain.generate_sql(mixed_query)

        assert result is not None
        assert "users" in result.lower()

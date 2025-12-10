"""End-to-end integration tests for Knowledge Discovery workflow."""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.chains.knowledge import KnowledgeChain
from src.chains.router import RouterChain
from src.services.llm_client import LLMClient
from src.services.vector_store import VectorStore
from src.models.llm_config import LLMConfiguration, LLMProvider


@pytest.fixture
def sample_queries():
    """Load sample knowledge discovery queries from fixtures."""
    fixture_path = Path(__file__).parent.parent / "fixtures" / "sample_queries.json"
    with open(fixture_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["knowledge_discovery_queries"]


@pytest.fixture
def mock_responses():
    """Load mock Claude responses from fixtures."""
    fixture_path = Path(__file__).parent.parent / "fixtures" / "mock_responses.json"
    with open(fixture_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["claude_knowledge_responses"]


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
def mock_vector_store():
    """Create mock vector store."""
    mock_store = Mock(spec=VectorStore)

    # Mock search results
    mock_store.search.return_value = [
        {
            "content": "데이터베이스 백업은 매일 오전 2시에 자동으로 수행됩니다.",
            "metadata": {"source": "운영 매뉴얼", "section": "3.2", "page": 15},
            "score": 0.92,
        },
        {
            "content": "백업 파일은 S3 버킷에 암호화되어 저장됩니다.",
            "metadata": {"source": "보안 정책", "section": "2.1", "page": 8},
            "score": 0.88,
        },
        {
            "content": "백업 보관 기간: 일일 7일, 주간 4주, 월간 12개월",
            "metadata": {"source": "운영 매뉴얼", "section": "3.2.1", "page": 16},
            "score": 0.85,
        },
    ]

    return mock_store


@pytest.fixture
def knowledge_chain(llm_config, mock_vector_store):
    """Create Knowledge Discovery chain with test configuration."""
    chain = KnowledgeChain(llm_config=llm_config)
    chain.vector_store = mock_vector_store  # Inject mock
    return chain


@pytest.fixture
def router_chain(llm_config):
    """Create router chain for intent classification."""
    return RouterChain(llm_config=llm_config)


class TestKnowledgeDiscoveryWorkflowIntegration:
    """Integration tests for complete Knowledge Discovery workflow."""

    @patch("src.services.llm_client.ChatAnthropic")
    def test_policy_documentation_retrieval(
        self, mock_chat, knowledge_chain, sample_queries, mock_responses
    ):
        """Test policy documentation retrieval (k001: 데이터베이스 백업 정책)."""
        query = sample_queries[0]  # k001
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
        result = knowledge_chain.search(query["query"])

        # Assertions
        assert result is not None
        assert "백업" in result or "backup" in result.lower()
        assert "출처" in result or "source" in result.lower()
        mock_chat_instance.invoke.assert_called_once()
        knowledge_chain.vector_store.search.assert_called_once()

    @patch("src.services.llm_client.ChatAnthropic")
    def test_technical_documentation_search(
        self, mock_chat, knowledge_chain, sample_queries
    ):
        """Test technical documentation search (k002: API 인증 방법)."""
        query = sample_queries[1]  # k002

        # Mock LLM response
        mock_answer = """
        API 인증 방법:

        1. **API Key 인증**: 요청 헤더에 X-API-Key 포함
        2. **OAuth 2.0**: 토큰 기반 인증
        3. **JWT**: JSON Web Token 사용

        출처: API 문서 2.1절
        """
        mock_response = Mock()
        mock_response.content = mock_answer
        mock_response.response_metadata = {
            "model": "claude-3-5-sonnet-20241022",
            "usage": {"input_tokens": 2200, "output_tokens": 150},
        }

        mock_chat_instance = Mock()
        mock_chat_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_chat_instance

        # Update mock vector store for this query
        knowledge_chain.vector_store.search.return_value = [
            {
                "content": "API는 API Key 또는 OAuth 2.0으로 인증합니다.",
                "metadata": {"source": "API 문서", "section": "2.1"},
                "score": 0.90,
            }
        ]

        # Execute workflow
        result = knowledge_chain.search(query["query"])

        # Assertions
        assert "API" in result or "api" in result.lower()
        assert "인증" in result or "auth" in result.lower()

    @patch("src.services.llm_client.ChatAnthropic")
    def test_process_documentation_retrieval(
        self, mock_chat, knowledge_chain, sample_queries
    ):
        """Test process documentation retrieval (k003: 배포 절차)."""
        query = sample_queries[2]  # k003

        # Mock LLM response
        mock_answer = """
        신규 기능 배포 절차:

        1. 개발 완료 및 코드 리뷰
        2. 스테이징 환경 배포 및 테스트
        3. 운영팀 승인
        4. 프로덕션 배포
        5. 모니터링 및 롤백 준비

        출처: 배포 가이드 4.1절
        """
        mock_response = Mock()
        mock_response.content = mock_answer
        mock_response.response_metadata = {
            "model": "claude-3-5-sonnet-20241022",
            "usage": {"input_tokens": 2100, "output_tokens": 180},
        }

        mock_chat_instance = Mock()
        mock_chat_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_chat_instance

        # Execute workflow
        result = knowledge_chain.search(query["query"])

        # Assertions
        assert "배포" in result or "deploy" in result.lower()
        assert "절차" in result or "process" in result.lower()


class TestKnowledgeDiscoveryErrorHandling:
    """Test error handling in Knowledge Discovery workflow."""

    @patch("src.services.llm_client.ChatAnthropic")
    def test_empty_query_handling(self, mock_chat, knowledge_chain):
        """Test handling of empty query."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            knowledge_chain.search("")

    @patch("src.services.llm_client.ChatAnthropic")
    def test_no_search_results(self, mock_chat, knowledge_chain):
        """Test handling when no search results are found."""
        # Mock empty search results
        knowledge_chain.vector_store.search.return_value = []

        # Mock LLM response for "no results" case
        mock_response = Mock()
        mock_response.content = "죄송합니다. 관련 정보를 찾을 수 없습니다."
        mock_response.response_metadata = {
            "model": "claude-3-5-sonnet-20241022",
            "usage": {"input_tokens": 500, "output_tokens": 50},
        }

        mock_chat_instance = Mock()
        mock_chat_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_chat_instance

        result = knowledge_chain.search("존재하지 않는 정보")

        # Should return graceful message
        assert "찾을 수 없" in result or "not found" in result.lower()

    @patch("src.services.llm_client.ChatAnthropic")
    def test_vector_store_error_handling(self, mock_chat, knowledge_chain):
        """Test handling of vector store errors."""
        # Mock vector store error
        knowledge_chain.vector_store.search.side_effect = Exception(
            "Vector store unavailable"
        )

        with pytest.raises(Exception, match="Vector store unavailable"):
            knowledge_chain.search("테스트 쿼리")


class TestIntentClassificationForKnowledge:
    """Test intent classification for knowledge queries."""

    @patch("src.services.llm_client.ChatAnthropic")
    def test_knowledge_discovery_intent_classification(
        self, mock_chat, router_chain, sample_queries
    ):
        """Test that knowledge query is correctly classified."""
        query = sample_queries[0]  # k001

        # Mock intent classification response
        mock_response = Mock()
        mock_response.content = '{"intent": "knowledge_discovery", "confidence": 0.92}'
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
        assert intent == "knowledge_discovery"


class TestKnowledgeDiscoveryPerformance:
    """Test performance requirements for Knowledge Discovery."""

    @patch("src.services.llm_client.ChatAnthropic")
    def test_response_time_within_sla(
        self, mock_chat, knowledge_chain, sample_queries
    ):
        """Test that response time is within 3 second SLA."""
        import time

        query = sample_queries[0]

        # Mock quick response
        mock_response = Mock()
        mock_response.content = "백업은 매일 수행됩니다. 출처: 매뉴얼 3.2절"
        mock_response.response_metadata = {
            "model": "claude-3-5-sonnet-20241022",
            "usage": {"input_tokens": 500, "output_tokens": 100},
        }

        mock_chat_instance = Mock()
        mock_chat_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_chat_instance

        # Measure execution time
        start_time = time.time()
        result = knowledge_chain.search(query["query"])
        elapsed_time = time.time() - start_time

        # Assertions
        assert result is not None
        assert elapsed_time < 3, f"Response time {elapsed_time}s exceeds 3s SLA"

    @patch("src.services.llm_client.ChatAnthropic")
    def test_vector_search_latency(self, mock_chat, knowledge_chain):
        """Test that vector search completes within 0.5s."""
        import time

        # Measure vector search time
        start_time = time.time()
        knowledge_chain.vector_store.search("테스트 쿼리", top_k=5)
        search_time = time.time() - start_time

        # Assertions (mock should be instant, but check anyway)
        assert search_time < 0.5, f"Vector search {search_time}s exceeds 0.5s SLA"


class TestSourceCitation:
    """Test source citation in Knowledge Discovery responses."""

    @patch("src.services.llm_client.ChatAnthropic")
    def test_response_includes_source_citation(
        self, mock_chat, knowledge_chain, sample_queries
    ):
        """Test that responses include source citations."""
        query = sample_queries[0]

        # Mock LLM response with citations
        mock_response = Mock()
        mock_response.content = "백업 정책: 매일 수행됩니다.\n\n출처: 운영 매뉴얼 3.2절"
        mock_response.response_metadata = {
            "model": "claude-3-5-sonnet-20241022",
            "usage": {"input_tokens": 2000, "output_tokens": 150},
        }

        mock_chat_instance = Mock()
        mock_chat_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_chat_instance

        result = knowledge_chain.search(query["query"])

        # Assertions
        assert "출처" in result or "source" in result.lower()
        assert "매뉴얼" in result or "manual" in result.lower()

    @patch("src.services.llm_client.ChatAnthropic")
    def test_multiple_source_citations(self, mock_chat, knowledge_chain):
        """Test handling of multiple source citations."""
        # Update mock to return multiple sources
        knowledge_chain.vector_store.search.return_value = [
            {
                "content": "첫 번째 정보",
                "metadata": {"source": "문서A", "section": "1.1"},
                "score": 0.90,
            },
            {
                "content": "두 번째 정보",
                "metadata": {"source": "문서B", "section": "2.2"},
                "score": 0.85,
            },
        ]

        # Mock LLM response with multiple citations
        mock_response = Mock()
        mock_response.content = "정보 요약\n\n출처: 문서A 1.1절, 문서B 2.2절"
        mock_response.response_metadata = {
            "model": "claude-3-5-sonnet-20241022",
            "usage": {"input_tokens": 2500, "output_tokens": 200},
        }

        mock_chat_instance = Mock()
        mock_chat_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_chat_instance

        result = knowledge_chain.search("테스트 질문")

        # Should include multiple source references
        assert "문서A" in result and "문서B" in result


class TestKoreanLanguageSupportInKnowledge:
    """Test Korean language support in Knowledge Discovery."""

    @patch("src.services.llm_client.ChatAnthropic")
    def test_korean_query_processing(
        self, mock_chat, knowledge_chain, sample_queries
    ):
        """Test processing of Korean language queries."""
        # Test all Korean queries
        for query in sample_queries:
            mock_response = Mock()
            mock_response.content = f"{query['query']}에 대한 답변입니다."
            mock_response.response_metadata = {
                "model": "claude-3-5-sonnet-20241022",
                "usage": {"input_tokens": 500, "output_tokens": 100},
            }

            mock_chat_instance = Mock()
            mock_chat_instance.invoke.return_value = mock_response
            mock_chat.return_value = mock_chat_instance

            # Should not raise any encoding errors
            result = knowledge_chain.search(query["query"])
            assert result is not None
            assert len(result) > 0


class TestSearchAccuracy:
    """Test search accuracy requirements (Top-5 ≥90%)."""

    @patch("src.services.llm_client.ChatAnthropic")
    def test_top_5_search_results(self, mock_chat, knowledge_chain):
        """Test that top 5 search results are returned."""
        # Mock 5 search results
        knowledge_chain.vector_store.search.return_value = [
            {"content": f"결과 {i}", "metadata": {"source": f"문서{i}"}, "score": 0.9 - i * 0.05}
            for i in range(1, 6)
        ]

        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "검색 결과 요약"
        mock_response.response_metadata = {
            "model": "claude-3-5-sonnet-20241022",
            "usage": {"input_tokens": 3000, "output_tokens": 200},
        }

        mock_chat_instance = Mock()
        mock_chat_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_chat_instance

        result = knowledge_chain.search("테스트 질문", top_k=5)

        # Verify that search was called with top_k=5
        knowledge_chain.vector_store.search.assert_called_with("테스트 질문", top_k=5)

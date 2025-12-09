"""Unit tests for Knowledge Discovery Chain."""

import pytest
from decimal import Decimal
from unittest.mock import Mock, patch
from uuid import uuid4

from src.chains.knowledge import KnowledgeChain
from src.models.query_request import QueryRequest, QueryType
from src.models.query_response import ResponseType, SourceDocument
from src.services.llm_client import LLMClient
from src.services.vector_store import VectorStore


class TestKnowledgeChain:
    """Test suite for KnowledgeChain class."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        return Mock(spec=LLMClient)

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        return Mock(spec=VectorStore)

    @pytest.fixture
    def knowledge_chain(self, mock_llm_client, mock_vector_store):
        """Create KnowledgeChain instance with mocks."""
        return KnowledgeChain(
            llm_client=mock_llm_client,
            vector_store=mock_vector_store,
        )

    @pytest.fixture
    def sample_query_request(self):
        """Create sample query request."""
        return QueryRequest(
            user_id="test_user",
            query_text="회원가입 절차가 어떻게 되나요?",
            query_type=QueryType.KNOWLEDGE,
        )

    @pytest.fixture
    def sample_search_results(self):
        """Create sample vector search results."""
        return {
            "ids": [["doc1", "doc2", "doc3"]],
            "documents": [[
                "회원가입 절차는 다음과 같습니다...",
                "이메일 인증이 필요합니다...",
                "프로필 정보를 입력하세요...",
            ]],
            "metadatas": [[
                {"title": "회원가입 가이드"},
                {"title": "이메일 인증"},
                {"title": "프로필 설정"},
            ]],
            "distances": [[0.1, 0.2, 0.3]],
        }

    def test_initialization(self, mock_llm_client, mock_vector_store):
        """Test chain initialization."""
        chain = KnowledgeChain(
            llm_client=mock_llm_client,
            vector_store=mock_vector_store,
        )

        assert chain.llm_client == mock_llm_client
        assert chain.vector_store == mock_vector_store
        assert chain.prompts is not None

    def test_answer_question_success(
        self,
        knowledge_chain,
        mock_llm_client,
        mock_vector_store,
        sample_query_request,
        sample_search_results,
    ):
        """Test successful question answering."""
        # Mock vector search
        mock_vector_store.query.return_value = sample_search_results

        # Mock LLM response
        mock_llm_client.format_messages.return_value = []
        mock_llm_client.invoke.return_value = {
            "content": "회원가입은 이메일 주소 입력으로 시작합니다.",
            "execution_time": 1.2,
            "token_usage": {"input_tokens": 200, "output_tokens": 80, "total_tokens": 280},
        }

        result = knowledge_chain.answer_question(sample_query_request)

        # Verify response
        assert result.query_id == sample_query_request.query_id
        assert result.response_type == ResponseType.DOCUMENT_ANSWER
        assert len(result.source_documents) == 3
        assert result.confidence_score > Decimal("0.0")
        assert result.execution_time == 1.2

    def test_answer_question_no_documents(
        self,
        knowledge_chain,
        mock_vector_store,
        sample_query_request,
    ):
        """Test handling when no documents are found."""
        # Mock empty search results
        mock_vector_store.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        result = knowledge_chain.answer_question(sample_query_request)

        # Should return no documents response
        assert result.response_type == ResponseType.ERROR
        assert len(result.source_documents) == 0
        assert "찾을 수 없습니다" in result.response_text

    def test_answer_question_search_error(
        self,
        knowledge_chain,
        mock_vector_store,
        sample_query_request,
    ):
        """Test handling of vector search errors."""
        # Mock search error
        mock_vector_store.query.side_effect = Exception("Search failed")

        result = knowledge_chain.answer_question(sample_query_request)

        # Should return error response
        assert result.response_type == ResponseType.ERROR
        assert result.error_message is not None
        assert "검색" in result.error_message

    def test_answer_question_llm_error(
        self,
        knowledge_chain,
        mock_llm_client,
        mock_vector_store,
        sample_query_request,
        sample_search_results,
    ):
        """Test handling of LLM invocation errors."""
        # Mock successful search
        mock_vector_store.query.return_value = sample_search_results

        # Mock LLM error
        mock_llm_client.format_messages.return_value = []
        mock_llm_client.invoke.side_effect = Exception("LLM failed")

        result = knowledge_chain.answer_question(sample_query_request)

        # Should return error response
        assert result.response_type == ResponseType.ERROR
        assert "답변 생성" in result.error_message

    def test_parse_search_results(
        self,
        knowledge_chain,
        sample_search_results,
    ):
        """Test parsing of search results."""
        documents = knowledge_chain._parse_search_results(sample_search_results)

        assert len(documents) == 3
        assert all(isinstance(doc, SourceDocument) for doc in documents)

        # Check first document
        assert documents[0].doc_id == "doc1"
        assert documents[0].title == "회원가입 가이드"
        assert documents[0].relevance_score > 0.0
        assert documents[0].content is not None

    def test_parse_search_results_empty(self, knowledge_chain):
        """Test parsing empty search results."""
        empty_results = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        documents = knowledge_chain._parse_search_results(empty_results)

        assert len(documents) == 0

    def test_parse_search_results_relevance_score(self, knowledge_chain):
        """Test relevance score calculation from distance."""
        results = {
            "ids": [["doc1", "doc2"]],
            "documents": [["content1", "content2"]],
            "metadatas": [[{"title": "Doc 1"}, {"title": "Doc 2"}]],
            "distances": [[0.0, 0.5]],  # 0 = perfect match, 0.5 = moderate
        }

        documents = knowledge_chain._parse_search_results(results)

        # Lower distance = higher relevance
        assert documents[0].relevance_score == 1.0  # Perfect match
        assert documents[1].relevance_score == 0.5  # Moderate match

    def test_parse_search_results_content_truncation(self, knowledge_chain):
        """Test content truncation to 500 chars."""
        long_content = "A" * 1000  # 1000 characters

        results = {
            "ids": [["doc1"]],
            "documents": [[long_content]],
            "metadatas": [[{"title": "Long Doc"}]],
            "distances": [[0.1]],
        }

        documents = knowledge_chain._parse_search_results(results)

        # Should be truncated to 500 chars
        assert len(documents[0].content) == 500

    def test_calculate_confidence_high_relevance(self, knowledge_chain):
        """Test confidence calculation with high relevance documents."""
        documents = [
            SourceDocument(
                doc_id="doc1",
                title="Test",
                relevance_score=0.9,
            ),
            SourceDocument(
                doc_id="doc2",
                title="Test",
                relevance_score=0.95,
            ),
        ]

        confidence = knowledge_chain._calculate_confidence(documents)

        assert confidence > Decimal("0.8")

    def test_calculate_confidence_low_relevance(self, knowledge_chain):
        """Test confidence calculation with low relevance documents."""
        documents = [
            SourceDocument(
                doc_id="doc1",
                title="Test",
                relevance_score=0.3,
            ),
            SourceDocument(
                doc_id="doc2",
                title="Test",
                relevance_score=0.4,
            ),
        ]

        confidence = knowledge_chain._calculate_confidence(documents)

        assert confidence < Decimal("0.5")

    def test_calculate_confidence_empty_documents(self, knowledge_chain):
        """Test confidence calculation with no documents."""
        documents = []

        confidence = knowledge_chain._calculate_confidence(documents)

        assert confidence == Decimal("0.0")

    def test_calculate_confidence_bounds(self, knowledge_chain):
        """Test confidence is always between 0 and 1."""
        # Very high relevance
        high_docs = [
            SourceDocument(doc_id=f"doc{i}", title="Test", relevance_score=1.0)
            for i in range(5)
        ]

        confidence = knowledge_chain._calculate_confidence(high_docs)
        assert Decimal("0.0") <= confidence <= Decimal("1.0")

        # Very low relevance
        low_docs = [
            SourceDocument(doc_id=f"doc{i}", title="Test", relevance_score=0.0)
            for i in range(5)
        ]

        confidence = knowledge_chain._calculate_confidence(low_docs)
        assert Decimal("0.0") <= confidence <= Decimal("1.0")

    def test_no_documents_response(
        self,
        knowledge_chain,
        sample_query_request,
    ):
        """Test no documents response creation."""
        response = knowledge_chain._no_documents_response(sample_query_request)

        assert response.query_id == sample_query_request.query_id
        assert response.response_type == ResponseType.ERROR
        assert len(response.source_documents) == 0
        assert response.confidence_score == Decimal("0.0")

    def test_error_response_creation(
        self,
        knowledge_chain,
        sample_query_request,
    ):
        """Test error response creation."""
        error_msg = "Test error"

        response = knowledge_chain._error_response(
            sample_query_request,
            error_msg,
        )

        assert response.query_id == sample_query_request.query_id
        assert response.response_type == ResponseType.ERROR
        assert response.error_message == error_msg

"""Unit tests for Text-to-SQL Chain."""

import pytest
from decimal import Decimal
from unittest.mock import Mock, patch
from uuid import uuid4

from src.chains.text_to_sql import TextToSQLChain
from src.db.postgres import PostgreSQLClient
from src.models.query_request import QueryRequest, QueryType
from src.models.query_response import ResponseType
from src.services.llm_client import LLMClient
from src.utils.errors import DatabaseError, ValidationError


class TestTextToSQLChain:
    """Test suite for TextToSQLChain class."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        return Mock(spec=LLMClient)

    @pytest.fixture
    def mock_db_client(self):
        """Create mock database client."""
        return Mock(spec=PostgreSQLClient)

    @pytest.fixture
    def text_to_sql_chain(self, mock_llm_client, mock_db_client):
        """Create TextToSQLChain instance with mocks."""
        return TextToSQLChain(
            llm_client=mock_llm_client,
            db_client=mock_db_client,
        )

    @pytest.fixture
    def sample_query_request(self):
        """Create sample query request."""
        return QueryRequest(
            user_id="test_user",
            query_text="지난달 신규 가입자 수는?",
            query_type=QueryType.TEXT_TO_SQL,
        )

    def test_initialization(self, mock_llm_client, mock_db_client):
        """Test chain initialization."""
        chain = TextToSQLChain(
            llm_client=mock_llm_client,
            db_client=mock_db_client,
        )

        assert chain.llm_client == mock_llm_client
        assert chain.db_client == mock_db_client
        assert chain.prompts is not None

    def test_generate_sql_success(
        self,
        text_to_sql_chain,
        mock_llm_client,
        mock_db_client,
        sample_query_request,
        database_schema,
    ):
        """Test successful SQL generation."""
        # Mock database schema
        mock_db_client.get_schema_info.return_value = database_schema

        # Mock LLM response
        mock_llm_client.format_messages.return_value = []
        mock_llm_client.invoke.return_value = {
            "content": "```sql\nSELECT COUNT(*) FROM users WHERE created_at >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month');\n```",
            "execution_time": 1.5,
            "token_usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        }

        result = text_to_sql_chain.generate_sql(sample_query_request)

        # Verify response
        assert result.query_id == sample_query_request.query_id
        assert result.response_type == ResponseType.SQL_QUERY
        assert result.sql_query is not None
        assert "SELECT" in result.sql_query
        assert result.confidence_score > Decimal("0.5")
        assert result.execution_time == 1.5

    def test_generate_sql_schema_error(
        self,
        text_to_sql_chain,
        mock_db_client,
        sample_query_request,
    ):
        """Test SQL generation with schema retrieval error."""
        # Mock schema error
        mock_db_client.get_schema_info.side_effect = DatabaseError(
            message="Connection failed",
            details={},
        )

        result = text_to_sql_chain.generate_sql(sample_query_request)

        # Should return error response
        assert result.response_type == ResponseType.ERROR
        assert result.error_message is not None
        assert "스키마" in result.error_message

    def test_extract_sql_from_code_block(self, text_to_sql_chain):
        """Test SQL extraction from code block."""
        llm_response = """Here's the SQL query:

```sql
SELECT * FROM users WHERE id = 1;
```

This query will return the user."""

        sql = text_to_sql_chain._extract_sql(llm_response)

        assert sql == "SELECT * FROM users WHERE id = 1;"

    def test_extract_sql_without_code_block(self, text_to_sql_chain):
        """Test SQL extraction without code block."""
        llm_response = "SELECT * FROM users WHERE id = 1;"

        sql = text_to_sql_chain._extract_sql(llm_response)

        assert sql == "SELECT * FROM users WHERE id = 1;"

    def test_extract_sql_adds_semicolon(self, text_to_sql_chain):
        """Test SQL extraction adds semicolon if missing."""
        llm_response = "SELECT * FROM users WHERE id = 1"

        sql = text_to_sql_chain._extract_sql(llm_response)

        assert sql.endswith(";")

    def test_validate_sql_empty_query(self, text_to_sql_chain):
        """Test validation of empty SQL query."""
        with pytest.raises(ValidationError) as exc_info:
            text_to_sql_chain._validate_sql("")

        assert "empty" in str(exc_info.value).lower()

    def test_validate_sql_non_select(self, text_to_sql_chain):
        """Test validation rejects non-SELECT statements."""
        with pytest.raises(ValidationError) as exc_info:
            text_to_sql_chain._validate_sql("UPDATE users SET name = 'test';")

        assert "SELECT" in str(exc_info.value)

    def test_validate_sql_dangerous_operations(self, text_to_sql_chain):
        """Test validation rejects dangerous operations."""
        dangerous_queries = [
            "SELECT * FROM users; DROP TABLE users;",
            "SELECT * FROM users; DELETE FROM users;",
            "SELECT * FROM users; INSERT INTO users VALUES (1);",
            "SELECT * FROM users; ALTER TABLE users ADD COLUMN test;",
        ]

        for query in dangerous_queries:
            with pytest.raises(ValidationError) as exc_info:
                text_to_sql_chain._validate_sql(query)

            assert "Dangerous operation" in str(exc_info.value)

    def test_validate_sql_valid_select(self, text_to_sql_chain):
        """Test validation passes for valid SELECT."""
        valid_query = "SELECT * FROM users WHERE id = 1;"

        # Should not raise exception
        text_to_sql_chain._validate_sql(valid_query)

    def test_calculate_confidence_simple_query(self, text_to_sql_chain):
        """Test confidence calculation for simple query."""
        sql = "SELECT * FROM users;"
        full_response = "Here's your query: SELECT * FROM users;"

        confidence = text_to_sql_chain._calculate_confidence(sql, full_response)

        assert confidence >= Decimal("0.5")
        assert confidence <= Decimal("1.0")

    def test_calculate_confidence_complex_query(self, text_to_sql_chain):
        """Test confidence calculation for complex query."""
        sql = """
        SELECT u.id, u.name
        FROM users u
        JOIN orders o1 ON u.id = o1.user_id
        JOIN orders o2 ON u.id = o2.user_id
        JOIN orders o3 ON u.id = o3.user_id
        JOIN orders o4 ON u.id = o4.user_id
        """
        full_response = "Here's the query"

        confidence = text_to_sql_chain._calculate_confidence(sql, full_response)

        # Should be lower due to complexity
        assert confidence < Decimal("0.9")

    def test_calculate_confidence_uncertain_response(self, text_to_sql_chain):
        """Test confidence calculation with uncertainty words."""
        sql = "SELECT * FROM users;"
        full_response = "I'm not sure, but maybe this query might work"

        confidence = text_to_sql_chain._calculate_confidence(sql, full_response)

        # Should be lower due to uncertainty
        assert confidence < Decimal("0.8")

    def test_format_schema(self, text_to_sql_chain, database_schema):
        """Test schema formatting."""
        formatted = text_to_sql_chain._format_schema(database_schema)

        assert "Table: users" in formatted
        assert "Table: orders" in formatted
        assert "user_id" in formatted
        assert "email" in formatted

    def test_default_few_shot_examples(self, text_to_sql_chain):
        """Test default few-shot examples."""
        examples = text_to_sql_chain._default_few_shot_examples()

        assert "Example" in examples
        assert "지난달" in examples
        assert "SELECT" in examples

    def test_error_response_creation(
        self,
        text_to_sql_chain,
        sample_query_request,
    ):
        """Test error response creation."""
        error_msg = "Test error message"

        response = text_to_sql_chain._error_response(
            sample_query_request,
            error_msg,
        )

        assert response.query_id == sample_query_request.query_id
        assert response.response_type == ResponseType.ERROR
        assert response.error_message == error_msg
        assert response.confidence_score == Decimal("0.0")

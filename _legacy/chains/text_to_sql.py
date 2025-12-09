"""Text-to-SQL chain using Claude Code."""

import re
from decimal import Decimal
from typing import Dict, Optional

from src.db.postgres import PostgreSQLClient
from src.models.query_request import QueryRequest
from src.models.query_response import QueryResponse, ResponseType
from src.services.llm_client import LLMClient
from src.utils.errors import DatabaseError, ValidationError
from src.utils.logging import logger
from src.utils.prompts import PromptTemplates


class TextToSQLChain:
    """
    Text-to-SQL chain using Claude Code for SQL generation.

    Generates PostgreSQL queries from Korean natural language questions.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        db_client: PostgreSQLClient,
    ):
        """
        Initialize Text-to-SQL chain.

        Args:
            llm_client: LLM client instance
            db_client: PostgreSQL client instance
        """
        self.llm_client = llm_client
        self.db_client = db_client
        self.prompts = PromptTemplates()

    def generate_sql(
        self,
        query_request: QueryRequest,
        few_shot_examples: Optional[str] = None,
    ) -> QueryResponse:
        """
        Generate SQL query from natural language question.

        Args:
            query_request: User query request
            few_shot_examples: Optional few-shot examples for context

        Returns:
            Query response with generated SQL

        Raises:
            LLMAPIError: If SQL generation fails
            DatabaseError: If schema retrieval fails
        """
        logger.info(f"Generating SQL for query: {query_request.query_id}")

        # Get database schema
        try:
            schema_info = self.db_client.get_schema_info()
            schema_text = self._format_schema(schema_info)
        except DatabaseError as e:
            logger.error(f"Failed to retrieve schema: {e}")
            return self._error_response(
                query_request,
                f"데이터베이스 스키마를 가져오는데 실패했습니다: {e.message}",
            )

        # Use default few-shot examples if not provided
        if few_shot_examples is None:
            few_shot_examples = self._default_few_shot_examples()

        # Format messages
        messages = self.llm_client.format_messages(
            system_prompt=self.prompts.text_to_sql_system_prompt(),
            user_prompt=self.prompts.text_to_sql_user_prompt(
                schema=schema_text,
                few_shot_examples=few_shot_examples,
                user_question=query_request.query_text,
            ),
        )

        # Invoke LLM
        try:
            result = self.llm_client.invoke(messages)
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            return self._error_response(
                query_request,
                f"SQL 생성 중 오류가 발생했습니다: {str(e)}",
            )

        # Extract SQL from response
        sql_query = self._extract_sql(result["content"])

        # Validate SQL
        try:
            self._validate_sql(sql_query)
        except ValidationError as e:
            logger.error(f"SQL validation failed: {e}")
            return self._error_response(
                query_request,
                f"생성된 SQL이 유효하지 않습니다: {e.message}",
            )

        # Calculate confidence score (simple heuristic)
        confidence_score = self._calculate_confidence(sql_query, result["content"])

        # Create response
        response = QueryResponse(
            query_id=query_request.query_id,
            response_text=f"다음 SQL 쿼리를 실행하세요:\n\n```sql\n{sql_query}\n```",
            response_type=ResponseType.SQL_QUERY,
            sql_query=sql_query,
            confidence_score=confidence_score,
            execution_time=result["execution_time"],
            token_usage=result["token_usage"],
        )

        logger.info(
            f"SQL generated successfully: {len(sql_query)} chars, "
            f"confidence: {confidence_score}"
        )

        return response

    def _format_schema(self, schema_info: Dict[str, list]) -> str:
        """Format schema information for prompt."""
        lines = []
        for table_name, columns in schema_info.items():
            lines.append(f"Table: {table_name}")
            for column in columns:
                lines.append(f"  - {column}")
            lines.append("")  # Empty line between tables
        return "\n".join(lines)

    def _default_few_shot_examples(self) -> str:
        """Provide default few-shot examples."""
        return """Example 1:
Question: 지난달 신규 가입자 수는?
SQL: SELECT COUNT(*) FROM users WHERE created_at >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month') AND created_at < DATE_TRUNC('month', CURRENT_DATE);

Example 2:
Question: 이번주 매출 총액은?
SQL: SELECT SUM(amount) FROM orders WHERE created_at >= DATE_TRUNC('week', CURRENT_DATE);

Example 3:
Question: 상위 10명의 고객과 그들의 총 구매액은?
SQL: SELECT user_id, SUM(amount) as total_amount FROM orders GROUP BY user_id ORDER BY total_amount DESC LIMIT 10;"""

    def _extract_sql(self, llm_response: str) -> str:
        """Extract SQL query from LLM response."""
        # Try to extract from code block
        code_block_pattern = r"```sql\s*(.*?)\s*```"
        match = re.search(code_block_pattern, llm_response, re.DOTALL | re.IGNORECASE)

        if match:
            return match.group(1).strip()

        # Try to extract SELECT statement
        select_pattern = r"(SELECT\s+.*?;?)\s*$"
        match = re.search(select_pattern, llm_response, re.DOTALL | re.IGNORECASE)

        if match:
            sql = match.group(1).strip()
            # Add semicolon if missing
            if not sql.endswith(";"):
                sql += ";"
            return sql

        # Return entire response as fallback
        return llm_response.strip()

    def _validate_sql(self, sql_query: str) -> None:
        """
        Validate SQL query.

        Args:
            sql_query: SQL query string

        Raises:
            ValidationError: If SQL is invalid
        """
        # Basic validation checks
        if not sql_query:
            raise ValidationError(
                message="SQL query is empty",
                details={"sql": sql_query},
            )

        # Must be a SELECT statement
        if not sql_query.upper().strip().startswith("SELECT"):
            raise ValidationError(
                message="Only SELECT statements are allowed",
                details={"sql": sql_query},
            )

        # Check for dangerous operations using word boundaries
        dangerous_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE"]
        sql_upper = sql_query.upper()
        for keyword in dangerous_keywords:
            # Use word boundary matching to avoid false positives like "CURRENT_DATE" matching "CREATE"
            if re.search(r'\b' + keyword + r'\b', sql_upper):
                raise ValidationError(
                    message=f"Dangerous operation detected: {keyword}",
                    details={"sql": sql_query},
                )

    def _calculate_confidence(self, sql_query: str, full_response: str) -> Decimal:
        """Calculate confidence score based on SQL characteristics."""
        confidence = Decimal("0.9")  # Base confidence

        # Reduce confidence for complex queries
        if sql_query.count("JOIN") > 3:
            confidence -= Decimal("0.1")

        # Reduce confidence if response contains uncertainty
        uncertainty_words = ["might", "maybe", "possibly", "uncertain"]
        if any(word in full_response.lower() for word in uncertainty_words):
            confidence -= Decimal("0.2")

        # Ensure confidence is in valid range
        confidence = max(Decimal("0.5"), min(Decimal("1.0"), confidence))

        return confidence

    def _error_response(
        self,
        query_request: QueryRequest,
        error_message: str,
    ) -> QueryResponse:
        """Create error response."""
        return QueryResponse(
            query_id=query_request.query_id,
            response_text=error_message,
            response_type=ResponseType.ERROR,
            confidence_score=Decimal("0.0"),
            execution_time=0.01,
            token_usage={"input_tokens": 0, "output_tokens": 0},
            error_message=error_message,
        )

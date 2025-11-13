"""Router chain for intent classification."""

from typing import Dict

from src.models.query_request import QueryRequest, QueryType
from src.services.llm_client import LLMClient
from src.utils.logging import logger
from src.utils.prompts import PromptTemplates


class RouterChain:
    """
    Intent classification chain using Claude Code.

    Classifies user queries into: text_to_sql, knowledge, or assistant.
    """

    def __init__(self, llm_client: LLMClient):
        """
        Initialize router chain.

        Args:
            llm_client: LLM client instance
        """
        self.llm_client = llm_client
        self.prompts = PromptTemplates()

    def classify(self, query_request: QueryRequest) -> QueryType:
        """
        Classify query intent.

        Args:
            query_request: User query request

        Returns:
            Classified query type

        Raises:
            LLMAPIError: If classification fails
        """
        logger.info(f"Classifying query: {query_request.query_id}")

        # Format messages
        messages = self.llm_client.format_messages(
            system_prompt=self.prompts.router_system_prompt(),
            user_prompt=self.prompts.router_user_prompt(query_request.query_text),
        )

        # Invoke LLM
        result = self.llm_client.invoke(messages)
        classification = result["content"].strip().lower()

        logger.info(f"Token usage: {result['token_usage'].total_tokens} tokens")

        # Parse classification
        if "text_to_sql" in classification or "sql" in classification:
            query_type = QueryType.TEXT_TO_SQL
        elif "knowledge" in classification or "document" in classification:
            query_type = QueryType.KNOWLEDGE
        else:
            query_type = QueryType.ASSISTANT

        logger.info(f"Query classified as: {query_type}")

        # Update query request
        query_request.query_type = query_type

        return query_type

    def route(self, query_request: QueryRequest) -> Dict[str, str]:
        """
        Classify and return routing decision.

        Args:
            query_request: User query request

        Returns:
            Dictionary with query_type and next_chain
        """
        query_type = self.classify(query_request)

        routing = {
            QueryType.TEXT_TO_SQL: "text_to_sql_chain",
            QueryType.KNOWLEDGE: "knowledge_chain",
            QueryType.ASSISTANT: "multi_turn_chain",
        }

        next_chain = routing[query_type]

        logger.info(f"Routing to: {next_chain}")

        return {
            "query_type": query_type.value,
            "next_chain": next_chain,
        }

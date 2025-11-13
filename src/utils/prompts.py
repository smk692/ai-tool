"""Prompt template utilities for LLM chains."""

from typing import Dict, List, Optional


class PromptTemplates:
    """Collection of prompt templates for different AI tasks."""

    @staticmethod
    def text_to_sql_system_prompt() -> str:
        """System prompt for Text-to-SQL chain."""
        return """You are a SQL expert for PostgreSQL databases. Your task is to generate accurate SQL queries from natural language questions in Korean.

Follow these guidelines:
1. Generate syntactically correct PostgreSQL queries
2. Use appropriate JOINs, WHERE clauses, and aggregations
3. Consider date/time handling for Korean date formats (e.g., "지난달" = last month)
4. Always include proper column aliases for readability
5. Validate table and column names against the provided schema
6. Return ONLY the SQL query, without explanations

Remember: Read-only access - use SELECT statements only, no INSERT/UPDATE/DELETE."""

    @staticmethod
    def text_to_sql_user_prompt(
        schema: str,
        few_shot_examples: str,
        user_question: str,
    ) -> str:
        """
        User prompt for Text-to-SQL with Claude XML format.

        Args:
            schema: Database schema description
            few_shot_examples: Example queries for context
            user_question: User's natural language question in Korean

        Returns:
            Formatted prompt string
        """
        return f"""<schema>
{schema}
</schema>

<examples>
{few_shot_examples}
</examples>

<question>
{user_question}
</question>

Generate a PostgreSQL query to answer the question above. Think step by step:
1. Identify the relevant tables and columns
2. Determine the required JOINs
3. Add appropriate WHERE clauses and filters
4. Apply any necessary aggregations
5. Generate the final SQL query

SQL Query:"""

    @staticmethod
    def knowledge_discovery_system_prompt() -> str:
        """System prompt for Knowledge Discovery chain."""
        return """You are a helpful assistant that answers questions based on retrieved documents. Always cite your sources and provide accurate information based on the given context.

Follow these guidelines:
1. Answer in Korean (한국어)
2. Use information ONLY from the provided documents
3. If the documents don't contain the answer, say "제공된 문서에서 답변을 찾을 수 없습니다"
4. Always cite document titles when referencing information
5. Be concise but complete in your answers"""

    @staticmethod
    def knowledge_discovery_user_prompt(
        documents: List[Dict[str, str]],
        user_question: str,
    ) -> str:
        """
        User prompt for Knowledge Discovery with retrieved documents.

        Args:
            documents: List of retrieved documents with title and content
            user_question: User's question in Korean

        Returns:
            Formatted prompt string
        """
        docs_text = "\n\n".join(
            [
                f"<document id=\"{i+1}\" title=\"{doc['title']}\">\n{doc['content']}\n</document>"
                for i, doc in enumerate(documents)
            ]
        )

        return f"""<documents>
{docs_text}
</documents>

<question>
{user_question}
</question>

Based on the documents provided above, answer the question. Include citations to specific documents."""

    @staticmethod
    def router_system_prompt() -> str:
        """System prompt for intent classification."""
        return """You are an intent classifier for an AI assistant. Your task is to classify user queries into one of three categories:

1. text_to_sql - Questions about database data, metrics, or analytics (e.g., "지난달 매출은?", "신규 가입자 수는?")
2. knowledge - Questions about documents, processes, or general knowledge (e.g., "회원가입 절차가 어떻게 되나요?")
3. assistant - General conversation or unclear requests (e.g., "안녕하세요", "도움이 필요해요")

Respond with ONLY the category name, nothing else."""

    @staticmethod
    def router_user_prompt(user_query: str) -> str:
        """User prompt for intent classification."""
        return f"""Classify the following query:

<query>
{user_query}
</query>

Category:"""

    @staticmethod
    def multi_turn_system_prompt() -> str:
        """System prompt for multi-turn conversation."""
        return """You are a helpful AI assistant that provides natural, conversational responses.

Guidelines:
- Provide clear, accurate, and helpful responses
- Maintain context from previous conversation turns
- Ask clarifying questions when user intent is unclear
- Admit when you don't know something
- Be concise but thorough
- Use Korean language for responses unless otherwise specified
- Maintain a friendly and professional tone"""

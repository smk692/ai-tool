"""Multi-turn conversation chain using Claude Code."""

from decimal import Decimal
from typing import List, Optional

from src.db.sqlite import SQLiteConversationMemory
from src.models.query_request import QueryRequest
from src.models.query_response import QueryResponse, ResponseType
from src.services.llm_client import LLMClient
from src.utils.logging import logger
from src.utils.prompts import PromptTemplates


class MultiTurnChain:
    """
    Multi-turn conversation chain using Claude Code.

    Maintains conversation history and provides contextual responses.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        memory: SQLiteConversationMemory,
    ):
        """
        Initialize Multi-turn conversation chain.

        Args:
            llm_client: LLM client instance
            memory: SQLite conversation memory instance
        """
        self.llm_client = llm_client
        self.memory = memory
        self.prompts = PromptTemplates()

    def chat(
        self,
        query_request: QueryRequest,
        max_history: int = 10,
    ) -> QueryResponse:
        """
        Generate conversational response with history context.

        Args:
            query_request: User query request
            max_history: Maximum conversation history to include

        Returns:
            Query response with conversational answer

        Raises:
            LLMAPIError: If response generation fails
        """
        logger.info(f"Processing conversation query: {query_request.query_id}")

        # Get conversation history if session exists
        conversation_history = []
        if query_request.session_id:
            try:
                history_messages = self.memory.get_conversation_history(
                    session_id=str(query_request.session_id),
                    limit=max_history,
                )
                conversation_history = self._format_history(history_messages)
            except Exception as e:
                logger.warning(f"Failed to retrieve conversation history: {e}")
                # Continue without history

        # Format messages with conversation context
        messages = self.llm_client.format_messages(
            system_prompt=self.prompts.multi_turn_system_prompt(),
            user_prompt=query_request.query_text,
            conversation_history=conversation_history,
        )

        # Invoke LLM
        try:
            result = self.llm_client.invoke(messages)
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            return self._error_response(
                query_request,
                f"대화 생성 중 오류가 발생했습니다: {str(e)}",
            )

        # Calculate confidence score (simple heuristic)
        confidence_score = self._calculate_confidence(result["content"])

        # Create response
        response = QueryResponse(
            query_id=query_request.query_id,
            response_text=result["content"],
            response_type=ResponseType.ASSISTANT_MESSAGE,
            confidence_score=confidence_score,
            execution_time=result["execution_time"],
            token_usage=result["token_usage"],
        )

        # Save conversation to memory if session exists
        if query_request.session_id:
            try:
                self.memory.save_conversation(
                    session_id=str(query_request.session_id),
                    user_message=query_request.query_text,
                    assistant_message=result["content"],
                    metadata={
                        "query_id": str(query_request.query_id),
                        "response_id": str(response.response_id),
                        "token_usage": result["token_usage"].model_dump(),
                        "confidence_score": float(confidence_score),
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to save conversation: {e}")
                # Continue even if save fails

        logger.info(
            f"Conversation response generated: {result['token_usage'].total_tokens} tokens, "
            f"confidence: {confidence_score}"
        )

        return response

    def _format_history(
        self,
        history_messages: List[dict],
    ) -> List[dict]:
        """
        Format conversation history for LLM context.

        Args:
            history_messages: Raw history from memory

        Returns:
            Formatted conversation history
        """
        formatted = []

        for msg in history_messages:
            # Add user message
            if msg.get("user_message"):
                formatted.append({
                    "role": "user",
                    "content": msg["user_message"],
                })

            # Add assistant message
            if msg.get("assistant_message"):
                formatted.append({
                    "role": "assistant",
                    "content": msg["assistant_message"],
                })

        return formatted

    def _calculate_confidence(self, response_text: str) -> Decimal:
        """Calculate confidence score based on response characteristics."""
        confidence = Decimal("0.8")  # Base confidence for conversations

        # Reduce confidence for short responses
        if len(response_text) < 50:
            confidence -= Decimal("0.1")

        # Reduce confidence if response contains uncertainty
        uncertainty_words = [
            "잘 모르겠",
            "확실하지 않",
            "아마도",
            "possibly",
            "maybe",
            "might",
            "uncertain",
        ]
        if any(word in response_text.lower() for word in uncertainty_words):
            confidence -= Decimal("0.2")

        # Ensure confidence is in valid range
        confidence = max(Decimal("0.3"), min(Decimal("1.0"), confidence))

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
            execution_time=0.0,
            token_usage={"input_tokens": 0, "output_tokens": 0},
            error_message=error_message,
        )

    def clear_session(self, session_id: str) -> bool:
        """
        Clear conversation history for a session.

        Args:
            session_id: Session identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            self.memory.clear_session(session_id)
            logger.info(f"Cleared conversation history for session: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear session: {e}")
            return False

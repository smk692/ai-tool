"""Knowledge Discovery chain using Claude Code and vector search."""

from decimal import Decimal
from typing import Dict, List

from src.models.query_request import QueryRequest
from src.models.query_response import QueryResponse, ResponseType, SourceDocument
from src.services.llm_client import LLMClient
from src.services.vector_store import VectorStore
from src.utils.logging import logger
from src.utils.prompts import PromptTemplates


class KnowledgeChain:
    """
    Knowledge Discovery chain using Claude Code and RAG.

    Retrieves relevant documents and generates answers with citations.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        vector_store: VectorStore,
    ):
        """
        Initialize Knowledge Discovery chain.

        Args:
            llm_client: LLM client instance
            vector_store: Vector store instance
        """
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.prompts = PromptTemplates()

    def answer_question(
        self,
        query_request: QueryRequest,
        n_results: int = 5,
    ) -> QueryResponse:
        """
        Answer question using retrieved documents.

        Args:
            query_request: User query request
            n_results: Number of documents to retrieve

        Returns:
            Query response with answer and source citations

        Raises:
            VectorStoreError: If vector search fails
            LLMAPIError: If answer generation fails
        """
        logger.info(f"Answering knowledge query: {query_request.query_id}")

        # Retrieve relevant documents
        try:
            search_results = self.vector_store.query(
                query_texts=[query_request.query_text],
                n_results=n_results,
            )
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return self._error_response(
                query_request,
                f"문서 검색 중 오류가 발생했습니다: {str(e)}",
            )

        # Parse search results
        documents = self._parse_search_results(search_results)

        if not documents:
            logger.warning("No relevant documents found")
            return self._no_documents_response(query_request)

        # Format documents for prompt
        docs_for_prompt = [
            {
                "title": doc.title,
                "content": doc.content or "",
            }
            for doc in documents
        ]

        # Format messages
        messages = self.llm_client.format_messages(
            system_prompt=self.prompts.knowledge_discovery_system_prompt(),
            user_prompt=self.prompts.knowledge_discovery_user_prompt(
                documents=docs_for_prompt,
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
                f"답변 생성 중 오류가 발생했습니다: {str(e)}",
            )

        # Calculate confidence score based on document relevance
        confidence_score = self._calculate_confidence(documents)

        # Create response
        response = QueryResponse(
            query_id=query_request.query_id,
            response_text=result["content"],
            response_type=ResponseType.DOCUMENT_ANSWER,
            source_documents=documents,
            confidence_score=confidence_score,
            execution_time=result["execution_time"],
            token_usage=result["token_usage"],
        )

        logger.info(
            f"Answer generated successfully: {len(documents)} sources, "
            f"confidence: {confidence_score}"
        )

        return response

    def _parse_search_results(
        self,
        search_results: Dict,
    ) -> List[SourceDocument]:
        """
        Parse ChromaDB search results.

        Args:
            search_results: Raw search results from ChromaDB

        Returns:
            List of SourceDocument objects
        """
        documents = []

        # ChromaDB returns results as lists
        ids_list = search_results.get("ids", [[]])[0]
        documents_list = search_results.get("documents", [[]])[0]
        metadatas_list = search_results.get("metadatas", [[]])[0]
        distances_list = search_results.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids_list):
            # Convert distance to relevance score (0-1)
            # Cosine distance: lower is better, convert to similarity
            distance = distances_list[i] if i < len(distances_list) else 1.0
            relevance_score = max(0.0, 1.0 - distance)

            metadata = metadatas_list[i] if i < len(metadatas_list) else {}
            content = documents_list[i] if i < len(documents_list) else ""

            source_doc = SourceDocument(
                doc_id=doc_id,
                title=metadata.get("title", f"Document {doc_id}"),
                relevance_score=relevance_score,
                content=content[:500],  # Truncate for response
            )
            documents.append(source_doc)

        return documents

    def _calculate_confidence(self, documents: List[SourceDocument]) -> Decimal:
        """Calculate confidence score based on document relevance."""
        if not documents:
            return Decimal("0.0")

        # Average relevance score of top documents
        avg_relevance = sum(doc.relevance_score for doc in documents) / len(documents)

        # Convert to Decimal
        confidence = Decimal(str(avg_relevance))

        # Ensure confidence is in valid range
        confidence = max(Decimal("0.0"), min(Decimal("1.0"), confidence))

        return confidence

    def _no_documents_response(
        self,
        query_request: QueryRequest,
    ) -> QueryResponse:
        """Create response when no documents are found."""
        return QueryResponse(
            query_id=query_request.query_id,
            response_text="죄송합니다. 제공된 문서에서 관련 정보를 찾을 수 없습니다.",
            response_type=ResponseType.ERROR,
            source_documents=[],
            confidence_score=Decimal("0.0"),
            execution_time=0.01,  # Small non-zero value for validation
            token_usage={"input_tokens": 0, "output_tokens": 0},
            error_message="No relevant documents found",
        )

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
            execution_time=0.01,  # Small non-zero value for validation
            token_usage={"input_tokens": 0, "output_tokens": 0},
            error_message=error_message,
        )

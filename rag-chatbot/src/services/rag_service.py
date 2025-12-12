"""RAG 서비스 모듈.

벡터DB 검색과 Claude LLM을 결합하여 질문에 답변을 생성합니다.
"""

import logging
import time
from typing import Any

from shared.embedding import get_embedding_model
from shared.vector_store import get_vector_store

from ..config import get_settings
from ..llm import ClaudeClient, build_rag_prompt, truncate_context
from ..models import Query, Response, SearchResult, SourceReference

logger = logging.getLogger(__name__)


class RAGService:
    """RAG 파이프라인 서비스.

    질문을 받아 벡터DB에서 관련 문서를 검색하고,
    Claude LLM을 통해 답변을 생성합니다.
    """

    def __init__(
        self,
        claude_client: ClaudeClient | None = None,
    ) -> None:
        """RAG 서비스 초기화.

        Args:
            claude_client: Claude 클라이언트 (None이면 자동 생성)
        """
        self.settings = get_settings()

        # 임베딩 모델
        self._embedding_model = get_embedding_model()

        # 벡터 스토어
        self._vector_store = get_vector_store(
            host=self.settings.qdrant_host,
            port=self.settings.qdrant_port,
            collection_name=self.settings.qdrant_collection,
        )

        # Claude 클라이언트
        self._claude_client = claude_client or ClaudeClient()

        logger.info("RAG 서비스 초기화 완료")

    async def answer(
        self,
        query: Query,
        conversation_context: str | None = None,
    ) -> Response:
        """질문에 대한 답변 생성.

        Args:
            query: 사용자 질문
            conversation_context: 이전 대화 컨텍스트 (선택)

        Returns:
            생성된 답변
        """
        start_time = time.time()

        try:
            # 1. 벡터 검색
            search_results = self._search_documents(query.text)
            logger.info(
                "검색 완료: %d개 결과 (threshold=%s)",
                len(search_results),
                self.settings.rag_score_threshold,
            )

            # 2. 관련성 있는 결과만 필터링
            relevant_results = [r for r in search_results if r.is_relevant]

            if not relevant_results:
                logger.info("관련 문서 없음 - 폴백 응답 생성")
                return Response.fallback_response("관련 문서를 찾을 수 없습니다.")

            # 3. 토큰 제한에 맞게 컨텍스트 자르기
            truncated_results = truncate_context(
                relevant_results,
                max_tokens=self.settings.rag_max_context_tokens,
            )

            # 4. 프롬프트 생성
            prompt = build_rag_prompt(
                question=query.text,
                search_results=truncated_results,
                conversation_context=conversation_context,
            )

            # 5. Claude LLM 호출
            llm_response = await self._claude_client.generate(prompt)

            # 6. 소스 참조 생성
            sources = self._build_source_references(truncated_results)

            # 7. 응답 생성
            generation_time_ms = int((time.time() - start_time) * 1000)

            response = Response(
                text=llm_response.text,
                sources=sources,
                model=llm_response.model,
                tokens_used=llm_response.tokens_used,
                generation_time_ms=generation_time_ms,
            )

            logger.info(
                f"답변 생성 완료: {llm_response.tokens_used} tokens, {generation_time_ms}ms"
            )

            return response

        except Exception as e:
            logger.error(f"답변 생성 실패: {e}", exc_info=True)
            return Response.fallback_response("답변 생성 중 오류가 발생했습니다.")

    def _search_documents(self, query_text: str) -> list[SearchResult]:
        """벡터DB에서 문서 검색.

        Args:
            query_text: 검색 쿼리 텍스트

        Returns:
            검색 결과 목록
        """
        # 쿼리 임베딩 생성
        query_embedding = self._embedding_model.embed_query(query_text)

        # 벡터 검색
        results = self._vector_store.search(
            query_vector=query_embedding,
            limit=self.settings.rag_top_k,
            score_threshold=self.settings.rag_score_threshold,
        )

        # SearchResult 모델로 변환
        search_results = []
        for result in results:
            try:
                search_result = SearchResult.from_qdrant_result(
                    point_id=result["id"],
                    score=result["score"],
                    payload=result["payload"],
                )
                search_results.append(search_result)
            except Exception as e:
                logger.warning(f"검색 결과 파싱 실패: {e}")
                continue

        return search_results

    def _build_source_references(
        self,
        search_results: list[SearchResult],
    ) -> list[SourceReference]:
        """소스 참조 목록 생성.

        중복 제거 및 정렬을 수행합니다.

        Args:
            search_results: 검색 결과 목록

        Returns:
            소스 참조 목록
        """
        # source_id 기준 중복 제거
        seen_sources: dict[str, SearchResult] = {}
        for result in search_results:
            if result.source_id not in seen_sources:
                seen_sources[result.source_id] = result

        # SourceReference 변환
        sources = [
            SourceReference(
                title=result.source_title,
                url=result.source_url,
                source_type=result.source_type,
            )
            for result in seen_sources.values()
        ]

        return sources

    async def search_only(
        self,
        query_text: str,
    ) -> list[SearchResult]:
        """검색만 수행 (LLM 호출 없이).

        디버깅 또는 검색 결과 미리보기용.

        Args:
            query_text: 검색 쿼리 텍스트

        Returns:
            검색 결과 목록
        """
        return self._search_documents(query_text)

    def health_check(self) -> dict[str, Any]:
        """서비스 상태 확인.

        Returns:
            상태 정보 딕셔너리
        """
        try:
            collection_info = self._vector_store.get_collection_info()
            return {
                "status": "healthy",
                "vector_store": {
                    "connected": True,
                    "collection": collection_info,
                },
                "embedding_model": {
                    "name": self._embedding_model.model_name,
                    "dimension": self._embedding_model.dimension,
                },
            }
        except Exception as e:
            logger.error(f"헬스체크 실패: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
            }


# 기본 인스턴스 (지연 로딩)
_default_service: RAGService | None = None


def get_rag_service() -> RAGService:
    """RAG 서비스 싱글톤 인스턴스 반환.

    Returns:
        RAGService 인스턴스
    """
    global _default_service
    if _default_service is None:
        _default_service = RAGService()
    return _default_service

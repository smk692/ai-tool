"""AI 기반 메타데이터 추출 서비스.

Claude를 사용하여 문서 내용에서 구조화된 메타데이터를 추출합니다.

주요 기능:
    - 콘텐츠 유형 자동 분류 (API 문서, 가이드, FAQ 등)
    - 주요 토픽 및 핵심 엔티티 추출
    - 난이도 수준 평가
    - 코드 샘플 포함 여부 감지
    - 문서 요약 생성

Phase 1.2 기능으로, AI 메타데이터를 통해 검색 품질을 개선합니다.
"""

import hashlib
import json

import structlog
from pydantic import BaseModel, Field

from ..config import settings

logger = structlog.get_logger()


class ExtractedMetadata(BaseModel):
    """AI가 추출한 구조화된 메타데이터.

    문서의 내용을 분석하여 추출한 메타데이터로,
    검색 필터링 및 랭킹에 활용됩니다.

    Attributes:
        content_type: 콘텐츠 유형.
            api_doc, guide, faq, tutorial, reference, general 중 하나.
        topics: 문서에서 다루는 주요 토픽 목록.
            최대 5개까지 추출됩니다.
        difficulty: 콘텐츠 난이도.
            beginner, intermediate, advanced 중 하나.
        has_code_samples: 코드 예제 포함 여부.
        key_entities: 핵심 기술 엔티티 목록.
            API 이름, 클래스, 함수 등 최대 10개.
        summary: 내용 요약 (1-2문장).
    """

    content_type: str = Field(
        default="general",
        description="콘텐츠 유형: api_doc, guide, faq, tutorial, reference, general",
    )
    topics: list[str] = Field(
        default_factory=list,
        description="문서에서 다루는 주요 토픽",
    )
    difficulty: str = Field(
        default="intermediate",
        description="콘텐츠 난이도: beginner, intermediate, advanced",
    )
    has_code_samples: bool = Field(
        default=False,
        description="코드 예제 포함 여부",
    )
    key_entities: list[str] = Field(
        default_factory=list,
        description="핵심 기술 엔티티 (API, 클래스, 함수)",
    )
    summary: str = Field(
        default="",
        description="내용 요약 (1-2문장)",
    )


# Claude에게 전달할 메타데이터 추출 프롬프트
EXTRACTION_PROMPT = """Analyze the following document and extract structured metadata.

Document Title: {title}
Document Content:
{content}

Extract the following information as JSON:
1. content_type: One of "api_doc", "guide", "faq", "tutorial", "reference", "general"
2. topics: List of main topics (max 5)
3. difficulty: One of "beginner", "intermediate", "advanced"
4. has_code_samples: true/false
5. key_entities: List of key technical entities like API names, class names, function names (max 10)
6. summary: Brief 1-2 sentence summary

Respond ONLY with valid JSON matching this schema:
{{
    "content_type": "string",
    "topics": ["string"],
    "difficulty": "string",
    "has_code_samples": boolean,
    "key_entities": ["string"],
    "summary": "string"
}}"""


class AIExtractor:
    """AI 기반 메타데이터 추출 서비스.

    Claude를 사용하여 문서에서 구조화된 메타데이터를 추출합니다.
    동일한 내용에 대한 반복 API 호출을 방지하기 위해 캐싱을 지원합니다.

    주요 특징:
        - Claude API를 통한 고품질 메타데이터 추출
        - 콘텐츠 해시 기반 캐싱으로 비용 절감
        - 마크다운 코드 블록 자동 처리
        - 긴 문서 자동 truncate

    Attributes:
        _api_key: Anthropic API 키.
        _model: 사용할 Claude 모델 이름.
        _max_tokens: 응답 최대 토큰 수.
        _timeout: 요청 타임아웃 (초).
        _cache_enabled: 캐싱 활성화 여부.
        _cache: 추출 결과 캐시 딕셔너리.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
        timeout: float = 30.0,
        cache_enabled: bool = True,
    ):
        """AI 추출기를 초기화합니다.

        Args:
            api_key: Anthropic API 키.
                None이면 settings에서 가져옵니다.
            model: 사용할 Claude 모델.
                None이면 settings에서 가져옵니다.
            max_tokens: 응답 최대 토큰 수.
            timeout: 요청 타임아웃 (초).
            cache_enabled: 추출 결과 캐싱 활성화 여부.
        """
        self._api_key = api_key or settings.ai.api_key
        self._model = model or settings.ai.model
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._cache_enabled = cache_enabled
        self._cache: dict[str, ExtractedMetadata] = {}
        self._client = None

    @property
    def client(self):
        """Anthropic 클라이언트를 지연 로딩합니다.

        첫 호출 시에만 클라이언트를 생성하여
        초기화 시간과 리소스를 절약합니다.

        Returns:
            Anthropic 클라이언트 인스턴스.

        Raises:
            ImportError: anthropic 패키지가 설치되지 않은 경우.
        """
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.Anthropic(
                    api_key=self._api_key,
                    timeout=self._timeout,
                )
            except ImportError:
                raise ImportError(
                    "anthropic 패키지가 필요합니다. "
                    "다음 명령어로 설치하세요: pip install anthropic"
                )
        return self._client

    def _get_cache_key(self, title: str, content: str) -> str:
        """콘텐츠 해시에서 캐시 키를 생성합니다.

        긴 문서의 경우 처음 5000자만 사용하여
        캐시 키를 생성합니다.

        Args:
            title: 문서 제목.
            content: 문서 내용.

        Returns:
            16자 해시 문자열.
        """
        combined = f"{title}:{content[:5000]}"  # 처음 5000자 사용
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def _parse_response(self, response_text: str) -> ExtractedMetadata:
        """Claude의 JSON 응답을 ExtractedMetadata로 파싱합니다.

        마크다운 코드 블록(```json)으로 감싸진 응답도
        올바르게 처리합니다.

        Args:
            response_text: Claude의 응답 텍스트.

        Returns:
            파싱된 ExtractedMetadata 객체.
            파싱 실패 시 기본값이 채워진 객체 반환.
        """
        try:
            # 응답에서 JSON 추출 시도
            text = response_text.strip()

            # 마크다운 코드 블록 처리
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            data = json.loads(text)

            # 검증 및 정규화
            return ExtractedMetadata(
                content_type=data.get("content_type", "general"),
                topics=data.get("topics", [])[:5],  # 최대 5개 토픽
                difficulty=data.get("difficulty", "intermediate"),
                has_code_samples=bool(data.get("has_code_samples", False)),
                key_entities=data.get("key_entities", [])[:10],  # 최대 10개 엔티티
                summary=data.get("summary", "")[:500],  # 최대 500자
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("AI 응답 파싱 실패", error=str(e))
            return ExtractedMetadata()

    def extract(
        self,
        title: str,
        content: str,
        max_content_length: int = 8000,
    ) -> ExtractedMetadata:
        """문서 내용에서 메타데이터를 추출합니다.

        캐시를 먼저 확인하고, 캐시 미스 시 Claude API를 호출합니다.
        긴 문서는 max_content_length로 truncate됩니다.

        Args:
            title: 문서 제목.
            content: 문서 내용.
            max_content_length: API에 전송할 최대 내용 길이.
                기본값 8000자로, Claude 컨텍스트를 고려한 설정.

        Returns:
            구조화된 정보가 담긴 ExtractedMetadata.
            API 호출 실패 시 기본값이 채워진 객체 반환.
        """
        # 먼저 캐시 확인
        cache_key = self._get_cache_key(title, content)
        if self._cache_enabled and cache_key in self._cache:
            logger.debug("AI 추출 캐시 히트", cache_key=cache_key)
            return self._cache[cache_key]

        # 내용이 너무 길면 truncate
        truncated_content = content[:max_content_length]
        if len(content) > max_content_length:
            truncated_content += "\n... [truncated]"

        # 프롬프트 구성
        prompt = EXTRACTION_PROMPT.format(
            title=title,
            content=truncated_content,
        )

        try:
            # Claude API 호출
            response = self.client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )

            # 응답 파싱
            response_text = response.content[0].text
            metadata = self._parse_response(response_text)

            # 결과 캐싱
            if self._cache_enabled:
                self._cache[cache_key] = metadata

            logger.info(
                "AI 메타데이터 추출 완료",
                title=title[:50],
                content_type=metadata.content_type,
                topics_count=len(metadata.topics),
            )

            return metadata

        except Exception as e:
            logger.error("AI 추출 실패", error=str(e), title=title[:50])
            return ExtractedMetadata()

    def extract_batch(
        self,
        documents: list[tuple[str, str]],
        max_content_length: int = 8000,
    ) -> list[ExtractedMetadata]:
        """여러 문서에서 메타데이터를 추출합니다.

        현재는 순차 처리하지만, 향후 배치 API나
        비동기 처리로 최적화할 수 있습니다.

        Args:
            documents: (title, content) 튜플 리스트.
            max_content_length: 문서당 최대 내용 길이.

        Returns:
            ExtractedMetadata 객체 리스트.
        """
        results = []
        for title, content in documents:
            metadata = self.extract(title, content, max_content_length)
            results.append(metadata)
        return results

    def clear_cache(self) -> int:
        """추출 캐시를 비웁니다.

        메모리 해제나 강제 재추출이 필요할 때 사용합니다.

        Returns:
            삭제된 항목 수.
        """
        count = len(self._cache)
        self._cache.clear()
        return count


# 모듈 레벨 싱글톤 인스턴스
_ai_extractor: AIExtractor | None = None


def get_ai_extractor() -> AIExtractor | None:
    """AI 추출기 인스턴스를 가져옵니다 (활성화된 경우).

    settings에서 AI 추출 기능이 활성화되어 있고,
    API 키가 설정되어 있어야 인스턴스를 반환합니다.

    Returns:
        AI 추출이 활성화되면 AIExtractor 인스턴스,
        그렇지 않으면 None.

    Note:
        API 키가 설정되지 않은 경우 경고 로그가 출력됩니다.
    """
    global _ai_extractor

    # AI 추출 비활성화 확인
    if not settings.ai.enabled:
        return None

    # API 키 확인
    if not settings.ai.api_key:
        logger.warning("AI 추출이 활성화되었지만 ANTHROPIC_API_KEY가 설정되지 않음")
        return None

    # 싱글톤 인스턴스 생성
    if _ai_extractor is None:
        _ai_extractor = AIExtractor(
            api_key=settings.ai.api_key,
            model=settings.ai.model,
            max_tokens=settings.ai.max_tokens,
            timeout=settings.ai.timeout,
            cache_enabled=settings.ai.cache_enabled,
        )

    return _ai_extractor

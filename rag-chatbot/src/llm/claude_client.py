"""Claude Agent SDK 클라이언트 래퍼.

Claude Agent SDK를 사용하여 LLM 응답을 생성합니다.
"""

import time

import anyio
import structlog
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)

# 이미지 콘텐츠 타입 (claude_agent_sdk에 없는 경우 직접 정의)
try:
    from claude_agent_sdk import ImageBlock
except ImportError:
    # Claude Agent SDK에 ImageBlock이 없는 경우 직접 정의
    ImageBlock = None
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import get_settings
from ..models import ImageContent, Response, SearchResult, SourceReference
from .prompts import build_rag_prompt

logger = structlog.get_logger(__name__)


class ClaudeClient:
    """Claude Agent SDK 클라이언트.

    RAG 컨텍스트를 포함한 질문에 대해 Claude API를 통해 답변을 생성합니다.
    MCP 서버 연동을 통해 외부 시스템(Grafana, Sentry, AWS, Swagger, Jira,
    Notion, Slack) 조회가 가능합니다.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 400000,
        enable_mcp: bool = True,
    ):
        """ClaudeClient 초기화.

        Args:
            model: 사용할 Claude 모델
            max_tokens: 최대 응답 토큰 수
            enable_mcp: MCP 서버 활성화 여부
        """
        self.model = model
        self.max_tokens = max_tokens
        self.enable_mcp = enable_mcp
        self._settings = get_settings()
        self._mcp_servers = self._build_mcp_servers() if enable_mcp else {}
        self._allowed_tools = self._build_allowed_tools() if enable_mcp else []

    def _build_mcp_servers(self) -> dict:
        """MCP 서버 설정 빌드.

        환경변수가 설정된 MCP 서버만 활성화합니다.
        """
        servers = {}

        # Grafana MCP (Prometheus/Loki/Tempo 조회)
        if self._settings.grafana_url and self._settings.grafana_service_account_token:
            servers["grafana"] = {
                "type": "stdio",
                "command": "docker",
                "args": [
                    "run", "--rm", "-i", "--network", "host",
                    "-e", f"GRAFANA_URL={self._settings.grafana_url}",
                    "-e",
                    f"GRAFANA_SERVICE_ACCOUNT_TOKEN="
                    f"{self._settings.grafana_service_account_token}",
                    "mcp/grafana", "--disable-write", "-t", "stdio",
                ],
            }
            logger.info("Grafana MCP 서버 활성화", url=self._settings.grafana_url)

        # Sentry MCP (에러 모니터링)
        if self._settings.sentry_access_token:
            sentry_args = ["@sentry/mcp-server@latest"]
            sentry_env = {"SENTRY_ACCESS_TOKEN": self._settings.sentry_access_token}
            if self._settings.sentry_host:
                sentry_env["SENTRY_HOST"] = self._settings.sentry_host
            servers["sentry"] = {
                "type": "stdio",
                "command": "npx",
                "args": sentry_args,
                "env": sentry_env,
            }
            logger.info("Sentry MCP 서버 활성화", host=self._settings.sentry_host or "sentry.io")

        # AWS MCP (EC2/ECS/MSK 등 readonly 조회)
        if self._settings.aws_profile:
            servers["aws"] = {
                "type": "stdio",
                "command": "uvx",
                "args": [
                    "mcp-proxy-for-aws@latest",
                    "https://aws-mcp.us-east-1.api.aws/mcp",
                    "--region", self._settings.aws_region,
                    "--profile", self._settings.aws_profile,
                    "--read-only",
                ],
                "env": {
                    "AWS_PROFILE": self._settings.aws_profile,
                    "AWS_REGION": self._settings.aws_region,
                },
            }
            logger.info("AWS MCP 서버 활성화 (readonly)", region=self._settings.aws_region)

        # Swagger MCP (API 명세 조회)
        if self._settings.swagger_mcp_jar_path:
            servers["swagger"] = {
                "type": "stdio",
                "command": "java",
                "args": ["-jar", self._settings.swagger_mcp_jar_path],
            }
            logger.info("Swagger MCP 서버 활성화", jar=self._settings.swagger_mcp_jar_path)

        # Jira MCP (이슈/프로젝트 조회)
        if (
            self._settings.atlassian_site_name
            and self._settings.atlassian_user_email
            and self._settings.atlassian_api_token
        ):
            servers["jira"] = {
                "type": "stdio",
                "command": "npx",
                "args": ["-y", "@aashari/mcp-server-atlassian-jira"],
                "env": {
                    "ATLASSIAN_SITE_NAME": self._settings.atlassian_site_name,
                    "ATLASSIAN_USER_EMAIL": self._settings.atlassian_user_email,
                    "ATLASSIAN_API_TOKEN": self._settings.atlassian_api_token,
                },
            }
            logger.info(
                "Jira MCP 서버 활성화",
                site=f"{self._settings.atlassian_site_name}.atlassian.net",
            )

        # Notion MCP (페이지/데이터베이스 조회 - OAuth 인증)
        if self._settings.notion_mcp_enabled:
            servers["notion"] = {
                "type": "stdio",
                "command": "npx",
                "args": ["-y", "mcp-remote", "https://mcp.notion.com/mcp"],
            }
            logger.info("Notion MCP 서버 활성화 (OAuth 인증)")

        # Slack MCP (채널/메시지/사용자 조회)
        if self._settings.slack_mcp_enabled and self._settings.slack_team_id:
            servers["slack"] = {
                "type": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-slack"],
                "env": {
                    "SLACK_BOT_TOKEN": self._settings.slack_bot_token,
                    "SLACK_TEAM_ID": self._settings.slack_team_id,
                },
            }
            logger.info(
                "Slack MCP 서버 활성화",
                team_id=self._settings.slack_team_id,
            )

        return servers

    def _build_allowed_tools(self) -> list[str]:
        """허용된 MCP 도구 목록 빌드.

        등록된 MCP 서버의 모든 도구를 허용합니다.
        도구 이름 형식: mcp__{server_name}__{tool_name}
        """
        allowed = []
        for server_name in self._mcp_servers:
            # 각 MCP 서버의 모든 도구 허용 (와일드카드 패턴)
            allowed.append(f"mcp__{server_name}__*")
        return allowed

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def generate_response(
        self,
        question: str,
        search_results: list[SearchResult],
        conversation_context: str | None = None,
        images: list[ImageContent] | None = None,
    ) -> Response:
        """RAG 컨텍스트 기반 답변 생성.

        Args:
            question: 사용자 질문
            search_results: 벡터DB 검색 결과 목록
            conversation_context: 이전 대화 컨텍스트 (선택)
            images: 이미지 콘텐츠 목록 (선택, Claude Vision API용)

        Returns:
            Response 객체
        """
        # 빈 응답 시 재시도 (최대 2회)
        max_empty_retries = 2
        for attempt in range(max_empty_retries + 1):
            result = await self._call_claude_api(
                question=question,
                search_results=search_results,
                conversation_context=conversation_context,
                images=images,
                attempt=attempt,
            )

            # 응답이 비어있는지 검증
            if result.text and result.text.strip():
                return result

            # 빈 응답인 경우 재시도
            if attempt < max_empty_retries:
                logger.warning(
                    "Claude API가 빈 응답 반환, 재시도",
                    attempt=attempt + 1,
                    max_retries=max_empty_retries,
                )
            else:
                # 모든 재시도 실패 시 폴백 응답 반환
                logger.error(
                    "Claude API 빈 응답 재시도 모두 실패",
                    total_attempts=max_empty_retries + 1,
                )
                fallback_text = (
                    "죄송합니다. 답변을 생성하는 데 문제가 발생했습니다. "
                    "잠시 후 다시 시도해주세요."
                )
                return Response(
                    text=fallback_text,
                    sources=self._extract_sources(search_results),
                    model=self.model,
                    tokens_used=result.tokens_used,
                    generation_time_ms=result.generation_time_ms,
                )

        # 이 코드는 도달하지 않지만 타입 체커를 위해 추가
        return result

    async def _call_claude_api(
        self,
        question: str,
        search_results: list[SearchResult],
        conversation_context: str | None,
        images: list[ImageContent] | None = None,
        attempt: int = 0,
    ) -> Response:
        """Claude API 실제 호출 (내부 메서드).

        Args:
            question: 사용자 질문
            search_results: 벡터DB 검색 결과 목록
            conversation_context: 이전 대화 컨텍스트
            images: 이미지 콘텐츠 목록 (선택)
            attempt: 현재 시도 횟수

        Returns:
            Response 객체
        """
        start_time = time.perf_counter()

        # 프롬프트 생성
        prompt = build_rag_prompt(
            question=question,
            search_results=search_results,
            conversation_context=conversation_context,
        )

        # 이미지가 있는 경우 멀티모달 프롬프트 구성
        if images and ImageBlock is not None:
            prompt = self._build_multimodal_prompt(prompt, images)

        logger.info(
            "Claude API 호출 시작",
            question_length=len(question),
            context_count=len(search_results),
            has_conversation=conversation_context is not None,
            image_count=len(images) if images else 0,
            attempt=attempt,
        )

        try:
            # Claude Agent SDK 옵션 설정
            # MCP 서버가 활성화된 경우 도구 사용 허용
            # Slack 봇은 비대화형이므로 MCP 도구 권한을 자동 허용
            options = ClaudeAgentOptions(
                system_prompt=self._get_system_prompt(),
                max_turns=10 if self._mcp_servers else 1,
                mcp_servers=self._mcp_servers if self._mcp_servers else None,
                allowed_tools=self._allowed_tools if self._allowed_tools else [],
                permission_mode="bypassPermissions" if self._mcp_servers else "default",
            )

            # 응답 수집
            response_text = ""
            tokens_used = 0

            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            response_text += block.text
                elif isinstance(message, ResultMessage):
                    tokens_used = getattr(message, "input_tokens", 0) + getattr(
                        message, "output_tokens", 0
                    )

            # 응답 텍스트는 그대로 반환 (분할 전송은 핸들러에서 처리)

            generation_time_ms = int((time.perf_counter() - start_time) * 1000)

            # 소스 참조 생성
            sources = self._extract_sources(search_results)

            logger.info(
                "Claude API 호출 완료",
                response_length=len(response_text),
                tokens_used=tokens_used,
                generation_time_ms=generation_time_ms,
                attempt=attempt,
            )

            return Response(
                text=response_text,
                sources=sources,
                model=self.model,
                tokens_used=tokens_used,
                generation_time_ms=generation_time_ms,
            )

        except Exception as e:
            logger.error("Claude API 호출 실패", error=str(e), attempt=attempt)
            raise

    def _get_system_prompt(self) -> str:
        """시스템 프롬프트 반환."""
        base_prompt = """당신은 회사 내부 문서와 API 스펙 기반 질문에 답변하는 어시스턴트입니다.

## 핵심 원칙
1. 제공된 컨텍스트 정보만을 기반으로 정확하게 답변합니다.
2. 컨텍스트에 없는 정보는 추측하지 않고, 모른다고 솔직하게 말합니다.
3. 기술적 질문에는 구체적이고 실용적인 답변을 제공합니다.
4. 한국어로 자연스럽게 답변합니다.
5. 답변은 간결하면서도 필요한 정보를 빠뜨리지 않습니다.

## Slack 메시지 포매팅 규칙 (필수)

### 텍스트 스타일링
- *굵게*: 중요한 키워드, 핵심 용어 강조
- _기울임_: 변수명, 파라미터명, 부가 설명
- ~취소선~: 사용 금지 사항, 더 이상 유효하지 않은 정보
- `인라인 코드`: 함수명, 메서드명, 짧은 코드, 파일명, 경로

### 코드 블록
여러 줄 코드는 반드시 언어를 명시한 코드 블록 사용:
```python
def example():
    return "Hello"
```

### 구조화된 정보
- 목록: 불릿 포인트(•, -, *)로 항목 나열
- 번호 목록: 순서가 중요한 단계별 설명
- 들여쓰기: 하위 항목은 공백 2칸으로 들여쓰기

### 섹션 구분
- 빈 줄로 논리적 섹션 구분
- 이모지로 섹션 시작 (선택적): 📌 핵심, ⚠️ 주의, 💡 팁, ✅ 완료, ❌ 금지

### 링크
- <URL|표시 텍스트> 형식으로 링크 작성
- 예: <https://example.com|문서 바로가기>

### 인용
> 원문 인용이나 중요 메시지는 인용 블록 사용

## 응답 구조 템플릿

1. *핵심 답변* (1-2문장으로 질문에 직접 답변)

2. *상세 설명* (필요시)
   - 배경 정보
   - 구체적인 방법/절차
   - 코드 예시

3. *참고 사항* (선택적)
   - 주의점, 팁, 관련 정보

## 금지 사항
- 마크다운 헤더(#, ##) 사용 금지 (Slack에서 지원 안 함)
- HTML 태그 사용 금지
- 불필요하게 긴 답변 금지 (핵심만 전달)
- 같은 내용 반복 금지"""

        # MCP 서버가 활성화된 경우 도구 사용 안내 추가
        if self._mcp_servers:
            mcp_guidance = "\n\n## 사용 가능한 도구\n"
            mcp_guidance += "컨텍스트에 정보가 없을 경우, 다음 도구를 사용하여 정보를 조회하세요:\n"

            if "swagger" in self._mcp_servers:
                mcp_guidance += (
                    "- **Swagger MCP**: API 명세 조회 - 엔드포인트, 파라미터, "
                    "응답 형식 등 API 관련 질문에 활용\n"
                )
            if "grafana" in self._mcp_servers:
                mcp_guidance += (
                    "- **Grafana MCP**: 메트릭/로그 조회 - 서비스 상태, "
                    "에러율, 응답 시간 등 모니터링 데이터 조회\n"
                )
            if "sentry" in self._mcp_servers:
                mcp_guidance += (
                    "- **Sentry MCP**: 에러 모니터링 - 최근 에러, "
                    "이슈 상태, 스택트레이스 조회\n"
                )
            if "aws" in self._mcp_servers:
                mcp_guidance += (
                    "- **AWS MCP**: AWS 리소스 상태 조회 - EC2, ECS, "
                    "MSK 등 인프라 상태 확인 (읽기 전용)\n"
                )
            if "jira" in self._mcp_servers:
                mcp_guidance += (
                    "- **Jira MCP**: Jira 이슈/프로젝트 조회 - 티켓 상세, "
                    "프로젝트 현황, 이슈 검색 등 Jira 관련 질문에 활용\n"
                )
            if "notion" in self._mcp_servers:
                mcp_guidance += (
                    "- **Notion MCP**: Notion 페이지/데이터베이스 조회 - "
                    "문서 검색, 페이지 내용 확인 등 Notion 관련 질문에 활용\n"
                )
            if "slack" in self._mcp_servers:
                mcp_guidance += (
                    "- **Slack MCP**: Slack 채널/메시지/사용자 조회 - "
                    "채널 목록, 메시지 검색, 사용자 정보 등 Slack 관련 질문에 활용\n"
                )

            mcp_guidance += (
                "\n**중요**: 사용자 질문에 답변하기 위해 필요한 정보가 "
                "컨텍스트에 없다면, 위 도구를 적극적으로 활용하세요."
            )

            # MCP 체이닝 규칙 추가
            mcp_guidance += (
                "\n\n## 참조 자동 확장\n"
                "MCP 응답에 외부 링크가 포함되면, 해당 링크의 내용을 "
                "추가 조회하여 완전한 답변을 제공하세요."
            )

            base_prompt += mcp_guidance

        return base_prompt

    def _build_multimodal_prompt(
        self,
        text_prompt: str,
        images: list[ImageContent],
    ) -> list:
        """멀티모달 프롬프트 구성 (이미지 + 텍스트).

        Claude Vision API 형식으로 이미지와 텍스트를 조합합니다.

        Args:
            text_prompt: 기존 텍스트 프롬프트
            images: 이미지 콘텐츠 목록

        Returns:
            멀티모달 프롬프트 블록 리스트
        """
        prompt_blocks = []

        # 이미지 블록 추가
        for idx, img in enumerate(images):
            if ImageBlock is not None:
                prompt_blocks.append(
                    ImageBlock(
                        type="image",
                        source={
                            "type": "base64",
                            "media_type": img.media_type,
                            "data": img.data,
                        },
                    )
                )
                logger.debug(
                    "이미지 블록 추가",
                    index=idx,
                    media_type=img.media_type,
                    filename=img.filename,
                )

        # 이미지 설명 요청 텍스트 추가
        if images:
            image_intro = (
                f"\n\n[첨부된 이미지 {len(images)}개를 분석하여 "
                "질문에 답변해주세요.]\n\n"
            )
            prompt_blocks.append(TextBlock(text=image_intro + text_prompt))
        else:
            prompt_blocks.append(TextBlock(text=text_prompt))

        return prompt_blocks

    def _extract_sources(
        self, search_results: list[SearchResult]
    ) -> list[SourceReference]:
        """검색 결과에서 소스 참조 추출.

        중복 제거하여 고유한 소스만 반환합니다.

        Args:
            search_results: 검색 결과 목록

        Returns:
            SourceReference 목록
        """
        seen_ids: set[str] = set()
        sources: list[SourceReference] = []

        for result in search_results:
            if result.source_id not in seen_ids:
                seen_ids.add(result.source_id)
                sources.append(
                    SourceReference(
                        title=result.source_title,
                        url=result.source_url,
                        source_type=result.source_type,
                    )
                )

        return sources


# 동기 래퍼 함수 (Slack 핸들러에서 사용)
def generate_response_sync(
    client: ClaudeClient,
    question: str,
    search_results: list[SearchResult],
    conversation_context: str | None = None,
    images: list[ImageContent] | None = None,
) -> Response:
    """동기 방식으로 응답 생성.

    Args:
        client: ClaudeClient 인스턴스
        question: 사용자 질문
        search_results: 검색 결과 목록
        conversation_context: 대화 컨텍스트
        images: 이미지 콘텐츠 목록 (선택)

    Returns:
        Response 객체
    """
    return anyio.from_thread.run(
        client.generate_response,
        question,
        search_results,
        conversation_context,
        images,
    )

"""Notion API 커넥터.

Notion 워크스페이스에서 페이지와 데이터베이스를 가져옵니다.
Notion 블록을 인덱싱용 일반 텍스트로 변환합니다.
"""

import hashlib
from datetime import UTC, datetime
from typing import Any, Iterator, Optional

from notion_client import Client
from notion_client.errors import APIResponseError

from ..logging_config import Loggers
from ..models import Document, NotionSourceConfig, Source
from ..utils.rate_limit import TokenBucketRateLimiter, NOTION_RATE_LIMIT
from ..utils.retry import notion_retry, NOTION_CONFIG, RetryConfig

logger = Loggers.notion()


class NotionConnector:
    """Notion API 커넥터 - 콘텐츠 가져오기 및 변환.

    주요 기능:
    - 페이지 콘텐츠 추출
    - 데이터베이스 항목 열거
    - 블록을 텍스트로 변환
    - 콘텐츠 해싱을 통한 변경 감지
    """

    def __init__(self, api_key: str, rate_limit: bool = True):
        """Notion 커넥터 초기화.

        Args:
            api_key: Notion 통합 API 키.
            rate_limit: 레이트 리미팅 활성화 (기본값: True).
        """
        self.client = Client(auth=api_key)
        self._block_converters = self._init_block_converters()
        self._rate_limiter = TokenBucketRateLimiter(NOTION_RATE_LIMIT) if rate_limit else None

    def _rate_limit(self) -> None:
        """레이트 리미팅 적용 (활성화된 경우)."""
        if self._rate_limiter:
            self._rate_limiter.acquire()

    def _init_block_converters(self) -> dict:
        """블록 유형별 변환기 초기화.

        Returns:
            블록 유형을 변환 함수에 매핑하는 딕셔너리.
        """
        return {
            "paragraph": self._convert_paragraph,
            "heading_1": self._convert_heading,
            "heading_2": self._convert_heading,
            "heading_3": self._convert_heading,
            "bulleted_list_item": self._convert_list_item,
            "numbered_list_item": self._convert_list_item,
            "to_do": self._convert_todo,
            "toggle": self._convert_toggle,
            "code": self._convert_code,
            "quote": self._convert_quote,
            "callout": self._convert_callout,
            "divider": self._convert_divider,
            "table": self._convert_table,
            "table_row": self._convert_table_row,
            "bookmark": self._convert_bookmark,
            "embed": self._convert_embed,
            "image": self._convert_image,
            "video": self._convert_video,
            "file": self._convert_file,
            "pdf": self._convert_pdf,
            "equation": self._convert_equation,
            "synced_block": self._convert_synced_block,
            "column_list": self._convert_column_list,
            "column": self._convert_column,
            "child_page": self._convert_child_page,
            "child_database": self._convert_child_database,
            "link_preview": self._convert_link_preview,
            "template": self._convert_template,
            "breadcrumb": lambda b: "",  # 빈 문자열 반환
            "table_of_contents": lambda b: "",  # 빈 문자열 반환
            "unsupported": lambda b: "",  # 지원하지 않는 블록
        }

    # ==================== Public API ====================

    @notion_retry
    def fetch_page(self, page_id: str) -> dict:
        """단일 페이지의 메타데이터 가져오기.

        Args:
            page_id: Notion 페이지 ID.

        Returns:
            페이지 메타데이터 딕셔너리.

        Raises:
            APIResponseError: 재시도 후에도 API 호출이 실패한 경우.
        """
        self._rate_limit()
        return self.client.pages.retrieve(page_id=page_id)

    def fetch_page_content(self, page_id: str) -> str:
        """페이지 콘텐츠를 가져와서 텍스트로 변환.

        Args:
            page_id: Notion 페이지 ID.

        Returns:
            일반 텍스트로 변환된 페이지 콘텐츠.
        """
        blocks = self._get_all_blocks(page_id)
        return self._blocks_to_text(blocks)

    def fetch_database_items(
        self,
        database_id: str,
        page_size: int = 100,
    ) -> Iterator[dict]:
        """데이터베이스의 모든 항목 순회.

        Args:
            database_id: Notion 데이터베이스 ID.
            page_size: API 요청당 항목 수.

        Yields:
            데이터베이스 항목(페이지) 딕셔너리.

        Raises:
            APIResponseError: 재시도 후에도 API 호출이 실패한 경우.
        """
        start_cursor = None
        has_more = True

        while has_more:
            response = self._query_database_page(
                database_id=database_id,
                start_cursor=start_cursor,
                page_size=page_size,
            )
            for item in response.get("results", []):
                yield item

            has_more = response.get("has_more", False)
            start_cursor = response.get("next_cursor")

    @notion_retry
    def _query_database_page(
        self,
        database_id: str,
        start_cursor: Optional[str],
        page_size: int,
    ) -> dict:
        """데이터베이스 결과의 단일 페이지 쿼리 (재시도 포함).

        Args:
            database_id: Notion 데이터베이스 ID.
            start_cursor: 페이지네이션 커서.
            page_size: 요청당 항목 수.

        Returns:
            API 응답 딕셔너리.
        """
        self._rate_limit()
        return self.client.databases.query(
            database_id=database_id,
            start_cursor=start_cursor,
            page_size=page_size,
        )

    def fetch_documents(
        self,
        source: Source,
        existing_docs: Optional[list[Document]] = None,
    ) -> tuple[list[Document], list[Document], list[str]]:
        """Notion 소스에서 모든 문서 가져오기.

        설정된 페이지와 데이터베이스를 순회하며 문서를 추출합니다.
        기존 문서와 비교하여 새 문서, 업데이트된 문서, 삭제된 문서를 분류합니다.

        Args:
            source: Notion 소스 설정.
            existing_docs: 이전에 인덱싱된 문서 목록.

        Returns:
            (새 문서 목록, 업데이트된 문서 목록, 삭제된 문서 ID 목록) 튜플.
        """
        config: NotionSourceConfig = source.config
        existing_map = {d.external_id: d for d in (existing_docs or [])}
        seen_external_ids = set()

        new_docs = []
        updated_docs = []

        # 설정된 페이지 가져오기
        for page_id in config.page_ids:
            doc = self._process_page(source.id, page_id, existing_map)
            if doc:
                seen_external_ids.add(doc.external_id)
                if doc.external_id in existing_map:
                    if existing_map[doc.external_id].content_hash != doc.content_hash:
                        updated_docs.append(doc)
                else:
                    new_docs.append(doc)

        # 설정된 데이터베이스 가져오기
        for db_id in config.database_ids:
            for item in self.fetch_database_items(db_id):
                page_id = item["id"]
                doc = self._process_page(source.id, page_id, existing_map)
                if doc:
                    seen_external_ids.add(doc.external_id)
                    if doc.external_id in existing_map:
                        if existing_map[doc.external_id].content_hash != doc.content_hash:
                            updated_docs.append(doc)
                    else:
                        new_docs.append(doc)

        # 삭제된 문서 찾기
        deleted_ids = [
            d.id
            for ext_id, d in existing_map.items()
            if ext_id not in seen_external_ids
        ]

        logger.info(
            "Notion 문서 가져오기 완료",
            source_id=source.id,
            new=len(new_docs),
            updated=len(updated_docs),
            deleted=len(deleted_ids),
        )

        return new_docs, updated_docs, deleted_ids

    # ==================== Internal Methods ====================

    def _process_page(
        self,
        source_id: str,
        page_id: str,
        existing_map: dict[str, Document],
    ) -> Optional[Document]:
        """단일 페이지를 Document로 처리.

        Args:
            source_id: 부모 소스 ID.
            page_id: Notion 페이지 ID.
            existing_map: external_id를 기존 Document에 매핑하는 딕셔너리.

        Returns:
            Document 또는 처리 실패 시 None.
        """
        try:
            page = self.fetch_page(page_id)
            title = self._extract_title(page)
            content = self.fetch_page_content(page_id)
            url = page.get("url", "")
            content_hash = self._hash_content(content)

            # 기존 문서가 있는지 확인
            existing = existing_map.get(page_id)
            if existing:
                # 기존 문서 업데이트
                existing.title = title
                existing.content = content
                existing.url = url
                existing.content_hash = content_hash
                existing.updated_at = datetime.now(UTC)
                existing.metadata = self._extract_metadata(page)
                return existing

            # 새 문서 생성
            return Document(
                source_id=source_id,
                external_id=page_id,
                title=title,
                content=content,
                url=url,
                content_hash=content_hash,
                metadata=self._extract_metadata(page),
            )
        except Exception as e:
            logger.error("페이지 처리 실패", page_id=page_id, error=str(e))
            return None

    def _get_all_blocks(self, block_id: str) -> list[dict]:
        """부모 아래의 모든 블록을 재귀적으로 가져오기.

        Args:
            block_id: 부모 블록/페이지 ID.

        Returns:
            중첩된 자식을 포함한 모든 블록 목록.
        """
        blocks = []
        start_cursor = None
        has_more = True

        while has_more:
            response = self._list_block_children(
                block_id=block_id,
                start_cursor=start_cursor,
            )
            for block in response.get("results", []):
                blocks.append(block)
                # 자식 블록 재귀적으로 가져오기
                if block.get("has_children"):
                    children = self._get_all_blocks(block["id"])
                    blocks.extend(children)

            has_more = response.get("has_more", False)
            start_cursor = response.get("next_cursor")

        return blocks

    @notion_retry
    def _list_block_children(
        self,
        block_id: str,
        start_cursor: Optional[str],
    ) -> dict:
        """블록 자식 목록 조회 (재시도 포함).

        Args:
            block_id: 부모 블록 ID.
            start_cursor: 페이지네이션 커서.

        Returns:
            API 응답 딕셔너리.
        """
        self._rate_limit()
        return self.client.blocks.children.list(
            block_id=block_id,
            start_cursor=start_cursor,
            page_size=100,
        )

    def _blocks_to_text(self, blocks: list[dict]) -> str:
        """블록을 일반 텍스트로 변환.

        Args:
            blocks: Notion 블록 목록.

        Returns:
            일반 텍스트 콘텐츠.
        """
        lines = []
        for block in blocks:
            block_type = block.get("type", "unsupported")
            converter = self._block_converters.get(
                block_type,
                lambda b: "",
            )
            text = converter(block)
            if text:
                lines.append(text)
        return "\n\n".join(lines)

    def _extract_title(self, page: dict) -> str:
        """페이지 속성에서 제목 추출.

        Args:
            page: 페이지 메타데이터 딕셔너리.

        Returns:
            페이지 제목 문자열.
        """
        properties = page.get("properties", {})

        # 일반적인 제목 속성 이름 시도
        for prop_name in ["title", "Title", "Name", "name", "이름"]:
            prop = properties.get(prop_name, {})
            if prop.get("type") == "title":
                title_items = prop.get("title", [])
                if title_items:
                    return self._rich_text_to_str(title_items)

        # 폴백: 모든 title 유형 속성 찾기
        for prop in properties.values():
            if prop.get("type") == "title":
                title_items = prop.get("title", [])
                if title_items:
                    return self._rich_text_to_str(title_items)

        return "제목 없음"

    def _extract_metadata(self, page: dict) -> dict:
        """페이지에서 유용한 메타데이터 추출.

        Args:
            page: 페이지 메타데이터 딕셔너리.

        Returns:
            메타데이터 딕셔너리.
        """
        return {
            "created_time": page.get("created_time"),
            "last_edited_time": page.get("last_edited_time"),
            "created_by": page.get("created_by", {}).get("id"),
            "last_edited_by": page.get("last_edited_by", {}).get("id"),
            "archived": page.get("archived", False),
            "icon": self._extract_icon(page.get("icon")),
        }

    def _extract_icon(self, icon: Optional[dict]) -> Optional[str]:
        """아이콘 이모지 또는 URL 추출.

        Args:
            icon: 아이콘 딕셔너리.

        Returns:
            이모지 문자열 또는 URL, 없으면 None.
        """
        if not icon:
            return None
        if icon.get("type") == "emoji":
            return icon.get("emoji")
        if icon.get("type") == "external":
            return icon.get("external", {}).get("url")
        return None

    def _hash_content(self, content: str) -> str:
        """콘텐츠의 SHA256 해시 생성.

        Args:
            content: 텍스트 콘텐츠.

        Returns:
            해시의 16진수 다이제스트.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _rich_text_to_str(self, rich_text: list[dict]) -> str:
        """리치 텍스트 배열을 일반 문자열로 변환.

        Args:
            rich_text: Notion 리치 텍스트 배열.

        Returns:
            일반 텍스트 문자열.
        """
        return "".join(item.get("plain_text", "") for item in rich_text)

    # ==================== Block Converters ====================

    def _convert_paragraph(self, block: dict) -> str:
        """문단 블록 변환."""
        data = block.get("paragraph", {})
        return self._rich_text_to_str(data.get("rich_text", []))

    def _convert_heading(self, block: dict) -> str:
        """제목 블록 변환 (마크다운 스타일)."""
        block_type = block.get("type", "")
        data = block.get(block_type, {})
        text = self._rich_text_to_str(data.get("rich_text", []))

        # 마크다운 스타일 제목 마커 추가
        level = int(block_type.split("_")[-1])
        prefix = "#" * level
        return f"{prefix} {text}"

    def _convert_list_item(self, block: dict) -> str:
        """목록 항목 블록 변환."""
        block_type = block.get("type", "")
        data = block.get(block_type, {})
        text = self._rich_text_to_str(data.get("rich_text", []))
        return f"• {text}"

    def _convert_todo(self, block: dict) -> str:
        """할 일 블록 변환."""
        data = block.get("to_do", {})
        text = self._rich_text_to_str(data.get("rich_text", []))
        checked = "x" if data.get("checked") else " "
        return f"[{checked}] {text}"

    def _convert_toggle(self, block: dict) -> str:
        """토글 블록 변환."""
        data = block.get("toggle", {})
        return self._rich_text_to_str(data.get("rich_text", []))

    def _convert_code(self, block: dict) -> str:
        """코드 블록 변환."""
        data = block.get("code", {})
        language = data.get("language", "")
        code = self._rich_text_to_str(data.get("rich_text", []))
        return f"```{language}\n{code}\n```"

    def _convert_quote(self, block: dict) -> str:
        """인용 블록 변환."""
        data = block.get("quote", {})
        text = self._rich_text_to_str(data.get("rich_text", []))
        return f"> {text}"

    def _convert_callout(self, block: dict) -> str:
        """콜아웃 블록 변환."""
        data = block.get("callout", {})
        icon = self._extract_icon(data.get("icon"))
        text = self._rich_text_to_str(data.get("rich_text", []))
        if icon:
            return f"{icon} {text}"
        return text

    def _convert_divider(self, block: dict) -> str:
        """구분선 블록 변환."""
        return "---"

    def _convert_table(self, block: dict) -> str:
        """테이블 블록 변환 (헤더만).

        테이블 내용은 table_row 블록에서 처리됩니다.
        """
        return ""

    def _convert_table_row(self, block: dict) -> str:
        """테이블 행 블록 변환."""
        data = block.get("table_row", {})
        cells = data.get("cells", [])
        cell_texts = [self._rich_text_to_str(cell) for cell in cells]
        return " | ".join(cell_texts)

    def _convert_bookmark(self, block: dict) -> str:
        """북마크 블록 변환."""
        data = block.get("bookmark", {})
        url = data.get("url", "")
        caption = self._rich_text_to_str(data.get("caption", []))
        if caption:
            return f"[{caption}]({url})"
        return url

    def _convert_embed(self, block: dict) -> str:
        """임베드 블록 변환."""
        data = block.get("embed", {})
        return data.get("url", "")

    def _convert_image(self, block: dict) -> str:
        """이미지 블록 변환."""
        data = block.get("image", {})
        caption = self._rich_text_to_str(data.get("caption", []))
        if caption:
            return f"[이미지: {caption}]"
        return "[이미지]"

    def _convert_video(self, block: dict) -> str:
        """비디오 블록 변환."""
        data = block.get("video", {})
        caption = self._rich_text_to_str(data.get("caption", []))
        if caption:
            return f"[비디오: {caption}]"
        return "[비디오]"

    def _convert_file(self, block: dict) -> str:
        """파일 블록 변환."""
        data = block.get("file", {})
        caption = self._rich_text_to_str(data.get("caption", []))
        name = data.get("name", "파일")
        if caption:
            return f"[{name}: {caption}]"
        return f"[{name}]"

    def _convert_pdf(self, block: dict) -> str:
        """PDF 블록 변환."""
        data = block.get("pdf", {})
        caption = self._rich_text_to_str(data.get("caption", []))
        if caption:
            return f"[PDF: {caption}]"
        return "[PDF]"

    def _convert_equation(self, block: dict) -> str:
        """수식 블록 변환."""
        data = block.get("equation", {})
        return data.get("expression", "")

    def _convert_synced_block(self, block: dict) -> str:
        """동기화 블록 변환.

        내용은 자식에 있으며, 재귀로 처리됩니다.
        """
        return ""

    def _convert_column_list(self, block: dict) -> str:
        """컬럼 목록 블록 변환.

        내용은 자식에 있으며, 재귀로 처리됩니다.
        """
        return ""

    def _convert_column(self, block: dict) -> str:
        """컬럼 블록 변환.

        내용은 자식에 있으며, 재귀로 처리됩니다.
        """
        return ""

    def _convert_child_page(self, block: dict) -> str:
        """하위 페이지 참조 변환."""
        data = block.get("child_page", {})
        title = data.get("title", "제목 없음")
        return f"[하위 페이지: {title}]"

    def _convert_child_database(self, block: dict) -> str:
        """하위 데이터베이스 참조 변환."""
        data = block.get("child_database", {})
        title = data.get("title", "제목 없음")
        return f"[하위 데이터베이스: {title}]"

    def _convert_link_preview(self, block: dict) -> str:
        """링크 미리보기 블록 변환."""
        data = block.get("link_preview", {})
        return data.get("url", "")

    def _convert_template(self, block: dict) -> str:
        """템플릿 블록 변환."""
        data = block.get("template", {})
        return self._rich_text_to_str(data.get("rich_text", []))

"""Tests for Notion connector module."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from src.connectors.notion import NotionConnector
from src.models import Document, NotionSourceConfig, Source, SourceType


class TestNotionConnector:
    """Tests for NotionConnector class."""

    @pytest.fixture
    def mock_notion_client(self):
        """Create a mock Notion client."""
        with patch("src.connectors.notion.Client") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def connector(self, mock_notion_client):
        """Create a connector with mocked client."""
        return NotionConnector(api_key="test-api-key", rate_limit=False)

    @pytest.fixture
    def sample_page(self):
        """Create a sample Notion page response."""
        return {
            "id": "page-123",
            "url": "https://notion.so/page-123",
            "created_time": "2024-01-01T00:00:00.000Z",
            "last_edited_time": "2024-01-02T00:00:00.000Z",
            "created_by": {"id": "user-1"},
            "last_edited_by": {"id": "user-2"},
            "archived": False,
            "icon": {"type": "emoji", "emoji": "📄"},
            "properties": {
                "title": {
                    "type": "title",
                    "title": [{"plain_text": "Test Page Title"}],
                }
            },
        }

    # ==================== Initialization ====================

    def test_initialization(self, mock_notion_client):
        """Test connector initialization."""
        with patch("src.connectors.notion.Client") as mock_client:
            mock_client.return_value = mock_notion_client
            connector = NotionConnector(api_key="test-key", rate_limit=False)

            mock_client.assert_called_once_with(auth="test-key")
            assert connector._rate_limiter is None

    def test_initialization_with_rate_limit(self, mock_notion_client):
        """Test connector initialization with rate limiting enabled."""
        with patch("src.connectors.notion.Client") as mock_client:
            mock_client.return_value = mock_notion_client
            connector = NotionConnector(api_key="test-key", rate_limit=True)

            assert connector._rate_limiter is not None

    # ==================== Block Converters ====================

    def test_convert_paragraph(self, connector):
        """Test paragraph block conversion."""
        block = {
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"plain_text": "This is a paragraph."}]
            },
        }
        result = connector._convert_paragraph(block)
        assert result == "This is a paragraph."

    def test_convert_heading_1(self, connector):
        """Test heading 1 block conversion."""
        block = {
            "type": "heading_1",
            "heading_1": {
                "rich_text": [{"plain_text": "Main Heading"}]
            },
        }
        result = connector._convert_heading(block)
        assert result == "# Main Heading"

    def test_convert_heading_2(self, connector):
        """Test heading 2 block conversion."""
        block = {
            "type": "heading_2",
            "heading_2": {
                "rich_text": [{"plain_text": "Sub Heading"}]
            },
        }
        result = connector._convert_heading(block)
        assert result == "## Sub Heading"

    def test_convert_heading_3(self, connector):
        """Test heading 3 block conversion."""
        block = {
            "type": "heading_3",
            "heading_3": {
                "rich_text": [{"plain_text": "Minor Heading"}]
            },
        }
        result = connector._convert_heading(block)
        assert result == "### Minor Heading"

    def test_convert_bulleted_list_item(self, connector):
        """Test bulleted list item conversion."""
        block = {
            "type": "bulleted_list_item",
            "bulleted_list_item": {
                "rich_text": [{"plain_text": "List item"}]
            },
        }
        result = connector._convert_list_item(block)
        assert result == "• List item"

    def test_convert_numbered_list_item(self, connector):
        """Test numbered list item conversion."""
        block = {
            "type": "numbered_list_item",
            "numbered_list_item": {
                "rich_text": [{"plain_text": "Numbered item"}]
            },
        }
        result = connector._convert_list_item(block)
        assert result == "• Numbered item"

    def test_convert_todo_unchecked(self, connector):
        """Test unchecked todo block conversion."""
        block = {
            "type": "to_do",
            "to_do": {
                "rich_text": [{"plain_text": "Todo item"}],
                "checked": False,
            },
        }
        result = connector._convert_todo(block)
        assert result == "[ ] Todo item"

    def test_convert_todo_checked(self, connector):
        """Test checked todo block conversion."""
        block = {
            "type": "to_do",
            "to_do": {
                "rich_text": [{"plain_text": "Done item"}],
                "checked": True,
            },
        }
        result = connector._convert_todo(block)
        assert result == "[x] Done item"

    def test_convert_toggle(self, connector):
        """Test toggle block conversion."""
        block = {
            "type": "toggle",
            "toggle": {
                "rich_text": [{"plain_text": "Toggle content"}]
            },
        }
        result = connector._convert_toggle(block)
        assert result == "Toggle content"

    def test_convert_code(self, connector):
        """Test code block conversion."""
        block = {
            "type": "code",
            "code": {
                "language": "python",
                "rich_text": [{"plain_text": "print('hello')"}],
            },
        }
        result = connector._convert_code(block)
        assert result == "```python\nprint('hello')\n```"

    def test_convert_quote(self, connector):
        """Test quote block conversion."""
        block = {
            "type": "quote",
            "quote": {
                "rich_text": [{"plain_text": "A wise quote"}]
            },
        }
        result = connector._convert_quote(block)
        assert result == "> A wise quote"

    def test_convert_callout_with_icon(self, connector):
        """Test callout block conversion with icon."""
        block = {
            "type": "callout",
            "callout": {
                "icon": {"type": "emoji", "emoji": "💡"},
                "rich_text": [{"plain_text": "Important note"}],
            },
        }
        result = connector._convert_callout(block)
        assert result == "💡 Important note"

    def test_convert_callout_without_icon(self, connector):
        """Test callout block conversion without icon."""
        block = {
            "type": "callout",
            "callout": {
                "icon": None,
                "rich_text": [{"plain_text": "Note without icon"}],
            },
        }
        result = connector._convert_callout(block)
        assert result == "Note without icon"

    def test_convert_divider(self, connector):
        """Test divider block conversion."""
        block = {"type": "divider", "divider": {}}
        result = connector._convert_divider(block)
        assert result == "---"

    def test_convert_table_row(self, connector):
        """Test table row block conversion."""
        block = {
            "type": "table_row",
            "table_row": {
                "cells": [
                    [{"plain_text": "Cell 1"}],
                    [{"plain_text": "Cell 2"}],
                    [{"plain_text": "Cell 3"}],
                ]
            },
        }
        result = connector._convert_table_row(block)
        assert result == "Cell 1 | Cell 2 | Cell 3"

    def test_convert_bookmark_with_caption(self, connector):
        """Test bookmark block conversion with caption."""
        block = {
            "type": "bookmark",
            "bookmark": {
                "url": "https://example.com",
                "caption": [{"plain_text": "Example Site"}],
            },
        }
        result = connector._convert_bookmark(block)
        assert result == "[Example Site](https://example.com)"

    def test_convert_bookmark_without_caption(self, connector):
        """Test bookmark block conversion without caption."""
        block = {
            "type": "bookmark",
            "bookmark": {
                "url": "https://example.com",
                "caption": [],
            },
        }
        result = connector._convert_bookmark(block)
        assert result == "https://example.com"

    def test_convert_embed(self, connector):
        """Test embed block conversion."""
        block = {
            "type": "embed",
            "embed": {"url": "https://youtube.com/watch?v=123"},
        }
        result = connector._convert_embed(block)
        assert result == "https://youtube.com/watch?v=123"

    def test_convert_image_with_caption(self, connector):
        """Test image block conversion with caption."""
        block = {
            "type": "image",
            "image": {"caption": [{"plain_text": "My image"}]},
        }
        result = connector._convert_image(block)
        assert result == "[Image: My image]"

    def test_convert_image_without_caption(self, connector):
        """Test image block conversion without caption."""
        block = {
            "type": "image",
            "image": {"caption": []},
        }
        result = connector._convert_image(block)
        assert result == "[Image]"

    def test_convert_video_with_caption(self, connector):
        """Test video block conversion with caption."""
        block = {
            "type": "video",
            "video": {"caption": [{"plain_text": "Tutorial video"}]},
        }
        result = connector._convert_video(block)
        assert result == "[Video: Tutorial video]"

    def test_convert_video_without_caption(self, connector):
        """Test video block conversion without caption."""
        block = {
            "type": "video",
            "video": {"caption": []},
        }
        result = connector._convert_video(block)
        assert result == "[Video]"

    def test_convert_file_with_caption(self, connector):
        """Test file block conversion with caption."""
        block = {
            "type": "file",
            "file": {
                "name": "document.pdf",
                "caption": [{"plain_text": "Important document"}],
            },
        }
        result = connector._convert_file(block)
        assert result == "[document.pdf: Important document]"

    def test_convert_file_without_caption(self, connector):
        """Test file block conversion without caption."""
        block = {
            "type": "file",
            "file": {
                "name": "report.xlsx",
                "caption": [],
            },
        }
        result = connector._convert_file(block)
        assert result == "[report.xlsx]"

    def test_convert_pdf_with_caption(self, connector):
        """Test PDF block conversion with caption."""
        block = {
            "type": "pdf",
            "pdf": {"caption": [{"plain_text": "Research paper"}]},
        }
        result = connector._convert_pdf(block)
        assert result == "[PDF: Research paper]"

    def test_convert_pdf_without_caption(self, connector):
        """Test PDF block conversion without caption."""
        block = {
            "type": "pdf",
            "pdf": {"caption": []},
        }
        result = connector._convert_pdf(block)
        assert result == "[PDF]"

    def test_convert_equation(self, connector):
        """Test equation block conversion."""
        block = {
            "type": "equation",
            "equation": {"expression": "E = mc^2"},
        }
        result = connector._convert_equation(block)
        assert result == "E = mc^2"

    def test_convert_child_page(self, connector):
        """Test child page reference conversion."""
        block = {
            "type": "child_page",
            "child_page": {"title": "Sub Page"},
        }
        result = connector._convert_child_page(block)
        assert result == "[Child Page: Sub Page]"

    def test_convert_child_database(self, connector):
        """Test child database reference conversion."""
        block = {
            "type": "child_database",
            "child_database": {"title": "Task Database"},
        }
        result = connector._convert_child_database(block)
        assert result == "[Child Database: Task Database]"

    def test_convert_link_preview(self, connector):
        """Test link preview block conversion."""
        block = {
            "type": "link_preview",
            "link_preview": {"url": "https://github.com/user/repo"},
        }
        result = connector._convert_link_preview(block)
        assert result == "https://github.com/user/repo"

    def test_convert_template(self, connector):
        """Test template block conversion."""
        block = {
            "type": "template",
            "template": {"rich_text": [{"plain_text": "Template content"}]},
        }
        result = connector._convert_template(block)
        assert result == "Template content"

    def test_convert_synced_block_empty(self, connector):
        """Test synced block returns empty (children handled separately)."""
        block = {"type": "synced_block", "synced_block": {}}
        result = connector._convert_synced_block(block)
        assert result == ""

    def test_convert_column_list_empty(self, connector):
        """Test column list returns empty (children handled separately)."""
        block = {"type": "column_list", "column_list": {}}
        result = connector._convert_column_list(block)
        assert result == ""

    def test_convert_column_empty(self, connector):
        """Test column returns empty (children handled separately)."""
        block = {"type": "column", "column": {}}
        result = connector._convert_column(block)
        assert result == ""

    def test_convert_table_empty(self, connector):
        """Test table header returns empty (rows handled separately)."""
        block = {"type": "table", "table": {}}
        result = connector._convert_table(block)
        assert result == ""

    # ==================== Title Extraction ====================

    def test_extract_title_standard(self, connector, sample_page):
        """Test extracting title from standard title property."""
        result = connector._extract_title(sample_page)
        assert result == "Test Page Title"

    def test_extract_title_name_property(self, connector):
        """Test extracting title from Name property."""
        page = {
            "properties": {
                "Name": {
                    "type": "title",
                    "title": [{"plain_text": "Named Page"}],
                }
            }
        }
        result = connector._extract_title(page)
        assert result == "Named Page"

    def test_extract_title_korean_property(self, connector):
        """Test extracting title from Korean property name."""
        page = {
            "properties": {
                "이름": {
                    "type": "title",
                    "title": [{"plain_text": "한글 제목"}],
                }
            }
        }
        result = connector._extract_title(page)
        assert result == "한글 제목"

    def test_extract_title_fallback_any_title(self, connector):
        """Test extracting title from any title property as fallback."""
        page = {
            "properties": {
                "CustomTitle": {
                    "type": "title",
                    "title": [{"plain_text": "Custom Title"}],
                }
            }
        }
        result = connector._extract_title(page)
        assert result == "Custom Title"

    def test_extract_title_untitled(self, connector):
        """Test extracting title returns Untitled when none found."""
        page = {"properties": {}}
        result = connector._extract_title(page)
        assert result == "Untitled"

    def test_extract_title_empty_title_items(self, connector):
        """Test extracting title with empty title items."""
        page = {
            "properties": {
                "title": {
                    "type": "title",
                    "title": [],
                }
            }
        }
        result = connector._extract_title(page)
        assert result == "Untitled"

    # ==================== Metadata Extraction ====================

    def test_extract_metadata(self, connector, sample_page):
        """Test extracting metadata from page."""
        result = connector._extract_metadata(sample_page)

        assert result["created_time"] == "2024-01-01T00:00:00.000Z"
        assert result["last_edited_time"] == "2024-01-02T00:00:00.000Z"
        assert result["created_by"] == "user-1"
        assert result["last_edited_by"] == "user-2"
        assert result["archived"] is False
        assert result["icon"] == "📄"

    def test_extract_icon_emoji(self, connector):
        """Test extracting emoji icon."""
        icon = {"type": "emoji", "emoji": "🚀"}
        result = connector._extract_icon(icon)
        assert result == "🚀"

    def test_extract_icon_external(self, connector):
        """Test extracting external icon URL."""
        icon = {
            "type": "external",
            "external": {"url": "https://example.com/icon.png"},
        }
        result = connector._extract_icon(icon)
        assert result == "https://example.com/icon.png"

    def test_extract_icon_none(self, connector):
        """Test extracting icon when None."""
        result = connector._extract_icon(None)
        assert result is None

    def test_extract_icon_unknown_type(self, connector):
        """Test extracting icon with unknown type."""
        icon = {"type": "unknown"}
        result = connector._extract_icon(icon)
        assert result is None

    # ==================== Content Hashing ====================

    def test_hash_content(self, connector):
        """Test content hashing."""
        content = "Test content for hashing"
        hash1 = connector._hash_content(content)

        assert len(hash1) == 64  # SHA256 hex digest
        assert hash1 == connector._hash_content(content)  # Deterministic

    def test_hash_content_different_inputs(self, connector):
        """Test different content produces different hashes."""
        hash1 = connector._hash_content("content 1")
        hash2 = connector._hash_content("content 2")
        assert hash1 != hash2

    def test_hash_content_korean(self, connector):
        """Test hashing Korean content."""
        content = "한글 콘텐츠 테스트"
        hash_result = connector._hash_content(content)
        assert len(hash_result) == 64

    # ==================== Rich Text Conversion ====================

    def test_rich_text_to_str_single(self, connector):
        """Test converting single rich text item."""
        rich_text = [{"plain_text": "Hello"}]
        result = connector._rich_text_to_str(rich_text)
        assert result == "Hello"

    def test_rich_text_to_str_multiple(self, connector):
        """Test converting multiple rich text items."""
        rich_text = [
            {"plain_text": "Hello "},
            {"plain_text": "World"},
        ]
        result = connector._rich_text_to_str(rich_text)
        assert result == "Hello World"

    def test_rich_text_to_str_empty(self, connector):
        """Test converting empty rich text."""
        result = connector._rich_text_to_str([])
        assert result == ""

    def test_rich_text_to_str_missing_plain_text(self, connector):
        """Test converting rich text with missing plain_text."""
        rich_text = [{"other_key": "value"}]
        result = connector._rich_text_to_str(rich_text)
        assert result == ""

    # ==================== Blocks to Text ====================

    def test_blocks_to_text_multiple(self, connector):
        """Test converting multiple blocks to text."""
        blocks = [
            {"type": "heading_1", "heading_1": {"rich_text": [{"plain_text": "Title"}]}},
            {"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "Content"}]}},
            {"type": "divider", "divider": {}},
        ]
        result = connector._blocks_to_text(blocks)
        assert "# Title" in result
        assert "Content" in result
        assert "---" in result

    def test_blocks_to_text_empty_block(self, connector):
        """Test blocks to text skips empty results."""
        blocks = [
            {"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "Keep this"}]}},
            {"type": "table", "table": {}},  # Returns empty string
            {"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "Also keep"}]}},
        ]
        result = connector._blocks_to_text(blocks)
        lines = [line for line in result.split("\n\n") if line]
        assert len(lines) == 2

    def test_blocks_to_text_unknown_type(self, connector):
        """Test blocks to text handles unknown block type."""
        blocks = [
            {"type": "unknown_type", "unknown_type": {}},
            {"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "Known"}]}},
        ]
        result = connector._blocks_to_text(blocks)
        assert "Known" in result


class TestNotionConnectorAPI:
    """Tests for NotionConnector API methods."""

    @pytest.fixture
    def mock_notion_client(self):
        """Create a mock Notion client."""
        with patch("src.connectors.notion.Client") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def connector(self, mock_notion_client):
        """Create a connector with mocked client."""
        return NotionConnector(api_key="test-api-key", rate_limit=False)

    def test_fetch_page(self, connector, mock_notion_client):
        """Test fetching a page."""
        mock_notion_client.pages.retrieve.return_value = {"id": "page-123"}

        result = connector.fetch_page("page-123")

        assert result["id"] == "page-123"
        mock_notion_client.pages.retrieve.assert_called_once_with(page_id="page-123")

    def test_fetch_database_items_single_page(self, connector, mock_notion_client):
        """Test fetching database items (single page)."""
        mock_notion_client.databases.query.return_value = {
            "results": [{"id": "item-1"}, {"id": "item-2"}],
            "has_more": False,
            "next_cursor": None,
        }

        items = list(connector.fetch_database_items("db-123"))

        assert len(items) == 2
        assert items[0]["id"] == "item-1"

    def test_fetch_database_items_pagination(self, connector, mock_notion_client):
        """Test fetching database items with pagination."""
        mock_notion_client.databases.query.side_effect = [
            {
                "results": [{"id": "item-1"}],
                "has_more": True,
                "next_cursor": "cursor-1",
            },
            {
                "results": [{"id": "item-2"}],
                "has_more": False,
                "next_cursor": None,
            },
        ]

        items = list(connector.fetch_database_items("db-123"))

        assert len(items) == 2
        assert mock_notion_client.databases.query.call_count == 2

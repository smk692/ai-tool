"""Unit tests for AI metadata extraction service."""

import json
import pytest
from unittest.mock import MagicMock, patch

from src.services.ai_extractor import (
    AIExtractor,
    ExtractedMetadata,
    EXTRACTION_PROMPT,
    get_ai_extractor,
)


class TestExtractedMetadata:
    """Tests for ExtractedMetadata model."""

    def test_default_values(self):
        """Test default values for ExtractedMetadata."""
        metadata = ExtractedMetadata()

        assert metadata.content_type == "general"
        assert metadata.topics == []
        assert metadata.difficulty == "intermediate"
        assert metadata.has_code_samples is False
        assert metadata.key_entities == []
        assert metadata.summary == ""

    def test_custom_values(self):
        """Test custom values for ExtractedMetadata."""
        metadata = ExtractedMetadata(
            content_type="api_doc",
            topics=["authentication", "security"],
            difficulty="advanced",
            has_code_samples=True,
            key_entities=["User", "Token"],
            summary="API documentation for user authentication.",
        )

        assert metadata.content_type == "api_doc"
        assert metadata.topics == ["authentication", "security"]
        assert metadata.difficulty == "advanced"
        assert metadata.has_code_samples is True
        assert metadata.key_entities == ["User", "Token"]
        assert metadata.summary == "API documentation for user authentication."

    def test_serialization(self):
        """Test model serialization."""
        metadata = ExtractedMetadata(
            content_type="guide",
            topics=["getting-started"],
            difficulty="beginner",
        )

        data = metadata.model_dump()
        assert data["content_type"] == "guide"
        assert data["topics"] == ["getting-started"]
        assert data["difficulty"] == "beginner"


class TestAIExtractor:
    """Tests for AIExtractor service."""

    @pytest.fixture
    def extractor(self):
        """Create AIExtractor instance for testing."""
        return AIExtractor(
            api_key="test-api-key",
            model="claude-3-haiku-20240307",
            cache_enabled=True,
        )

    def test_init_with_defaults(self):
        """Test AIExtractor initialization with defaults."""
        extractor = AIExtractor(api_key="test-key")

        assert extractor._api_key == "test-key"
        assert extractor._cache_enabled is True
        assert extractor._client is None

    def test_init_with_custom_values(self):
        """Test AIExtractor initialization with custom values."""
        extractor = AIExtractor(
            api_key="custom-key",
            model="claude-3-opus-20240229",
            max_tokens=2048,
            timeout=60.0,
            cache_enabled=False,
        )

        assert extractor._api_key == "custom-key"
        assert extractor._model == "claude-3-opus-20240229"
        assert extractor._max_tokens == 2048
        assert extractor._timeout == 60.0
        assert extractor._cache_enabled is False

    def test_get_cache_key(self, extractor):
        """Test cache key generation."""
        key1 = extractor._get_cache_key("Title", "Content")
        key2 = extractor._get_cache_key("Title", "Content")
        key3 = extractor._get_cache_key("Different", "Content")

        assert key1 == key2  # Same input = same key
        assert key1 != key3  # Different input = different key
        assert len(key1) == 16  # Fixed length

    def test_parse_response_valid_json(self, extractor):
        """Test parsing valid JSON response."""
        response = json.dumps({
            "content_type": "tutorial",
            "topics": ["python", "testing"],
            "difficulty": "beginner",
            "has_code_samples": True,
            "key_entities": ["pytest", "unittest"],
            "summary": "A tutorial on Python testing.",
        })

        metadata = extractor._parse_response(response)

        assert metadata.content_type == "tutorial"
        assert metadata.topics == ["python", "testing"]
        assert metadata.difficulty == "beginner"
        assert metadata.has_code_samples is True
        assert metadata.key_entities == ["pytest", "unittest"]
        assert metadata.summary == "A tutorial on Python testing."

    def test_parse_response_json_in_code_block(self, extractor):
        """Test parsing JSON wrapped in markdown code block."""
        response = """```json
{
    "content_type": "api_doc",
    "topics": ["rest", "api"],
    "difficulty": "intermediate",
    "has_code_samples": false,
    "key_entities": ["GET", "POST"],
    "summary": "REST API documentation."
}
```"""

        metadata = extractor._parse_response(response)

        assert metadata.content_type == "api_doc"
        assert metadata.topics == ["rest", "api"]

    def test_parse_response_truncates_long_lists(self, extractor):
        """Test that long lists are truncated."""
        response = json.dumps({
            "content_type": "general",
            "topics": [f"topic{i}" for i in range(10)],  # More than 5
            "difficulty": "intermediate",
            "has_code_samples": False,
            "key_entities": [f"entity{i}" for i in range(15)],  # More than 10
            "summary": "x" * 1000,  # More than 500 chars
        })

        metadata = extractor._parse_response(response)

        assert len(metadata.topics) == 5  # Truncated to 5
        assert len(metadata.key_entities) == 10  # Truncated to 10
        assert len(metadata.summary) == 500  # Truncated to 500

    def test_parse_response_invalid_json(self, extractor):
        """Test parsing invalid JSON returns default metadata."""
        response = "This is not valid JSON"

        metadata = extractor._parse_response(response)

        assert metadata.content_type == "general"
        assert metadata.topics == []
        assert metadata.difficulty == "intermediate"

    def test_parse_response_missing_fields(self, extractor):
        """Test parsing JSON with missing fields uses defaults."""
        response = json.dumps({
            "content_type": "faq",
        })

        metadata = extractor._parse_response(response)

        assert metadata.content_type == "faq"
        assert metadata.topics == []  # Default
        assert metadata.difficulty == "intermediate"  # Default

    def test_extract_calls_api(self, extractor):
        """Test extract method calls Claude API."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "content_type": "guide",
            "topics": ["setup"],
            "difficulty": "beginner",
            "has_code_samples": True,
            "key_entities": ["Config"],
            "summary": "Setup guide.",
        }))]
        extractor._client = MagicMock()
        extractor._client.messages.create.return_value = mock_response

        result = extractor.extract(
            title="Setup Guide",
            content="This is a setup guide for the application.",
        )

        assert result.content_type == "guide"
        assert result.topics == ["setup"]
        extractor._client.messages.create.assert_called_once()

    def test_extract_uses_cache(self, extractor):
        """Test extract method uses cache on second call."""
        # Pre-populate cache
        cache_key = extractor._get_cache_key("Title", "Content")
        cached_metadata = ExtractedMetadata(content_type="cached")
        extractor._cache[cache_key] = cached_metadata

        result = extractor.extract(title="Title", content="Content")

        assert result.content_type == "cached"

    def test_extract_truncates_long_content(self, extractor):
        """Test extract truncates content longer than max_content_length."""
        extractor._client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"content_type": "general"}')]
        extractor._client.messages.create.return_value = mock_response

        long_content = "x" * 10000  # Longer than default 8000
        extractor.extract(title="Test", content=long_content, max_content_length=100)

        # Check that truncated content was passed
        call_args = extractor._client.messages.create.call_args
        prompt = call_args.kwargs["messages"][0]["content"]
        assert "[truncated]" in prompt

    def test_extract_handles_api_error(self, extractor):
        """Test extract returns default metadata on API error."""
        extractor._client = MagicMock()
        extractor._client.messages.create.side_effect = Exception("API Error")

        result = extractor.extract(title="Test", content="Content")

        assert result.content_type == "general"
        assert result.topics == []

    def test_extract_batch(self, extractor):
        """Test batch extraction."""
        # Pre-populate cache for all documents
        docs = [("Title1", "Content1"), ("Title2", "Content2")]
        for title, content in docs:
            cache_key = extractor._get_cache_key(title, content)
            extractor._cache[cache_key] = ExtractedMetadata(
                content_type=f"type_{title.lower()}"
            )

        results = extractor.extract_batch(docs)

        assert len(results) == 2
        assert results[0].content_type == "type_title1"
        assert results[1].content_type == "type_title2"

    def test_clear_cache(self, extractor):
        """Test cache clearing."""
        extractor._cache["key1"] = ExtractedMetadata()
        extractor._cache["key2"] = ExtractedMetadata()

        count = extractor.clear_cache()

        assert count == 2
        assert len(extractor._cache) == 0


class TestGetAIExtractor:
    """Tests for get_ai_extractor factory function."""

    @patch("src.services.ai_extractor.settings")
    def test_returns_none_when_disabled(self, mock_settings):
        """Test returns None when AI extraction is disabled."""
        mock_settings.ai.enabled = False

        # Reset module-level instance
        import src.services.ai_extractor as module
        module._ai_extractor = None

        result = get_ai_extractor()

        assert result is None

    @patch("src.services.ai_extractor.settings")
    def test_returns_none_when_no_api_key(self, mock_settings):
        """Test returns None when no API key is set."""
        mock_settings.ai.enabled = True
        mock_settings.ai.api_key = ""

        # Reset module-level instance
        import src.services.ai_extractor as module
        module._ai_extractor = None

        result = get_ai_extractor()

        assert result is None

    @patch("src.services.ai_extractor.settings")
    def test_creates_instance_when_enabled(self, mock_settings):
        """Test creates AIExtractor instance when enabled and configured."""
        mock_settings.ai.enabled = True
        mock_settings.ai.api_key = "test-api-key"
        mock_settings.ai.model = "claude-3-haiku-20240307"
        mock_settings.ai.max_tokens = 1024
        mock_settings.ai.timeout = 30.0
        mock_settings.ai.cache_enabled = True

        # Reset module-level instance
        import src.services.ai_extractor as module
        module._ai_extractor = None

        result = get_ai_extractor()

        assert result is not None
        assert isinstance(result, AIExtractor)

    @patch("src.services.ai_extractor.settings")
    def test_returns_cached_instance(self, mock_settings):
        """Test returns cached instance on subsequent calls."""
        mock_settings.ai.enabled = True
        mock_settings.ai.api_key = "test-api-key"
        mock_settings.ai.model = "claude-3-haiku-20240307"
        mock_settings.ai.max_tokens = 1024
        mock_settings.ai.timeout = 30.0
        mock_settings.ai.cache_enabled = True

        # Reset module-level instance
        import src.services.ai_extractor as module
        module._ai_extractor = None

        result1 = get_ai_extractor()
        result2 = get_ai_extractor()

        assert result1 is result2


class TestExtractionPrompt:
    """Tests for extraction prompt template."""

    def test_prompt_contains_required_fields(self):
        """Test that prompt contains all required field names."""
        assert "content_type" in EXTRACTION_PROMPT
        assert "topics" in EXTRACTION_PROMPT
        assert "difficulty" in EXTRACTION_PROMPT
        assert "has_code_samples" in EXTRACTION_PROMPT
        assert "key_entities" in EXTRACTION_PROMPT
        assert "summary" in EXTRACTION_PROMPT

    def test_prompt_formatting(self):
        """Test prompt can be formatted with title and content."""
        formatted = EXTRACTION_PROMPT.format(
            title="Test Title",
            content="Test content here.",
        )

        assert "Test Title" in formatted
        assert "Test content here." in formatted

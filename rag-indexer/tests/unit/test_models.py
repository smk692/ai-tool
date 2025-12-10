"""Unit tests for data models."""

import pytest
from datetime import UTC, datetime
from uuid import UUID

from src.models import Document, Chunk, Source, NotionSourceConfig, SwaggerSourceConfig


class TestDocument:
    """Tests for Document model."""

    def test_document_creation_minimal(self):
        """Test document creation with minimal required fields."""
        doc = Document(
            source_id="source-1",
            external_id="ext-1",
            title="Test",
        )
        assert doc.source_id == "source-1"
        assert doc.external_id == "ext-1"
        assert doc.title == "Test"
        assert doc.content == ""
        assert doc.url is None
        assert isinstance(doc.id, str)
        assert UUID(doc.id)  # Valid UUID

    def test_document_creation_full(self, sample_document):
        """Test document creation with all fields."""
        assert sample_document.id == "doc-123"
        assert sample_document.source_id == "source-456"
        assert sample_document.external_id == "notion-page-789"
        assert sample_document.title == "Test Document"
        assert "test document content" in sample_document.content
        assert sample_document.url == "https://notion.so/page/789"
        assert sample_document.content_hash == "abc123hash"
        assert sample_document.metadata == {"created_by": "user-1"}

    def test_document_timestamps(self):
        """Test document timestamps are set correctly."""
        before = datetime.now(UTC)
        doc = Document(source_id="s", external_id="e", title="t")
        after = datetime.now(UTC)

        assert before <= doc.created_at <= after
        assert before <= doc.updated_at <= after
        assert doc.indexed_at is None

    def test_mark_indexed(self, sample_document):
        """Test mark_indexed updates timestamps."""
        original_updated = sample_document.updated_at
        assert sample_document.indexed_at is None

        sample_document.mark_indexed()

        assert sample_document.indexed_at is not None
        assert sample_document.updated_at >= original_updated

    def test_needs_reindex_changed(self, sample_document):
        """Test needs_reindex returns True for different hash."""
        assert sample_document.needs_reindex("different_hash") is True

    def test_needs_reindex_unchanged(self, sample_document):
        """Test needs_reindex returns False for same hash."""
        assert sample_document.needs_reindex("abc123hash") is False

    def test_model_dump_json_safe(self, sample_document):
        """Test JSON-safe serialization."""
        data = sample_document.model_dump_json_safe()

        assert isinstance(data["created_at"], str)
        assert isinstance(data["updated_at"], str)
        assert data["indexed_at"] is None

        # Mark indexed and check again
        sample_document.mark_indexed()
        data = sample_document.model_dump_json_safe()
        assert isinstance(data["indexed_at"], str)

    def test_from_json_safe(self, sample_document):
        """Test deserialization from JSON-safe dict."""
        sample_document.mark_indexed()
        data = sample_document.model_dump_json_safe()

        restored = Document.from_json_safe(data)

        assert restored.id == sample_document.id
        assert restored.title == sample_document.title
        assert isinstance(restored.created_at, datetime)
        assert isinstance(restored.indexed_at, datetime)


class TestChunk:
    """Tests for Chunk model."""

    def test_chunk_creation_minimal(self):
        """Test chunk creation with minimal required fields."""
        chunk = Chunk(
            document_id="doc-1",
            chunk_index=0,
            text="Sample text",
        )
        assert chunk.document_id == "doc-1"
        assert chunk.chunk_index == 0
        assert chunk.text == "Sample text"
        assert chunk.embedding is None
        assert chunk.token_count == 0

    def test_chunk_creation_with_embedding(self, chunk_with_embedding):
        """Test chunk creation with embedding."""
        assert len(chunk_with_embedding.embedding) == 1024
        assert all(v == 0.1 for v in chunk_with_embedding.embedding)

    def test_estimate_tokens(self, sample_chunk):
        """Test token estimation."""
        sample_chunk.text = "a" * 30  # 30 characters
        tokens = sample_chunk.estimate_tokens()

        assert tokens == 10  # 30 // 3
        assert sample_chunk.token_count == 10

    def test_estimate_tokens_korean(self):
        """Test token estimation with Korean text."""
        chunk = Chunk(
            document_id="doc-1",
            chunk_index=0,
            text="한글" * 10,  # 20 Korean characters
        )
        tokens = chunk.estimate_tokens()
        assert tokens == 6  # 20 // 3

    def test_to_qdrant_point(self, chunk_with_embedding):
        """Test conversion to Qdrant point format."""
        point = chunk_with_embedding.to_qdrant_point(
            source_id="source-1",
            source_type="notion",
            title="Test Doc",
            url="https://example.com",
        )

        assert point["id"] == chunk_with_embedding.id
        assert len(point["vector"]) == 1024
        assert point["payload"]["chunk_id"] == chunk_with_embedding.id
        assert point["payload"]["document_id"] == chunk_with_embedding.document_id
        assert point["payload"]["source_id"] == "source-1"
        assert point["payload"]["source_type"] == "notion"
        assert point["payload"]["title"] == "Test Doc"
        assert point["payload"]["url"] == "https://example.com"
        assert point["payload"]["chunk_index"] == 0

    def test_to_qdrant_point_no_embedding(self, sample_chunk):
        """Test to_qdrant_point raises error without embedding."""
        with pytest.raises(ValueError, match="no embedding"):
            sample_chunk.to_qdrant_point(
                source_id="s",
                source_type="notion",
                title="t",
            )

    def test_model_dump_json_safe(self, chunk_with_embedding):
        """Test JSON-safe serialization excludes embedding."""
        data = chunk_with_embedding.model_dump_json_safe()

        assert "embedding" not in data
        assert isinstance(data["created_at"], str)
        assert data["text"] == chunk_with_embedding.text


class TestDetectLanguage:
    """Tests for detect_language function."""

    def test_detect_korean_text(self):
        """Test detection of Korean text."""
        from src.models.chunk import detect_language

        korean_text = "안녕하세요, 이것은 한국어 테스트입니다."
        assert detect_language(korean_text) == "ko"

    def test_detect_english_text(self):
        """Test detection of English text."""
        from src.models.chunk import detect_language

        english_text = "Hello, this is an English test."
        assert detect_language(english_text) == "en"

    def test_detect_mixed_text_korean_dominant(self):
        """Test detection of mixed text with Korean dominant."""
        from src.models.chunk import detect_language

        # More than 30% Korean characters
        mixed_text = "안녕하세요 Hello 한국어가 많습니다"
        assert detect_language(mixed_text) == "ko"

    def test_detect_mixed_text_english_dominant(self):
        """Test detection of mixed text with English dominant."""
        from src.models.chunk import detect_language

        # Less than 30% Korean characters
        mixed_text = "Hello World This is mostly English 안녕"
        assert detect_language(mixed_text) == "en"

    def test_detect_empty_text(self):
        """Test detection of empty text returns English default."""
        from src.models.chunk import detect_language

        assert detect_language("") == "en"
        assert detect_language(None) == "en" if detect_language(None) else True

    def test_detect_numbers_only(self):
        """Test detection of text with only numbers returns English default."""
        from src.models.chunk import detect_language

        assert detect_language("12345 67890") == "en"

    def test_detect_special_characters(self):
        """Test detection handles special characters."""
        from src.models.chunk import detect_language

        assert detect_language("!@#$%^&*()") == "en"


class TestChunkEnhancedMetadata:
    """Tests for enhanced metadata in Chunk model."""

    def test_to_qdrant_point_enhanced_fields(self, chunk_with_embedding):
        """Test that enhanced metadata fields are included in Qdrant point."""
        chunk_with_embedding.token_count = 50
        chunk_with_embedding.metadata = {"total_chunks": 5}

        point = chunk_with_embedding.to_qdrant_point(
            source_id="source-1",
            source_type="notion",
            title="Test Doc",
            url="https://example.com",
        )

        # Check enhanced fields
        payload = point["payload"]
        assert "language" in payload
        assert payload["language"] in ["ko", "en"]
        assert payload["token_count"] == 50
        assert payload["total_chunks"] == 5
        assert "created_at" in payload

    def test_to_qdrant_point_with_notion_metadata(self, chunk_with_embedding):
        """Test that Notion-specific metadata is included."""
        document_metadata = {
            "notion_page_id": "page-123",
            "notion_parent_id": "parent-456",
            "notion_database_id": "db-789",
            "notion_last_edited_time": "2025-01-01T00:00:00Z",
        }

        point = chunk_with_embedding.to_qdrant_point(
            source_id="source-1",
            source_type="notion",
            title="Test Doc",
            url="https://example.com",
            document_metadata=document_metadata,
        )

        payload = point["payload"]
        assert payload["notion_page_id"] == "page-123"
        assert payload["notion_parent_id"] == "parent-456"
        assert payload["notion_database_id"] == "db-789"
        assert payload["notion_last_edited_time"] == "2025-01-01T00:00:00Z"

    def test_to_qdrant_point_with_swagger_metadata(self, chunk_with_embedding):
        """Test that Swagger-specific metadata is included."""
        document_metadata = {
            "api_endpoint": "/users/{id}",
            "http_method": "GET",
            "api_version": "v2",
            "operation_id": "getUserById",
            "tags": ["users", "public"],
        }

        point = chunk_with_embedding.to_qdrant_point(
            source_id="source-1",
            source_type="swagger",
            title="GET /users/{id}",
            url="https://api.example.com",
            document_metadata=document_metadata,
        )

        payload = point["payload"]
        assert payload["api_endpoint"] == "/users/{id}"
        assert payload["http_method"] == "GET"
        assert payload["api_version"] == "v2"
        assert payload["operation_id"] == "getUserById"
        assert payload["tags"] == ["users", "public"]

    def test_to_qdrant_point_with_generic_metadata(self, chunk_with_embedding):
        """Test that generic metadata fields are included."""
        document_metadata = {
            "content_type": "api_doc",
            "author": "test-user",
            "category": "authentication",
        }

        point = chunk_with_embedding.to_qdrant_point(
            source_id="source-1",
            source_type="notion",
            title="Test Doc",
            document_metadata=document_metadata,
        )

        payload = point["payload"]
        assert payload["content_type"] == "api_doc"
        assert payload["author"] == "test-user"
        assert payload["category"] == "authentication"

    def test_to_qdrant_point_language_detection_korean(self):
        """Test language detection for Korean content."""
        chunk = Chunk(
            document_id="doc-1",
            chunk_index=0,
            text="이것은 한국어로 작성된 테스트 문서입니다.",
            embedding=[0.1] * 1024,
        )

        point = chunk.to_qdrant_point(
            source_id="source-1",
            source_type="notion",
            title="Korean Doc",
        )

        assert point["payload"]["language"] == "ko"

    def test_to_qdrant_point_language_detection_english(self):
        """Test language detection for English content."""
        chunk = Chunk(
            document_id="doc-1",
            chunk_index=0,
            text="This is a test document written in English.",
            embedding=[0.1] * 1024,
        )

        point = chunk.to_qdrant_point(
            source_id="source-1",
            source_type="notion",
            title="English Doc",
        )

        assert point["payload"]["language"] == "en"

    def test_to_qdrant_point_with_ai_metadata(self, chunk_with_embedding):
        """Test that AI-extracted metadata fields are included."""
        document_metadata = {
            "ai_content_type": "api_doc",
            "ai_topics": ["authentication", "security"],
            "ai_difficulty": "advanced",
            "ai_has_code_samples": True,
            "ai_key_entities": ["User", "Token", "OAuth"],
            "ai_summary": "Documentation for user authentication API.",
        }

        point = chunk_with_embedding.to_qdrant_point(
            source_id="source-1",
            source_type="notion",
            title="Auth API Doc",
            url="https://docs.example.com/auth",
            document_metadata=document_metadata,
        )

        payload = point["payload"]
        assert payload["ai_content_type"] == "api_doc"
        assert payload["ai_topics"] == ["authentication", "security"]
        assert payload["ai_difficulty"] == "advanced"
        assert payload["ai_has_code_samples"] is True
        assert payload["ai_key_entities"] == ["User", "Token", "OAuth"]
        assert payload["ai_summary"] == "Documentation for user authentication API."

    def test_to_qdrant_point_partial_ai_metadata(self, chunk_with_embedding):
        """Test that partial AI metadata is handled correctly."""
        document_metadata = {
            "ai_content_type": "guide",
            "ai_topics": ["setup"],
            # Other AI fields missing - should not cause errors
        }

        point = chunk_with_embedding.to_qdrant_point(
            source_id="source-1",
            source_type="notion",
            title="Setup Guide",
            document_metadata=document_metadata,
        )

        payload = point["payload"]
        assert payload["ai_content_type"] == "guide"
        assert payload["ai_topics"] == ["setup"]
        # Missing fields should not be present
        assert "ai_difficulty" not in payload
        assert "ai_summary" not in payload

    def test_to_qdrant_point_combined_metadata(self, chunk_with_embedding):
        """Test combining Notion, generic, and AI metadata."""
        document_metadata = {
            # Notion metadata
            "notion_page_id": "page-123",
            "notion_database_id": "db-456",
            # Generic metadata
            "content_type": "tutorial",
            "author": "dev-team",
            # AI metadata
            "ai_content_type": "tutorial",
            "ai_topics": ["getting-started"],
            "ai_difficulty": "beginner",
            "ai_has_code_samples": True,
            "ai_key_entities": ["Config", "Setup"],
            "ai_summary": "Getting started tutorial.",
        }

        point = chunk_with_embedding.to_qdrant_point(
            source_id="source-1",
            source_type="notion",
            title="Getting Started",
            url="https://docs.example.com/start",
            document_metadata=document_metadata,
        )

        payload = point["payload"]
        # Check all metadata types are present
        assert payload["notion_page_id"] == "page-123"
        assert payload["notion_database_id"] == "db-456"
        assert payload["content_type"] == "tutorial"
        assert payload["author"] == "dev-team"
        assert payload["ai_content_type"] == "tutorial"
        assert payload["ai_topics"] == ["getting-started"]
        assert payload["ai_difficulty"] == "beginner"
        assert payload["ai_has_code_samples"] is True


class TestSource:
    """Tests for Source model."""

    def test_notion_source_creation(self, notion_source):
        """Test Notion source creation."""
        assert notion_source.id == "notion-source-1"
        assert notion_source.name == "Test Notion Workspace"
        assert notion_source.source_type == "notion"
        assert notion_source.enabled is True
        assert isinstance(notion_source.config, NotionSourceConfig)
        assert notion_source.config.page_ids == ["page-1", "page-2"]
        assert notion_source.config.database_ids == ["db-1"]

    def test_swagger_source_creation(self, swagger_source):
        """Test Swagger source creation."""
        assert swagger_source.id == "swagger-source-1"
        assert swagger_source.name == "Test API Docs"
        assert swagger_source.source_type == "swagger"
        assert isinstance(swagger_source.config, SwaggerSourceConfig)
        assert swagger_source.config.url == "https://api.example.com/swagger.json"

    def test_notion_config_validation(self):
        """Test NotionSourceConfig validation."""
        config = NotionSourceConfig(
            page_ids=["p1"],
            database_ids=[],
        )
        assert config.page_ids == ["p1"]
        assert config.database_ids == []

    def test_swagger_config_validation(self):
        """Test SwaggerSourceConfig validation."""
        config = SwaggerSourceConfig(url="https://api.example.com/openapi.yaml")
        assert config.url == "https://api.example.com/openapi.yaml"

    def test_source_disabled(self):
        """Test disabled source."""
        source = Source(
            id="s1",
            name="Disabled Source",
            source_type="notion",
            config=NotionSourceConfig(page_ids=["p1"]),
            enabled=False,
        )
        assert source.enabled is False

    def test_update_sync_time(self):
        """Test update_sync_time updates timestamps."""
        source = Source(
            id="s1",
            name="Test Source",
            source_type="notion",
            config=NotionSourceConfig(page_ids=["p1"]),
        )
        assert source.last_synced_at is None
        old_updated = source.updated_at

        source.update_sync_time()

        assert source.last_synced_at is not None
        assert source.updated_at >= old_updated

    def test_model_dump_json_safe_with_last_synced(self):
        """Test model_dump_json_safe with last_synced_at set."""
        source = Source(
            id="s1",
            name="Test Source",
            source_type="notion",
            config=NotionSourceConfig(page_ids=["p1"]),
        )
        source.update_sync_time()

        data = source.model_dump_json_safe()

        assert "last_synced_at" in data
        assert isinstance(data["last_synced_at"], str)
        assert isinstance(data["created_at"], str)
        assert isinstance(data["updated_at"], str)

    def test_from_json_safe_with_last_synced(self):
        """Test from_json_safe parses last_synced_at correctly."""
        now = datetime.now(UTC)
        data = {
            "id": "s1",
            "name": "Test Source",
            "source_type": "notion",
            "config": {"page_ids": ["p1"], "database_ids": []},
            "enabled": True,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "last_synced_at": now.isoformat(),
        }

        source = Source.from_json_safe(data)

        assert isinstance(source.last_synced_at, datetime)
        assert isinstance(source.created_at, datetime)
        assert isinstance(source.updated_at, datetime)

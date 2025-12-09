"""Tests for configuration module."""

import tempfile
from pathlib import Path

import pytest

from src.config import (
    AISettings,
    ChunkingSettings,
    EmbeddingSettings,
    NotionSettings,
    QdrantSettings,
    SchedulerSettings,
    Settings,
    get_settings,
)


class TestQdrantSettings:
    """Tests for QdrantSettings."""

    def test_default_values(self):
        """Test QdrantSettings default values."""
        settings = QdrantSettings()
        assert settings.host == "localhost"
        assert settings.port == 6333
        assert settings.collection_name == "rag_documents"
        assert settings.api_key is None

    def test_url_property(self):
        """Test url property returns formatted URL."""
        settings = QdrantSettings()
        assert settings.url == "http://localhost:6333"

    def test_url_property_with_custom_values(self):
        """Test url property with custom host and port."""
        # Use model_construct to bypass validation and set values directly
        settings = QdrantSettings.model_construct(
            host="custom-host",
            port=6334,
            collection_name="rag_documents",
            api_key=None,
        )
        assert settings.url == "http://custom-host:6334"


class TestNotionSettings:
    """Tests for NotionSettings."""

    def test_default_values(self):
        """Test NotionSettings default values."""
        settings = NotionSettings()
        assert settings.api_key == ""
        assert settings.rate_limit_delay == 0.35
        assert settings.max_retries == 5


class TestEmbeddingSettings:
    """Tests for EmbeddingSettings."""

    def test_default_values(self):
        """Test EmbeddingSettings default values."""
        settings = EmbeddingSettings()
        assert settings.model_name == "intfloat/multilingual-e5-large-instruct"
        assert settings.dimension == 1024
        assert settings.batch_size == 32


class TestChunkingSettings:
    """Tests for ChunkingSettings."""

    def test_default_values(self):
        """Test ChunkingSettings default values."""
        settings = ChunkingSettings()
        assert settings.chunk_size == 1000
        assert settings.chunk_overlap == 200


class TestSchedulerSettings:
    """Tests for SchedulerSettings."""

    def test_default_values(self):
        """Test SchedulerSettings default values."""
        settings = SchedulerSettings()
        assert settings.enabled is False
        assert settings.cron_expression == "0 6 * * *"
        assert settings.timezone == "Asia/Seoul"


class TestAISettings:
    """Tests for AISettings."""

    def test_default_values(self):
        """Test AISettings default values."""
        settings = AISettings()
        assert settings.enabled is False
        assert settings.api_key == ""
        assert settings.model == "claude-3-haiku-20240307"
        assert settings.max_tokens == 1024
        assert settings.timeout == 30.0
        assert settings.cache_enabled is True


class TestSettings:
    """Tests for main Settings class."""

    def test_default_values(self):
        """Test Settings default values."""
        settings = Settings()
        assert settings.app_name == "rag-indexer"
        assert settings.debug is False
        assert settings.log_level == "INFO"
        assert settings.data_dir == Path("data")

    def test_sub_settings_initialization(self):
        """Test that sub-settings are properly initialized."""
        settings = Settings()
        assert isinstance(settings.qdrant, QdrantSettings)
        assert isinstance(settings.notion, NotionSettings)
        assert isinstance(settings.embedding, EmbeddingSettings)
        assert isinstance(settings.chunking, ChunkingSettings)
        assert isinstance(settings.scheduler, SchedulerSettings)
        assert isinstance(settings.ai, AISettings)

    def test_ensure_data_dir_creates_directory(self):
        """Test ensure_data_dir creates directory if not exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_data_dir = Path(tmpdir) / "new_data_dir"
            # Use model_construct to set data_dir directly
            settings = Settings.model_construct(
                app_name="rag-indexer",
                debug=False,
                log_level="INFO",
                data_dir=new_data_dir,
                qdrant=QdrantSettings(),
                notion=NotionSettings(),
                embedding=EmbeddingSettings(),
                chunking=ChunkingSettings(),
                scheduler=SchedulerSettings(),
                ai=AISettings(),
            )

            # Directory should not exist yet
            assert not new_data_dir.exists()

            # Call ensure_data_dir
            result = settings.ensure_data_dir()

            # Directory should now exist
            assert new_data_dir.exists()
            assert new_data_dir.is_dir()
            assert result == new_data_dir

    def test_ensure_data_dir_existing_directory(self):
        """Test ensure_data_dir with existing directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            existing_dir = Path(tmpdir)
            # Use model_construct to set data_dir directly
            settings = Settings.model_construct(
                app_name="rag-indexer",
                debug=False,
                log_level="INFO",
                data_dir=existing_dir,
                qdrant=QdrantSettings(),
                notion=NotionSettings(),
                embedding=EmbeddingSettings(),
                chunking=ChunkingSettings(),
                scheduler=SchedulerSettings(),
                ai=AISettings(),
            )

            # Directory already exists
            assert existing_dir.exists()

            # Should not raise and return the path
            result = settings.ensure_data_dir()
            assert result == existing_dir

    def test_ensure_data_dir_nested_directory(self):
        """Test ensure_data_dir creates nested directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "level1" / "level2" / "level3"
            # Use model_construct to set data_dir directly
            settings = Settings.model_construct(
                app_name="rag-indexer",
                debug=False,
                log_level="INFO",
                data_dir=nested_dir,
                qdrant=QdrantSettings(),
                notion=NotionSettings(),
                embedding=EmbeddingSettings(),
                chunking=ChunkingSettings(),
                scheduler=SchedulerSettings(),
                ai=AISettings(),
            )

            # Nested directory should not exist
            assert not nested_dir.exists()

            # Call ensure_data_dir
            result = settings.ensure_data_dir()

            # All nested directories should now exist
            assert nested_dir.exists()
            assert nested_dir.is_dir()
            assert result == nested_dir


class TestGetSettings:
    """Tests for get_settings function."""

    def test_returns_settings_instance(self):
        """Test get_settings returns Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_returns_valid_settings(self):
        """Test get_settings returns valid settings with defaults."""
        settings = get_settings()
        assert settings.app_name == "rag-indexer"
        assert isinstance(settings.qdrant, QdrantSettings)

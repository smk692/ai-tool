"""Tests for embedder service module."""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest

from src.models import Chunk
from src.services.embedder import Embedder, get_embedder


class TestEmbedder:
    """Tests for Embedder class."""

    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock EmbeddingModel."""
        with patch("src.services.embedder.get_embedding_model") as mock:
            mock_model = MagicMock()
            mock_model.dimension = 1024
            mock_model.embed_documents.return_value = [[0.1] * 1024]
            mock_model.embed_query.return_value = [0.2] * 1024
            mock.return_value = mock_model
            yield mock_model

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        return [
            Chunk(
                document_id=str(uuid4()),
                source_id=str(uuid4()),
                chunk_index=i,
                text=f"Test chunk content {i}",
                metadata={"key": "value"},
            )
            for i in range(3)
        ]

    # ==================== Initialization ====================

    def test_initialization_with_defaults(self):
        """Test Embedder initializes with default values."""
        embedder = Embedder()
        assert embedder.model_name == "intfloat/multilingual-e5-large-instruct"
        assert embedder.batch_size == 32
        assert embedder._model is None

    def test_initialization_with_custom_values(self):
        """Test Embedder initializes with custom values."""
        embedder = Embedder(
            model_name="custom-model",
            batch_size=64,
        )
        assert embedder.model_name == "custom-model"
        assert embedder.batch_size == 64

    # ==================== Model Lazy Loading ====================

    def test_model_lazy_loading(self, mock_embedding_model):
        """Test model is lazy-loaded on first access."""
        with patch("src.services.embedder.get_embedding_model") as mock_get:
            mock_get.return_value = mock_embedding_model

            embedder = Embedder()
            assert embedder._model is None

            # Access model property
            _ = embedder.model
            mock_get.assert_called_once_with(embedder.model_name)
            assert embedder._model is not None

    def test_model_cached_after_load(self, mock_embedding_model):
        """Test model is cached after first load."""
        with patch("src.services.embedder.get_embedding_model") as mock_get:
            mock_get.return_value = mock_embedding_model

            embedder = Embedder()

            # Access model multiple times
            _ = embedder.model
            _ = embedder.model
            _ = embedder.model

            # Should only be called once
            mock_get.assert_called_once()

    # ==================== Dimension Property ====================

    def test_dimension_property(self, mock_embedding_model):
        """Test dimension property returns model dimension."""
        with patch("src.services.embedder.get_embedding_model") as mock_get:
            mock_get.return_value = mock_embedding_model

            embedder = Embedder()
            dimension = embedder.dimension

            assert dimension == 1024

    # ==================== Embed Chunks ====================

    def test_embed_chunks_empty_list(self, mock_embedding_model):
        """Test embed_chunks with empty list returns empty list."""
        with patch("src.services.embedder.get_embedding_model") as mock_get:
            mock_get.return_value = mock_embedding_model

            embedder = Embedder()
            result = embedder.embed_chunks([])

            assert result == []
            mock_embedding_model.embed_documents.assert_not_called()

    def test_embed_chunks_attaches_embeddings(self, mock_embedding_model, sample_chunks):
        """Test embed_chunks attaches embeddings to chunks."""
        embeddings = [[0.1 * (i + 1)] * 1024 for i in range(len(sample_chunks))]
        mock_embedding_model.embed_documents.return_value = embeddings

        with patch("src.services.embedder.get_embedding_model") as mock_get:
            mock_get.return_value = mock_embedding_model

            embedder = Embedder()
            result = embedder.embed_chunks(sample_chunks)

            assert len(result) == len(sample_chunks)
            for i, chunk in enumerate(result):
                assert chunk.embedding == embeddings[i]

    def test_embed_chunks_uses_batch_size(self, mock_embedding_model, sample_chunks):
        """Test embed_chunks passes batch_size to model."""
        with patch("src.services.embedder.get_embedding_model") as mock_get:
            mock_get.return_value = mock_embedding_model
            mock_embedding_model.embed_documents.return_value = [[0.1] * 1024] * 3

            embedder = Embedder(batch_size=16)
            embedder.embed_chunks(sample_chunks)

            call_kwargs = mock_embedding_model.embed_documents.call_args[1]
            assert call_kwargs["batch_size"] == 16

    def test_embed_chunks_extracts_texts(self, mock_embedding_model, sample_chunks):
        """Test embed_chunks extracts text from chunks."""
        with patch("src.services.embedder.get_embedding_model") as mock_get:
            mock_get.return_value = mock_embedding_model
            mock_embedding_model.embed_documents.return_value = [[0.1] * 1024] * 3

            embedder = Embedder()
            embedder.embed_chunks(sample_chunks)

            call_args = mock_embedding_model.embed_documents.call_args[0]
            texts = call_args[0]
            assert len(texts) == 3
            assert texts[0] == "Test chunk content 0"

    def test_embed_chunks_show_progress(self, mock_embedding_model, sample_chunks):
        """Test embed_chunks passes show_progress parameter."""
        with patch("src.services.embedder.get_embedding_model") as mock_get:
            mock_get.return_value = mock_embedding_model
            mock_embedding_model.embed_documents.return_value = [[0.1] * 1024] * 3

            embedder = Embedder()
            embedder.embed_chunks(sample_chunks, show_progress=True)

            call_kwargs = mock_embedding_model.embed_documents.call_args[1]
            assert call_kwargs["show_progress"] is True

    def test_embed_chunks_modifies_in_place(self, mock_embedding_model, sample_chunks):
        """Test embed_chunks modifies chunks in place."""
        embeddings = [[0.1] * 1024] * len(sample_chunks)
        mock_embedding_model.embed_documents.return_value = embeddings

        with patch("src.services.embedder.get_embedding_model") as mock_get:
            mock_get.return_value = mock_embedding_model

            embedder = Embedder()
            original_chunks = sample_chunks  # Same reference
            result = embedder.embed_chunks(sample_chunks)

            # Should be same objects
            assert result is original_chunks
            for chunk in original_chunks:
                assert chunk.embedding is not None

    # ==================== Embed Query ====================

    def test_embed_query_returns_list(self, mock_embedding_model):
        """Test embed_query returns embedding as list."""
        with patch("src.services.embedder.get_embedding_model") as mock_get:
            mock_get.return_value = mock_embedding_model

            embedder = Embedder()
            result = embedder.embed_query("test query")

            assert isinstance(result, list)
            assert len(result) == 1024

    def test_embed_query_calls_model(self, mock_embedding_model):
        """Test embed_query calls model.embed_query."""
        with patch("src.services.embedder.get_embedding_model") as mock_get:
            mock_get.return_value = mock_embedding_model

            embedder = Embedder()
            embedder.embed_query("my search query")

            mock_embedding_model.embed_query.assert_called_once_with("my search query")

    def test_embed_query_korean_text(self, mock_embedding_model):
        """Test embed_query with Korean text."""
        with patch("src.services.embedder.get_embedding_model") as mock_get:
            mock_get.return_value = mock_embedding_model

            embedder = Embedder()
            result = embedder.embed_query("한글 검색 쿼리입니다")

            assert result is not None
            mock_embedding_model.embed_query.assert_called_once_with("한글 검색 쿼리입니다")


class TestGetEmbedder:
    """Tests for get_embedder function."""

    def test_get_embedder_returns_instance(self):
        """Test get_embedder returns Embedder instance."""
        import src.services.embedder

        # Reset global
        src.services.embedder._embedder = None

        embedder = get_embedder()
        assert isinstance(embedder, Embedder)

    def test_get_embedder_cached(self):
        """Test get_embedder returns cached instance."""
        import src.services.embedder

        # Reset global
        src.services.embedder._embedder = None

        embedder1 = get_embedder()
        embedder2 = get_embedder()
        assert embedder1 is embedder2

    def test_get_embedder_with_custom_params(self):
        """Test get_embedder with custom parameters."""
        import src.services.embedder

        # Reset global
        src.services.embedder._embedder = None

        embedder = get_embedder(
            model_name="custom-model",
            batch_size=64,
        )
        assert embedder.model_name == "custom-model"
        assert embedder.batch_size == 64


class TestEmbedderIntegration:
    """Integration-style tests for embedder."""

    @pytest.fixture
    def mock_full_workflow(self):
        """Create mock for full workflow testing."""
        with patch("src.services.embedder.get_embedding_model") as mock_get:
            mock_model = MagicMock()
            mock_model.dimension = 1024

            def embed_docs_side_effect(texts, **kwargs):
                # Return different embeddings for each text
                return [[float(i) / 100] * 1024 for i in range(len(texts))]

            mock_model.embed_documents.side_effect = embed_docs_side_effect
            mock_model.embed_query.return_value = [0.5] * 1024
            mock_get.return_value = mock_model
            yield mock_model

    def test_full_embedding_workflow(self, mock_full_workflow):
        """Test complete embedding workflow."""
        # Create chunks
        chunks = [
            Chunk(
                document_id=str(uuid4()),
                source_id=str(uuid4()),
                chunk_index=i,
                text=f"Document content {i}",
            )
            for i in range(5)
        ]

        # Embed chunks
        embedder = get_embedder()
        embedded_chunks = embedder.embed_chunks(chunks)

        # Verify all chunks have embeddings
        assert len(embedded_chunks) == 5
        for i, chunk in enumerate(embedded_chunks):
            assert chunk.embedding is not None
            assert len(chunk.embedding) == 1024
            # Each chunk should have different embedding
            assert chunk.embedding[0] == float(i) / 100

    def test_query_embedding_workflow(self, mock_full_workflow):
        """Test query embedding workflow."""
        embedder = get_embedder()

        # Embed multiple queries
        queries = ["query 1", "query 2", "query 3"]
        embeddings = [embedder.embed_query(q) for q in queries]

        # All should return valid embeddings
        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == 1024

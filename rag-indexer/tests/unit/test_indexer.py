"""Tests for indexer service module."""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from src.models import Chunk, ChunkType, Document, Source, SourceType, NotionSourceConfig
from src.services.indexer import Indexer, get_indexer


class TestIndexerInit:
    """Tests for Indexer initialization."""

    def test_init_with_defaults(self):
        """Test Indexer initializes with default values."""
        indexer = Indexer()
        assert indexer._vector_store is None
        assert indexer._embedder is None
        assert indexer._chunker is None
        assert indexer._hierarchical_chunker is None
        assert indexer._storage is None
        assert indexer._ai_extractor is None
        assert indexer.collection_name == "rag_documents"
        assert indexer._collection_initialized is False
        assert indexer._enable_ai_extraction is False
        assert indexer._enable_hierarchical is False

    def test_init_with_custom_values(self):
        """Test Indexer initializes with custom values."""
        mock_vs = MagicMock()
        mock_embedder = MagicMock()
        mock_chunker = MagicMock()
        mock_hier_chunker = MagicMock()
        mock_storage = MagicMock()
        mock_ai_extractor = MagicMock()

        indexer = Indexer(
            vector_store=mock_vs,
            embedder=mock_embedder,
            chunker=mock_chunker,
            hierarchical_chunker=mock_hier_chunker,
            storage=mock_storage,
            ai_extractor=mock_ai_extractor,
            collection_name="custom_collection",
            enable_ai_extraction=True,
            enable_hierarchical=True,
        )

        assert indexer._vector_store is mock_vs
        assert indexer._embedder is mock_embedder
        assert indexer._chunker is mock_chunker
        assert indexer._hierarchical_chunker is mock_hier_chunker
        assert indexer._storage is mock_storage
        assert indexer._ai_extractor is mock_ai_extractor
        assert indexer.collection_name == "custom_collection"
        assert indexer._enable_ai_extraction is True
        assert indexer._enable_hierarchical is True


class TestIndexerLazyLoading:
    """Tests for Indexer lazy loading properties."""

    def test_vector_store_lazy_load(self):
        """Test vector_store is lazy loaded."""
        with patch("src.services.indexer.get_vector_store") as mock_get:
            mock_vs = MagicMock()
            mock_get.return_value = mock_vs

            indexer = Indexer()
            assert indexer._vector_store is None

            # Access property
            result = indexer.vector_store
            mock_get.assert_called_once()
            assert result is mock_vs

    def test_vector_store_cached(self):
        """Test vector_store is cached after first load."""
        mock_vs = MagicMock()
        indexer = Indexer(vector_store=mock_vs)

        # Access multiple times
        _ = indexer.vector_store
        _ = indexer.vector_store
        _ = indexer.vector_store

        # Should be same instance
        assert indexer.vector_store is mock_vs

    def test_embedder_lazy_load(self):
        """Test embedder is lazy loaded."""
        with patch("src.services.indexer.get_embedder") as mock_get:
            mock_embedder = MagicMock()
            mock_get.return_value = mock_embedder

            indexer = Indexer()
            assert indexer._embedder is None

            result = indexer.embedder
            mock_get.assert_called_once()
            assert result is mock_embedder

    def test_chunker_lazy_load(self):
        """Test chunker is lazy loaded."""
        with patch("src.services.indexer.get_chunker") as mock_get:
            mock_chunker = MagicMock()
            mock_get.return_value = mock_chunker

            indexer = Indexer()
            assert indexer._chunker is None

            result = indexer.chunker
            mock_get.assert_called_once()
            assert result is mock_chunker

    def test_hierarchical_chunker_lazy_load(self):
        """Test hierarchical_chunker is lazy loaded."""
        with patch("src.services.indexer.get_hierarchical_chunker") as mock_get:
            mock_hier = MagicMock()
            mock_get.return_value = mock_hier

            indexer = Indexer()
            assert indexer._hierarchical_chunker is None

            result = indexer.hierarchical_chunker
            mock_get.assert_called_once()
            assert result is mock_hier

    def test_storage_lazy_load(self):
        """Test storage is lazy loaded."""
        with patch("src.services.indexer.get_storage") as mock_get:
            mock_storage = MagicMock()
            mock_get.return_value = mock_storage

            indexer = Indexer()
            assert indexer._storage is None

            result = indexer.storage
            mock_get.assert_called_once()
            assert result is mock_storage

    def test_ai_extractor_disabled(self):
        """Test ai_extractor returns None when disabled."""
        indexer = Indexer(enable_ai_extraction=False)
        assert indexer.ai_extractor is None

    def test_ai_extractor_enabled_lazy_load(self):
        """Test ai_extractor is lazy loaded when enabled."""
        with patch("src.services.indexer.get_ai_extractor") as mock_get:
            mock_ai = MagicMock()
            mock_get.return_value = mock_ai

            indexer = Indexer(enable_ai_extraction=True)
            assert indexer._ai_extractor is None

            result = indexer.ai_extractor
            mock_get.assert_called_once()
            assert result is mock_ai

    def test_ai_extractor_returns_provided_instance(self):
        """Test ai_extractor returns provided instance when enabled."""
        mock_ai = MagicMock()
        indexer = Indexer(ai_extractor=mock_ai, enable_ai_extraction=True)
        assert indexer.ai_extractor is mock_ai


class TestExtractAIMetadata:
    """Tests for _extract_ai_metadata method."""

    @pytest.fixture
    def sample_document(self):
        """Create sample document."""
        return Document(
            id=str(uuid4()),
            source_id=str(uuid4()),
            external_id="test-external",
            title="Test Document",
            content="Test content for extraction",
        )

    def test_extract_ai_metadata_disabled(self, sample_document):
        """Test extraction returns empty dict when disabled."""
        indexer = Indexer(enable_ai_extraction=False)
        result = indexer._extract_ai_metadata(sample_document)
        assert result == {}

    def test_extract_ai_metadata_enabled(self, sample_document):
        """Test extraction returns metadata when enabled."""
        mock_ai = MagicMock()
        mock_extracted = MagicMock()
        mock_extracted.content_type = "guide"
        mock_extracted.topics = ["testing", "python"]
        mock_extracted.difficulty = "intermediate"
        mock_extracted.has_code_samples = True
        mock_extracted.key_entities = ["pytest", "mock"]
        mock_extracted.summary = "A guide about testing"
        mock_ai.extract.return_value = mock_extracted

        indexer = Indexer(ai_extractor=mock_ai, enable_ai_extraction=True)
        result = indexer._extract_ai_metadata(sample_document)

        assert result["ai_content_type"] == "guide"
        assert result["ai_topics"] == ["testing", "python"]
        assert result["ai_difficulty"] == "intermediate"
        assert result["ai_has_code_samples"] is True
        assert result["ai_key_entities"] == ["pytest", "mock"]
        assert result["ai_summary"] == "A guide about testing"

    def test_extract_ai_metadata_handles_exception(self, sample_document):
        """Test extraction returns empty dict on exception."""
        mock_ai = MagicMock()
        mock_ai.extract.side_effect = Exception("AI error")

        indexer = Indexer(ai_extractor=mock_ai, enable_ai_extraction=True)
        result = indexer._extract_ai_metadata(sample_document)
        assert result == {}


class TestEnsureCollection:
    """Tests for ensure_collection method."""

    def test_ensure_collection_creates_once(self):
        """Test collection is only created once."""
        mock_vs = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.dimension = 1024

        indexer = Indexer(vector_store=mock_vs, embedder=mock_embedder)

        # Call multiple times
        indexer.ensure_collection()
        indexer.ensure_collection()
        indexer.ensure_collection()

        # Should only call once
        mock_vs.ensure_collection.assert_called_once_with(dimension=1024)

    def test_ensure_collection_uses_embedder_dimension(self):
        """Test collection uses embedder dimension."""
        mock_vs = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.dimension = 768

        indexer = Indexer(vector_store=mock_vs, embedder=mock_embedder)
        indexer.ensure_collection()

        mock_vs.ensure_collection.assert_called_once_with(dimension=768)


class TestIndexDocument:
    """Tests for index_document method."""

    @pytest.fixture
    def mock_indexer(self):
        """Create indexer with mocked dependencies."""
        mock_vs = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.dimension = 1024
        mock_chunker = MagicMock()
        mock_hier_chunker = MagicMock()
        mock_storage = MagicMock()

        indexer = Indexer(
            vector_store=mock_vs,
            embedder=mock_embedder,
            chunker=mock_chunker,
            hierarchical_chunker=mock_hier_chunker,
            storage=mock_storage,
        )
        return indexer

    @pytest.fixture
    def sample_document(self):
        """Create sample document."""
        return Document(
            id=str(uuid4()),
            source_id=str(uuid4()),
            external_id="test-external",
            title="Test Document",
            content="Test content",
        )

    @pytest.fixture
    def sample_source(self):
        """Create sample source."""
        return Source(
            id=str(uuid4()),
            name="Test Source",
            source_type=SourceType.NOTION,
            config=NotionSourceConfig(
                page_ids=["page1"],
                database_ids=[],
            ),
        )

    @pytest.fixture
    def sample_chunks(self, sample_document):
        """Create sample chunks."""
        return [
            Chunk(
                document_id=sample_document.id,
                source_id=sample_document.source_id,
                chunk_index=i,
                text=f"Chunk {i} content",
                embedding=[0.1] * 1024,
            )
            for i in range(3)
        ]

    def test_index_document_standard_chunking(
        self, mock_indexer, sample_document, sample_source, sample_chunks
    ):
        """Test indexing with standard chunking."""
        mock_indexer._chunker.chunk_document.return_value = sample_chunks

        result = mock_indexer.index_document(sample_document, sample_source)

        assert result == 3
        mock_indexer._chunker.chunk_document.assert_called_once_with(sample_document)
        mock_indexer._embedder.embed_chunks.assert_called_once()
        mock_indexer._vector_store.upsert.assert_called_once()
        mock_indexer._storage.upsert_document.assert_called_once()

    def test_index_document_hierarchical_chunking(
        self, mock_indexer, sample_document, sample_source, sample_chunks
    ):
        """Test indexing with hierarchical chunking."""
        mock_indexer._enable_hierarchical = True
        mock_indexer._hierarchical_chunker.chunk_document.return_value = sample_chunks

        result = mock_indexer.index_document(sample_document, sample_source)

        assert result == 3
        mock_indexer._hierarchical_chunker.chunk_document.assert_called_once_with(
            sample_document, include_parents=True
        )

    def test_index_document_hierarchical_override(
        self, mock_indexer, sample_document, sample_source, sample_chunks
    ):
        """Test hierarchical parameter overrides instance setting."""
        mock_indexer._enable_hierarchical = False
        mock_indexer._hierarchical_chunker.chunk_document.return_value = sample_chunks

        result = mock_indexer.index_document(
            sample_document, sample_source, hierarchical=True
        )

        assert result == 3
        mock_indexer._hierarchical_chunker.chunk_document.assert_called_once()

    def test_index_document_empty_chunks(
        self, mock_indexer, sample_document, sample_source
    ):
        """Test indexing returns 0 for empty document."""
        mock_indexer._chunker.chunk_document.return_value = []

        result = mock_indexer.index_document(sample_document, sample_source)

        assert result == 0
        mock_indexer._embedder.embed_chunks.assert_not_called()
        mock_indexer._vector_store.upsert.assert_not_called()

    def test_index_document_deletes_existing_chunks(
        self, mock_indexer, sample_document, sample_source, sample_chunks
    ):
        """Test existing chunks are deleted before indexing."""
        mock_indexer._chunker.chunk_document.return_value = sample_chunks

        mock_indexer.index_document(sample_document, sample_source)

        mock_indexer._vector_store.delete_by_filter.assert_called_once_with(
            field="document_id",
            value=sample_document.id,
        )

    def test_index_document_with_ai_extraction(
        self, mock_indexer, sample_document, sample_source, sample_chunks
    ):
        """Test indexing with AI extraction enabled."""
        mock_ai = MagicMock()
        mock_extracted = MagicMock()
        mock_extracted.content_type = "guide"
        mock_extracted.topics = []
        mock_extracted.difficulty = "basic"
        mock_extracted.has_code_samples = False
        mock_extracted.key_entities = []
        mock_extracted.summary = "Test summary"
        mock_ai.extract.return_value = mock_extracted

        mock_indexer._ai_extractor = mock_ai
        mock_indexer._enable_ai_extraction = True
        mock_indexer._chunker.chunk_document.return_value = sample_chunks

        mock_indexer.index_document(sample_document, sample_source)

        mock_ai.extract.assert_called_once_with(
            title=sample_document.title,
            content=sample_document.content,
        )


class TestIndexDocuments:
    """Tests for index_documents method."""

    @pytest.fixture
    def mock_indexer(self):
        """Create indexer with mocked dependencies."""
        mock_vs = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.dimension = 1024
        mock_chunker = MagicMock()
        mock_storage = MagicMock()

        indexer = Indexer(
            vector_store=mock_vs,
            embedder=mock_embedder,
            chunker=mock_chunker,
            storage=mock_storage,
        )
        return indexer

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents."""
        source_id = str(uuid4())
        return [
            Document(
                id=str(uuid4()),
                source_id=source_id,
                external_id=f"doc-{i}",
                title=f"Document {i}",
                content=f"Content {i}",
            )
            for i in range(3)
        ]

    @pytest.fixture
    def sample_source(self):
        """Create sample source."""
        return Source(
            id=str(uuid4()),
            name="Test Source",
            source_type=SourceType.NOTION,
            config=NotionSourceConfig(
                page_ids=["page1"],
                database_ids=[],
            ),
        )

    def test_index_documents_success(
        self, mock_indexer, sample_documents, sample_source
    ):
        """Test indexing multiple documents."""
        # Each document creates 2 chunks
        mock_indexer._chunker.chunk_document.return_value = [
            Chunk(
                document_id=str(uuid4()),
                source_id=sample_source.id,
                chunk_index=0,
                text="Chunk",
                embedding=[0.1] * 1024,
            ),
            Chunk(
                document_id=str(uuid4()),
                source_id=sample_source.id,
                chunk_index=1,
                text="Chunk",
                embedding=[0.1] * 1024,
            ),
        ]

        result = mock_indexer.index_documents(sample_documents, sample_source)

        assert result["documents_processed"] == 3
        assert result["chunks_created"] == 6
        assert result["errors"] == 0

    def test_index_documents_with_error(
        self, mock_indexer, sample_documents, sample_source
    ):
        """Test indexing propagates errors."""
        mock_indexer._chunker.chunk_document.side_effect = Exception("Chunking error")

        with pytest.raises(Exception, match="Chunking error"):
            mock_indexer.index_documents(sample_documents, sample_source)


class TestDeleteMethods:
    """Tests for delete methods."""

    def test_delete_document_chunks(self):
        """Test deleting document chunks."""
        mock_vs = MagicMock()
        mock_vs.delete_by_filter.return_value = True

        indexer = Indexer(vector_store=mock_vs)
        result = indexer.delete_document_chunks("doc-123")

        assert result is True
        mock_vs.delete_by_filter.assert_called_once_with(
            field="document_id",
            value="doc-123",
        )

    def test_delete_source_chunks(self):
        """Test deleting source chunks."""
        mock_vs = MagicMock()
        mock_vs.delete_by_filter.return_value = True

        indexer = Indexer(vector_store=mock_vs)
        result = indexer.delete_source_chunks("source-456")

        assert result is True
        mock_vs.delete_by_filter.assert_called_once_with(
            field="source_id",
            value="source-456",
        )


class TestSearch:
    """Tests for search methods."""

    @pytest.fixture
    def mock_indexer(self):
        """Create indexer with mocked dependencies."""
        mock_vs = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.dimension = 1024
        mock_embedder.embed_query.return_value = [0.1] * 1024

        indexer = Indexer(vector_store=mock_vs, embedder=mock_embedder)
        indexer._collection_initialized = True
        return indexer

    def test_search_basic(self, mock_indexer):
        """Test basic search."""
        mock_indexer._vector_store.search.return_value = [
            {"id": "1", "score": 0.95, "payload": {"text": "Result 1"}},
            {"id": "2", "score": 0.90, "payload": {"text": "Result 2"}},
        ]

        results = mock_indexer.search("test query", limit=5)

        assert len(results) == 2
        mock_indexer._embedder.embed_query.assert_called_once_with("test query")
        mock_indexer._vector_store.search.assert_called_once()

    def test_search_with_source_filter(self, mock_indexer):
        """Test search with source filter."""
        mock_indexer._vector_store.search.return_value = []

        mock_indexer.search("query", source_id="source-123")

        call_kwargs = mock_indexer._vector_store.search.call_args[1]
        assert call_kwargs["filter_conditions"]["source_id"] == "source-123"

    def test_search_with_source_type_filter(self, mock_indexer):
        """Test search with source type filter."""
        mock_indexer._vector_store.search.return_value = []

        mock_indexer.search("query", source_type="notion")

        call_kwargs = mock_indexer._vector_store.search.call_args[1]
        assert call_kwargs["filter_conditions"]["source_type"] == "notion"

    def test_search_with_language_filter(self, mock_indexer):
        """Test search with language filter."""
        mock_indexer._vector_store.search.return_value = []

        mock_indexer.search("query", language="ko")

        call_kwargs = mock_indexer._vector_store.search.call_args[1]
        assert call_kwargs["filter_conditions"]["language"] == "ko"

    def test_search_with_content_type_filter(self, mock_indexer):
        """Test search with content type filter."""
        mock_indexer._vector_store.search.return_value = []

        mock_indexer.search("query", content_type="api_doc")

        call_kwargs = mock_indexer._vector_store.search.call_args[1]
        assert call_kwargs["filter_conditions"]["content_type"] == "api_doc"

    def test_search_with_http_method_filter(self, mock_indexer):
        """Test search with HTTP method filter."""
        mock_indexer._vector_store.search.return_value = []

        mock_indexer.search("query", http_method="GET")

        call_kwargs = mock_indexer._vector_store.search.call_args[1]
        assert call_kwargs["filter_conditions"]["http_method"] == "GET"

    def test_search_with_chunk_type_filter(self, mock_indexer):
        """Test search with chunk type filter."""
        mock_indexer._vector_store.search.return_value = []

        mock_indexer.search("query", chunk_type="child")

        call_kwargs = mock_indexer._vector_store.search.call_args[1]
        assert call_kwargs["filter_conditions"]["chunk_type"] == "child"

    def test_search_with_score_threshold(self, mock_indexer):
        """Test search with score threshold."""
        mock_indexer._vector_store.search.return_value = []

        mock_indexer.search("query", score_threshold=0.8)

        call_kwargs = mock_indexer._vector_store.search.call_args[1]
        assert call_kwargs["score_threshold"] == 0.8

    def test_search_with_multiple_filters(self, mock_indexer):
        """Test search with multiple filters."""
        mock_indexer._vector_store.search.return_value = []

        mock_indexer.search(
            "query",
            source_id="src-1",
            source_type="swagger",
            language="en",
            http_method="POST",
        )

        call_kwargs = mock_indexer._vector_store.search.call_args[1]
        assert call_kwargs["filter_conditions"]["source_id"] == "src-1"
        assert call_kwargs["filter_conditions"]["source_type"] == "swagger"
        assert call_kwargs["filter_conditions"]["language"] == "en"
        assert call_kwargs["filter_conditions"]["http_method"] == "POST"


class TestSearchWithParentContext:
    """Tests for search_with_parent_context method."""

    @pytest.fixture
    def mock_indexer(self):
        """Create indexer with mocked dependencies."""
        mock_vs = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.dimension = 1024
        mock_embedder.embed_query.return_value = [0.1] * 1024

        indexer = Indexer(vector_store=mock_vs, embedder=mock_embedder)
        indexer._collection_initialized = True
        return indexer

    def test_search_with_parent_context_enriches_results(self, mock_indexer):
        """Test search enriches results with parent context."""
        parent_id = str(uuid4())
        mock_indexer._vector_store.search.return_value = [
            {
                "id": "child-1",
                "score": 0.95,
                "payload": {"text": "Child text", "parent_id": parent_id},
            },
        ]
        mock_indexer._vector_store.get_point.return_value = {
            "id": parent_id,
            "payload": {"text": "Parent context text"},
        }

        results = mock_indexer.search_with_parent_context("query")

        assert len(results) == 1
        assert results[0]["parent_context"] == "Parent context text"
        mock_indexer._vector_store.get_point.assert_called_once_with(parent_id)

    def test_search_with_parent_context_no_parent(self, mock_indexer):
        """Test search handles results without parent."""
        mock_indexer._vector_store.search.return_value = [
            {
                "id": "child-1",
                "score": 0.95,
                "payload": {"text": "Child text"},  # No parent_id
            },
        ]

        results = mock_indexer.search_with_parent_context("query")

        assert len(results) == 1
        assert results[0]["parent_context"] is None
        mock_indexer._vector_store.get_point.assert_not_called()

    def test_search_with_parent_context_parent_not_found(self, mock_indexer):
        """Test search handles missing parent gracefully."""
        parent_id = str(uuid4())
        mock_indexer._vector_store.search.return_value = [
            {
                "id": "child-1",
                "score": 0.95,
                "payload": {"text": "Child text", "parent_id": parent_id},
            },
        ]
        mock_indexer._vector_store.get_point.return_value = None

        results = mock_indexer.search_with_parent_context("query")

        assert len(results) == 1
        assert results[0]["parent_context"] is None


class TestGetParentChunk:
    """Tests for get_parent_chunk method."""

    def test_get_parent_chunk_found(self):
        """Test retrieving existing parent chunk."""
        mock_vs = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.dimension = 1024

        parent_data = {
            "id": "parent-123",
            "payload": {"text": "Parent content"},
        }
        mock_vs.get_point.return_value = parent_data

        indexer = Indexer(vector_store=mock_vs, embedder=mock_embedder)
        result = indexer.get_parent_chunk("parent-123")

        assert result == parent_data
        mock_vs.get_point.assert_called_once_with("parent-123")

    def test_get_parent_chunk_not_found(self):
        """Test retrieving non-existent parent chunk."""
        mock_vs = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.dimension = 1024
        mock_vs.get_point.return_value = None

        indexer = Indexer(vector_store=mock_vs, embedder=mock_embedder)
        result = indexer.get_parent_chunk("nonexistent")

        assert result is None


class TestGetCollectionStats:
    """Tests for get_collection_stats method."""

    def test_get_collection_stats(self):
        """Test retrieving collection statistics."""
        mock_vs = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.dimension = 1024

        # Mock counts for different chunk types
        mock_vs.count.side_effect = [
            100,  # Total
            60,  # Standard
            20,  # Parent
            20,  # Child
        ]

        indexer = Indexer(
            vector_store=mock_vs,
            embedder=mock_embedder,
            collection_name="test_collection",
        )
        stats = indexer.get_collection_stats()

        assert stats["collection_name"] == "test_collection"
        assert stats["total_chunks"] == 100
        assert stats["standard_chunks"] == 60
        assert stats["parent_chunks"] == 20
        assert stats["child_chunks"] == 20


class TestGetIndexer:
    """Tests for get_indexer function."""

    def test_get_indexer_returns_instance(self):
        """Test get_indexer returns Indexer instance."""
        import src.services.indexer

        # Reset global
        src.services.indexer._indexer = None

        indexer = get_indexer()
        assert isinstance(indexer, Indexer)

    def test_get_indexer_cached(self):
        """Test get_indexer returns cached instance."""
        import src.services.indexer

        # Reset global
        src.services.indexer._indexer = None

        indexer1 = get_indexer()
        indexer2 = get_indexer()
        assert indexer1 is indexer2

    def test_get_indexer_with_collection_name(self):
        """Test get_indexer with custom collection name."""
        import src.services.indexer

        # Reset global
        src.services.indexer._indexer = None

        indexer = get_indexer(collection_name="custom_collection")
        assert indexer.collection_name == "custom_collection"

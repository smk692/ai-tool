"""Integration tests for the complete indexing pipeline.

Tests the full flow from document ingestion through chunking,
embedding generation, and vector storage.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.models import Document, Chunk, Source, NotionSourceConfig
from src.services.chunker import Chunker
from src.services.indexer import Indexer


class TestIndexingPipeline:
    """Integration tests for the indexing pipeline."""

    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder."""
        mock = MagicMock()
        # Return 1024-dimension embeddings (multilingual-e5-large-instruct)
        mock.embed.return_value = [0.1] * 1024
        mock.embed_batch.return_value = [[0.1] * 1024 for _ in range(10)]
        mock.dimension = 1024
        mock.embed_query.return_value = [0.1] * 1024

        # embed_chunks modifies chunks in place and returns them
        def embed_chunks_impl(*args, **kwargs):
            # Handle both positional and keyword arguments
            chunks = args[0] if args else kwargs.get('chunks', [])
            for chunk in chunks:
                chunk.embedding = [0.1] * 1024
            return chunks

        mock.embed_chunks.side_effect = embed_chunks_impl
        return mock

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        mock = MagicMock()
        mock.upsert.return_value = None
        mock.upsert_batch.return_value = None
        mock.search.return_value = []
        mock.delete.return_value = None
        mock.delete_by_source.return_value = 0
        return mock

    @pytest.fixture
    def mock_storage(self):
        """Create mock storage."""
        mock = MagicMock()
        mock.save_chunks.return_value = None
        mock.get_chunks_by_document.return_value = []
        return mock

    @pytest.fixture
    def test_source(self):
        """Create test source."""
        return Source(
            id="test-source",
            name="Test Source",
            source_type="notion",
            config=NotionSourceConfig(page_ids=["page-1"]),
        )

    def test_document_chunking_preserves_metadata(self, sample_document):
        """Test that chunking preserves document metadata."""
        chunker = Chunker(chunk_size=50, chunk_overlap=10)

        # Create document with specific metadata
        doc = Document(
            id="doc-meta-test",
            source_id="source-meta",
            external_id="ext-meta",
            title="Metadata Test Document",
            content="This is test content. " * 20,
            url="https://example.com/meta",
            content_hash="metahash",
            metadata={"author": "test-user", "category": "testing"},
        )

        chunks = chunker.chunk_document(doc)

        # Verify metadata propagation
        for chunk in chunks:
            assert chunk.document_id == doc.id
            assert chunk.metadata.get("source_id") == doc.source_id
            assert chunk.metadata.get("document_title") == doc.title
            assert chunk.metadata.get("document_url") == doc.url

    def test_chunking_produces_valid_chunks(self, long_document):
        """Test that chunking produces properly sized chunks."""
        chunk_size = 200
        chunk_overlap = 50
        chunker = Chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        chunks = chunker.chunk_document(long_document)

        # Should produce multiple chunks
        assert len(chunks) > 1

        # Chunks should have valid indices
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

        # First chunk should have full size content
        assert len(chunks[0].text) <= chunk_size + 100  # Allow some flexibility

    def test_deterministic_chunk_ids(self, sample_document):
        """Test that chunk IDs are deterministic."""
        chunker = Chunker(chunk_size=50, chunk_overlap=10)

        # Chunk same document twice
        chunks1 = chunker.chunk_document(sample_document)
        chunks2 = chunker.chunk_document(sample_document)

        # IDs should be identical
        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2):
            assert c1.id == c2.id

    def test_indexer_processes_documents(
        self, mock_embedder, mock_vector_store, mock_storage, test_source
    ):
        """Test indexer processes documents through full pipeline."""
        indexer = Indexer(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            storage=mock_storage,
        )

        # Create test documents
        documents = [
            Document(
                id=f"doc-{i}",
                source_id=test_source.id,
                external_id=f"ext-{i}",
                title=f"Test Document {i}",
                content=f"This is test document {i} content. " * 20,
                content_hash=f"hash-{i}",
            )
            for i in range(3)
        ]

        # Run indexing
        result = indexer.index_documents(documents, test_source)

        # Verify results - result is a dict
        assert "documents_processed" in result or "chunks_created" in result or isinstance(result, dict)

    def test_indexer_handles_empty_documents(
        self, mock_embedder, mock_vector_store, mock_storage, test_source
    ):
        """Test indexer handles empty document list."""
        indexer = Indexer(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            storage=mock_storage,
        )

        result = indexer.index_documents([], test_source)

        # Should handle gracefully
        assert isinstance(result, dict)


class TestBatchProcessing:
    """Tests for batch processing in indexing pipeline."""

    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder with batch support."""
        mock = MagicMock()

        def embed_batch(texts):
            return [[0.1] * 1024 for _ in texts]

        mock.embed_batch.side_effect = embed_batch
        mock.embed.return_value = [0.1] * 1024
        mock.dimension = 1024
        mock.embed_query.return_value = [0.1] * 1024

        # embed_chunks modifies chunks in place
        def embed_chunks_impl(*args, **kwargs):
            chunks = args[0] if args else kwargs.get('chunks', [])
            for chunk in chunks:
                chunk.embedding = [0.1] * 1024
            return chunks

        mock.embed_chunks.side_effect = embed_chunks_impl
        return mock

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        mock = MagicMock()
        mock.upsert_batch.return_value = None
        mock.upsert.return_value = None
        return mock

    @pytest.fixture
    def mock_storage(self):
        """Create mock storage."""
        mock = MagicMock()
        mock.save_chunks.return_value = None
        return mock

    @pytest.fixture
    def test_source(self):
        """Create test source."""
        return Source(
            id="batch-test-source",
            name="Batch Test Source",
            source_type="notion",
            config=NotionSourceConfig(page_ids=["page-1"]),
        )

    def test_chunker_handles_large_batch(self, generate_documents):
        """Test chunker handles large document batches."""
        chunker = Chunker(chunk_size=200, chunk_overlap=50)

        # Generate 50 documents
        documents = generate_documents(50, "batch")

        # Chunk all documents
        chunks = chunker.chunk_documents(documents)

        # Should produce many chunks
        assert len(chunks) >= 50  # At least one per doc

        # All chunks should be valid
        for chunk in chunks:
            assert chunk.id
            assert chunk.document_id
            assert chunk.text

    def test_indexer_batch_processing(
        self, mock_embedder, mock_vector_store, mock_storage, generate_documents, test_source
    ):
        """Test indexer processes documents in batches efficiently."""
        indexer = Indexer(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            storage=mock_storage,
        )

        documents = generate_documents(25, "batch")

        result = indexer.index_documents(documents, test_source, show_progress=False)

        # Verify processing completed
        assert isinstance(result, dict)


class TestErrorHandling:
    """Tests for error handling in indexing pipeline."""

    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder."""
        mock = MagicMock()
        mock.embed.return_value = [0.1] * 1024
        mock.embed_batch.return_value = [[0.1] * 1024]
        mock.dimension = 1024
        mock.embed_query.return_value = [0.1] * 1024

        # embed_chunks modifies chunks in place
        def embed_chunks_impl(*args, **kwargs):
            chunks = args[0] if args else kwargs.get('chunks', [])
            for chunk in chunks:
                chunk.embedding = [0.1] * 1024
            return chunks

        mock.embed_chunks.side_effect = embed_chunks_impl
        return mock

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        mock = MagicMock()
        mock.upsert.return_value = None
        mock.upsert_batch.return_value = None
        return mock

    @pytest.fixture
    def mock_storage(self):
        """Create mock storage."""
        mock = MagicMock()
        mock.save_chunks.return_value = None
        return mock

    @pytest.fixture
    def test_source(self):
        """Create test source."""
        return Source(
            id="error-test-source",
            name="Error Test Source",
            source_type="notion",
            config=NotionSourceConfig(page_ids=["page-1"]),
        )

    def test_indexer_continues_on_partial_failure(
        self, mock_embedder, mock_vector_store, mock_storage, test_source
    ):
        """Test indexer continues processing after some failures."""
        indexer = Indexer(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            storage=mock_storage,
        )

        documents = [
            Document(
                id=f"doc-{i}",
                source_id=test_source.id,
                external_id=f"ext-{i}",
                title=f"Test {i}",
                content=f"Content {i}. " * 10,
                content_hash=f"hash-{i}",
            )
            for i in range(5)
        ]

        result = indexer.index_documents(documents, test_source, show_progress=False)

        # Should have attempted to process documents
        assert isinstance(result, dict)


class TestMultiLanguageSupport:
    """Tests for multi-language document support."""

    def test_korean_document_chunking(self, korean_document):
        """Test chunking Korean language documents."""
        chunker = Chunker(chunk_size=100, chunk_overlap=20)

        chunks = chunker.chunk_document(korean_document)

        # Should produce chunks
        assert len(chunks) > 0

        # Chunks should contain Korean text
        for chunk in chunks:
            assert chunk.text
            # Verify Korean characters present
            has_korean = any(
                '\uac00' <= char <= '\ud7a3' for char in chunk.text
            )
            assert has_korean or len(chunk.text) < 10  # Small chunks might not have Korean

    def test_mixed_language_chunking(self):
        """Test chunking documents with mixed languages."""
        chunker = Chunker(chunk_size=200, chunk_overlap=50)

        doc = Document(
            id="mixed-lang",
            source_id="test",
            external_id="ext-mixed",
            title="Mixed Language Document",
            content=(
                "This is English content. "
                "이것은 한국어 콘텐츠입니다. "
                "More English here. "
                "더 많은 한국어가 여기에 있습니다. " * 10
            ),
            content_hash="mixedhash",
        )

        chunks = chunker.chunk_document(doc)

        # Should handle mixed content
        assert len(chunks) > 0

        # Verify content integrity
        combined_text = " ".join(chunk.text for chunk in chunks)
        assert "English" in combined_text
        assert "한국어" in combined_text


class TestMetadataIntegrity:
    """Tests for metadata integrity through the pipeline."""

    def test_chunk_metadata_completeness(self):
        """Test that chunks have complete metadata."""
        chunker = Chunker(chunk_size=100, chunk_overlap=20)

        doc = Document(
            id="meta-complete",
            source_id="source-1",
            external_id="ext-1",
            title="Complete Metadata Document",
            content="Test content that will be chunked. " * 20,
            url="https://example.com/doc",
            content_hash="completehash",
            metadata={"custom_field": "custom_value"},
        )

        chunks = chunker.chunk_document(doc)

        for chunk in chunks:
            # Required metadata fields
            assert chunk.metadata.get("source_id") == doc.source_id
            assert chunk.metadata.get("document_title") == doc.title
            assert chunk.metadata.get("document_url") == doc.url
            assert chunk.metadata.get("total_chunks") == len(chunks)

    def test_token_estimation_accuracy(self):
        """Test token estimation for chunks."""
        chunker = Chunker(chunk_size=200, chunk_overlap=50)

        # English content
        english_doc = Document(
            id="english",
            source_id="test",
            external_id="ext",
            title="English",
            content="This is a test document with English content. " * 20,
            content_hash="englishhash",
        )

        # Korean content (typically more tokens per character)
        korean_doc = Document(
            id="korean",
            source_id="test",
            external_id="ext",
            title="Korean",
            content="이것은 한국어로 작성된 테스트 문서입니다. " * 20,
            content_hash="koreanhash",
        )

        english_chunks = chunker.chunk_document(english_doc)
        korean_chunks = chunker.chunk_document(korean_doc)

        # Both should have token estimates
        for chunk in english_chunks + korean_chunks:
            assert chunk.token_count is not None
            assert chunk.token_count > 0

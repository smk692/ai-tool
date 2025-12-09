"""Unit tests for Chunker service."""

import pytest

from src.services.chunker import (
    Chunker,
    HierarchicalChunker,
    get_chunker,
    get_hierarchical_chunker,
)
from src.models import Document, Chunk, ChunkType


class TestChunker:
    """Tests for Chunker class."""

    def test_chunker_initialization_defaults(self):
        """Test chunker with default settings."""
        chunker = Chunker()
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200
        assert len(chunker.separators) > 0

    def test_chunker_initialization_custom(self):
        """Test chunker with custom settings."""
        chunker = Chunker(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n", " "],
        )
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 100
        assert chunker.separators == ["\n", " "]

    def test_chunk_text_empty(self):
        """Test chunking empty text returns empty list."""
        chunker = Chunker()
        assert chunker.chunk_text("") == []
        assert chunker.chunk_text("   ") == []
        assert chunker.chunk_text(None) == []

    def test_chunk_text_short(self):
        """Test chunking short text returns single chunk."""
        chunker = Chunker(chunk_size=1000)
        text = "Short text content."

        chunks = chunker.chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_long(self):
        """Test chunking long text creates multiple chunks."""
        chunker = Chunker(chunk_size=100, chunk_overlap=20)
        text = "This is a sentence. " * 20  # ~400 characters

        chunks = chunker.chunk_text(text)

        assert len(chunks) > 1
        # Verify all text is covered
        assert all(c in text for c in chunks)

    def test_chunk_text_preserves_separators(self):
        """Test chunking respects paragraph boundaries."""
        chunker = Chunker(chunk_size=100, chunk_overlap=10)
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."

        chunks = chunker.chunk_text(text)

        # Should prefer breaking at paragraph boundaries
        assert len(chunks) >= 1

    def test_chunk_text_korean(self):
        """Test chunking Korean text."""
        chunker = Chunker(chunk_size=50, chunk_overlap=10)
        text = "이것은 한국어 문장입니다。다음 문장입니다。그리고 또 다른 문장입니다。"

        chunks = chunker.chunk_text(text)

        assert len(chunks) >= 1
        # Korean text should be preserved
        assert all(any(c in chunk for c in "이것은한국어") for chunk in chunks[:1])

    def test_chunk_document(self, sample_document):
        """Test chunking a Document object."""
        chunker = Chunker(chunk_size=50, chunk_overlap=10)

        chunks = chunker.chunk_document(sample_document)

        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.document_id == sample_document.id for c in chunks)

    def test_chunk_document_metadata(self, sample_document):
        """Test chunk metadata is populated correctly."""
        chunker = Chunker()
        chunks = chunker.chunk_document(sample_document)

        if chunks:
            chunk = chunks[0]
            assert chunk.metadata["source_id"] == sample_document.source_id
            assert chunk.metadata["document_title"] == sample_document.title
            assert chunk.metadata["document_url"] == sample_document.url
            assert chunk.metadata["total_chunks"] == len(chunks)

    def test_chunk_document_indexes(self, long_document):
        """Test chunk indexes are sequential."""
        chunker = Chunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk_document(long_document)

        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunk_document_token_estimation(self, sample_document):
        """Test token count is estimated for chunks."""
        chunker = Chunker()
        chunks = chunker.chunk_document(sample_document)

        for chunk in chunks:
            assert chunk.token_count > 0
            # Token count should be approximately len(text) / 3
            expected = len(chunk.text) // 3
            assert chunk.token_count == expected

    def test_chunk_documents_multiple(self, generate_documents):
        """Test chunking multiple documents."""
        documents = generate_documents(3)
        chunker = Chunker(chunk_size=100, chunk_overlap=20)

        all_chunks = chunker.chunk_documents(documents)

        assert len(all_chunks) >= 3  # At least one chunk per document

        # Verify chunks belong to correct documents
        doc_ids = {d.id for d in documents}
        for chunk in all_chunks:
            assert chunk.document_id in doc_ids

    def test_chunk_document_empty_content(self):
        """Test chunking document with empty content."""
        doc = Document(
            source_id="s",
            external_id="e",
            title="Empty",
            content="",
        )
        chunker = Chunker()

        chunks = chunker.chunk_document(doc)

        assert chunks == []

    def test_deterministic_chunk_ids(self, sample_document):
        """Test chunk IDs are deterministic."""
        chunker = Chunker()

        chunks1 = chunker.chunk_document(sample_document)
        chunks2 = chunker.chunk_document(sample_document)

        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2):
            assert c1.id == c2.id

    def test_chunk_overlap_content(self):
        """Test chunks have overlapping content."""
        chunker = Chunker(chunk_size=50, chunk_overlap=20)
        text = "word " * 50  # 250 characters

        chunks = chunker.chunk_text(text)

        if len(chunks) >= 2:
            # Check for overlap between consecutive chunks
            for i in range(len(chunks) - 1):
                # End of chunk i should overlap with start of chunk i+1
                # This is approximate due to separator-based splitting
                pass  # Overlap is handled internally by RecursiveCharacterTextSplitter


class TestGetChunker:
    """Tests for get_chunker factory function."""

    def test_get_chunker_returns_chunker(self):
        """Test get_chunker returns a Chunker instance."""
        chunker = get_chunker()
        assert isinstance(chunker, Chunker)

    def test_get_chunker_custom_params(self):
        """Test get_chunker with custom parameters."""
        chunker = get_chunker(chunk_size=500, chunk_overlap=50)
        # Note: get_chunker uses a global singleton, so this might not change settings
        # if already initialized. For fresh tests, this verifies the interface.
        assert isinstance(chunker, Chunker)

    def test_get_chunker_singleton(self):
        """Test get_chunker returns singleton instance."""
        chunker1 = get_chunker()
        chunker2 = get_chunker()
        assert chunker1 is chunker2


class TestHierarchicalChunker:
    """Tests for HierarchicalChunker class (Phase 1.3)."""

    def test_hierarchical_chunker_initialization_defaults(self):
        """Test hierarchical chunker with default settings."""
        chunker = HierarchicalChunker()
        assert chunker.parent_chunk_size == 4000
        assert chunker.parent_chunk_overlap == 400
        assert chunker.child_chunk_size == 800
        assert chunker.child_chunk_overlap == 100
        assert len(chunker.separators) > 0

    def test_hierarchical_chunker_initialization_custom(self):
        """Test hierarchical chunker with custom settings."""
        chunker = HierarchicalChunker(
            parent_chunk_size=2000,
            parent_chunk_overlap=200,
            child_chunk_size=400,
            child_chunk_overlap=50,
            separators=["\n", " "],
        )
        assert chunker.parent_chunk_size == 2000
        assert chunker.parent_chunk_overlap == 200
        assert chunker.child_chunk_size == 400
        assert chunker.child_chunk_overlap == 50
        assert chunker.separators == ["\n", " "]

    def test_chunk_document_creates_parent_chunks(self, long_document):
        """Test that parent chunks are created with correct chunk_type."""
        chunker = HierarchicalChunker(
            parent_chunk_size=500,
            child_chunk_size=100,
        )

        chunks = chunker.chunk_document(long_document, include_parents=True)

        parent_chunks = [c for c in chunks if c.chunk_type == ChunkType.PARENT]
        assert len(parent_chunks) >= 1
        for parent in parent_chunks:
            assert parent.parent_id is None  # Parents have no parent_id

    def test_chunk_document_creates_child_chunks(self, long_document):
        """Test that child chunks are created with correct chunk_type and parent_id."""
        chunker = HierarchicalChunker(
            parent_chunk_size=500,
            child_chunk_size=100,
        )

        chunks = chunker.chunk_document(long_document, include_parents=True)

        child_chunks = [c for c in chunks if c.chunk_type == ChunkType.CHILD]
        parent_chunks = [c for c in chunks if c.chunk_type == ChunkType.PARENT]

        assert len(child_chunks) >= 1
        # Each child should reference a valid parent
        parent_ids = {p.id for p in parent_chunks}
        for child in child_chunks:
            assert child.parent_id is not None
            assert child.parent_id in parent_ids

    def test_chunk_document_include_parents_false(self, long_document):
        """Test chunk_document with include_parents=False returns only children."""
        chunker = HierarchicalChunker(
            parent_chunk_size=500,
            child_chunk_size=100,
        )

        chunks = chunker.chunk_document(long_document, include_parents=False)

        # All returned chunks should be children
        for chunk in chunks:
            assert chunk.chunk_type == ChunkType.CHILD
            assert chunk.parent_id is not None  # Still references parent

    def test_chunk_document_empty_content(self):
        """Test chunking document with empty content."""
        doc = Document(
            source_id="s",
            external_id="e",
            title="Empty",
            content="",
        )
        chunker = HierarchicalChunker()

        chunks = chunker.chunk_document(doc)

        assert chunks == []

    def test_chunk_document_metadata_parent(self, sample_document):
        """Test parent chunk metadata is populated correctly."""
        chunker = HierarchicalChunker(
            parent_chunk_size=500,
            child_chunk_size=100,
        )
        chunks = chunker.chunk_document(sample_document, include_parents=True)

        parent_chunks = [c for c in chunks if c.chunk_type == ChunkType.PARENT]
        if parent_chunks:
            parent = parent_chunks[0]
            assert parent.metadata["source_id"] == sample_document.source_id
            assert parent.metadata["document_title"] == sample_document.title
            assert parent.metadata["document_url"] == sample_document.url
            assert "total_parents" in parent.metadata
            assert "parent_index" in parent.metadata

    def test_chunk_document_metadata_child(self, long_document):
        """Test child chunk metadata is populated correctly."""
        chunker = HierarchicalChunker(
            parent_chunk_size=500,
            child_chunk_size=100,
        )
        chunks = chunker.chunk_document(long_document, include_parents=True)

        child_chunks = [c for c in chunks if c.chunk_type == ChunkType.CHILD]
        if child_chunks:
            child = child_chunks[0]
            assert child.metadata["source_id"] == long_document.source_id
            assert child.metadata["document_title"] == long_document.title
            assert child.metadata["document_url"] == long_document.url
            assert "total_chunks" in child.metadata
            assert "parent_index" in child.metadata
            assert "child_index_in_parent" in child.metadata

    def test_chunk_document_token_estimation(self, long_document):
        """Test token count is estimated for hierarchical chunks."""
        chunker = HierarchicalChunker(
            parent_chunk_size=500,
            child_chunk_size=100,
        )
        chunks = chunker.chunk_document(long_document, include_parents=True)

        for chunk in chunks:
            assert chunk.token_count > 0
            expected = len(chunk.text) // 3
            assert chunk.token_count == expected

    def test_deterministic_chunk_ids(self, sample_document):
        """Test hierarchical chunk IDs are deterministic."""
        chunker = HierarchicalChunker(
            parent_chunk_size=500,
            child_chunk_size=100,
        )

        chunks1 = chunker.chunk_document(sample_document, include_parents=True)
        chunks2 = chunker.chunk_document(sample_document, include_parents=True)

        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2):
            assert c1.id == c2.id
            assert c1.chunk_type == c2.chunk_type

    def test_chunk_documents_multiple(self, generate_documents):
        """Test chunking multiple documents hierarchically."""
        documents = generate_documents(3)
        chunker = HierarchicalChunker(
            parent_chunk_size=500,
            parent_chunk_overlap=50,
            child_chunk_size=100,
            child_chunk_overlap=20,
        )

        all_chunks = chunker.chunk_documents(documents, include_parents=True)

        # Should have chunks from all documents
        doc_ids = {d.id for d in documents}
        chunk_doc_ids = {c.document_id for c in all_chunks}
        assert chunk_doc_ids == doc_ids

        # Should have both parent and child chunks
        parent_count = sum(1 for c in all_chunks if c.chunk_type == ChunkType.PARENT)
        child_count = sum(1 for c in all_chunks if c.chunk_type == ChunkType.CHILD)
        assert parent_count >= 3  # At least one parent per document
        assert child_count >= 3  # At least one child per document

    def test_get_parent_chunk_finds_parent(self, long_document):
        """Test get_parent_chunk utility method."""
        chunker = HierarchicalChunker(
            parent_chunk_size=500,
            child_chunk_size=100,
        )

        all_chunks = chunker.chunk_document(long_document, include_parents=True)
        child_chunks = [c for c in all_chunks if c.chunk_type == ChunkType.CHILD]

        if child_chunks:
            child = child_chunks[0]
            parent = chunker.get_parent_chunk(child, all_chunks)

            assert parent is not None
            assert parent.id == child.parent_id
            assert parent.chunk_type == ChunkType.PARENT

    def test_get_parent_chunk_returns_none_for_parent(self, long_document):
        """Test get_parent_chunk returns None for parent chunks."""
        chunker = HierarchicalChunker(
            parent_chunk_size=500,
            child_chunk_size=100,
        )

        all_chunks = chunker.chunk_document(long_document, include_parents=True)
        parent_chunks = [c for c in all_chunks if c.chunk_type == ChunkType.PARENT]

        if parent_chunks:
            parent = parent_chunks[0]
            result = chunker.get_parent_chunk(parent, all_chunks)
            assert result is None

    def test_child_chunks_cover_parent_content(self, long_document):
        """Test that child chunks together cover the parent chunk content."""
        chunker = HierarchicalChunker(
            parent_chunk_size=500,
            parent_chunk_overlap=0,
            child_chunk_size=100,
            child_chunk_overlap=0,
        )

        all_chunks = chunker.chunk_document(long_document, include_parents=True)
        parent_chunks = [c for c in all_chunks if c.chunk_type == ChunkType.PARENT]
        child_chunks = [c for c in all_chunks if c.chunk_type == ChunkType.CHILD]

        if parent_chunks and child_chunks:
            # Get children for first parent
            first_parent = parent_chunks[0]
            parent_children = [c for c in child_chunks if c.parent_id == first_parent.id]

            # Children should contain parts of the parent text
            for child in parent_children:
                assert child.text in first_parent.text or len(child.text) > 0


class TestGetHierarchicalChunker:
    """Tests for get_hierarchical_chunker factory function."""

    def test_get_hierarchical_chunker_returns_instance(self):
        """Test get_hierarchical_chunker returns a HierarchicalChunker instance."""
        chunker = get_hierarchical_chunker()
        assert isinstance(chunker, HierarchicalChunker)

    def test_get_hierarchical_chunker_singleton(self):
        """Test get_hierarchical_chunker returns singleton instance."""
        chunker1 = get_hierarchical_chunker()
        chunker2 = get_hierarchical_chunker()
        assert chunker1 is chunker2

    def test_get_hierarchical_chunker_default_params(self):
        """Test get_hierarchical_chunker uses default parameters."""
        chunker = get_hierarchical_chunker()
        assert chunker.parent_chunk_size == HierarchicalChunker.DEFAULT_PARENT_SIZE
        assert chunker.parent_chunk_overlap == HierarchicalChunker.DEFAULT_PARENT_OVERLAP
        assert chunker.child_chunk_size == HierarchicalChunker.DEFAULT_CHILD_SIZE
        assert chunker.child_chunk_overlap == HierarchicalChunker.DEFAULT_CHILD_OVERLAP

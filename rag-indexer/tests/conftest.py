"""Pytest configuration and shared fixtures."""

import pytest
from datetime import datetime
from uuid import uuid4

from src.models import Document, Chunk, Source, NotionSourceConfig, SwaggerSourceConfig


# ==================== Document Fixtures ====================


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    return Document(
        id="doc-123",
        source_id="source-456",
        external_id="notion-page-789",
        title="Test Document",
        content="This is a test document content.\n\nIt has multiple paragraphs.\n\nAnd some more text.",
        url="https://notion.so/page/789",
        content_hash="abc123hash",
        metadata={"created_by": "user-1"},
    )


@pytest.fixture
def long_document():
    """Create a document with long content for chunking tests."""
    paragraphs = [
        f"This is paragraph {i}. " * 20
        for i in range(10)
    ]
    content = "\n\n".join(paragraphs)
    return Document(
        id="doc-long",
        source_id="source-456",
        external_id="notion-page-long",
        title="Long Test Document",
        content=content,
        url="https://notion.so/page/long",
        content_hash="longhash",
    )


@pytest.fixture
def korean_document():
    """Create a document with Korean content."""
    return Document(
        id="doc-korean",
        source_id="source-456",
        external_id="notion-page-korean",
        title="한국어 테스트 문서",
        content="이것은 한국어 테스트 문서입니다.\n\n여러 단락이 있습니다.\n\n더 많은 텍스트가 있습니다.",
        url="https://notion.so/page/korean",
        content_hash="koreanhash",
    )


# ==================== Chunk Fixtures ====================


@pytest.fixture
def sample_chunk():
    """Create a sample chunk for testing."""
    return Chunk(
        id="chunk-123",
        document_id="doc-123",
        chunk_index=0,
        text="This is a sample chunk text.",
        token_count=10,
        metadata={"source_id": "source-456"},
    )


@pytest.fixture
def chunk_with_embedding():
    """Create a chunk with embedding for testing."""
    return Chunk(
        id="chunk-emb",
        document_id="doc-123",
        chunk_index=0,
        text="Chunk with embedding.",
        token_count=5,
        embedding=[0.1] * 1024,  # 1024-dimension embedding
        metadata={"source_id": "source-456"},
    )


# ==================== Source Fixtures ====================


@pytest.fixture
def notion_source():
    """Create a Notion source for testing."""
    return Source(
        id="notion-source-1",
        name="Test Notion Workspace",
        source_type="notion",
        config=NotionSourceConfig(
            page_ids=["page-1", "page-2"],
            database_ids=["db-1"],
        ),
    )


@pytest.fixture
def swagger_source():
    """Create a Swagger source for testing."""
    return Source(
        id="swagger-source-1",
        name="Test API Docs",
        source_type="swagger",
        config=SwaggerSourceConfig(
            url="https://api.example.com/swagger.json",
        ),
    )


# ==================== Test Helpers ====================


@pytest.fixture
def generate_documents():
    """Factory fixture to generate multiple documents."""
    def _generate(count: int, prefix: str = "doc"):
        return [
            Document(
                id=f"{prefix}-{i}",
                source_id="source-test",
                external_id=f"external-{i}",
                title=f"Document {i}",
                content=f"Content for document {i}. " * 10,
                content_hash=f"hash-{i}",
            )
            for i in range(count)
        ]
    return _generate


@pytest.fixture
def generate_chunks():
    """Factory fixture to generate multiple chunks."""
    def _generate(document_id: str, count: int):
        return [
            Chunk(
                id=f"chunk-{document_id}-{i}",
                document_id=document_id,
                chunk_index=i,
                text=f"Chunk {i} text content.",
                token_count=5,
            )
            for i in range(count)
        ]
    return _generate

"""Unit tests for VectorStore."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import chromadb

from src.services.vector_store import VectorStore
from src.services.embeddings import HuggingFaceEmbedding
from src.utils.errors import VectorStoreError


class TestVectorStore:
    """Test suite for VectorStore class."""

    @pytest.fixture
    def mock_embedding_service(self):
        """Create mock embedding service."""
        mock = Mock(spec=HuggingFaceEmbedding)
        mock.get_embedding_dimension.return_value = 384
        mock.embed_text.return_value = [0.1] * 384
        mock.embed_texts.return_value = [[0.1] * 384, [0.2] * 384]
        return mock

    @pytest.fixture
    def temp_persist_dir(self, tmp_path):
        """Create temporary persist directory."""
        return str(tmp_path / "chroma_test")

    @pytest.fixture
    def vector_store(self, mock_embedding_service, temp_persist_dir):
        """Create VectorStore instance with mocks."""
        return VectorStore(
            embedding_service=mock_embedding_service,
            persist_directory=temp_persist_dir,
            collection_name="test_collection",
        )

    def test_initialization(self, mock_embedding_service, temp_persist_dir):
        """Test VectorStore initialization."""
        store = VectorStore(
            embedding_service=mock_embedding_service,
            persist_directory=temp_persist_dir,
            collection_name="test_collection",
        )

        assert store.embedding_service == mock_embedding_service
        assert str(store.persist_directory) == temp_persist_dir
        assert store.collection_name == "test_collection"
        assert Path(temp_persist_dir).exists()

    def test_initialization_default_values(self, mock_embedding_service):
        """Test VectorStore initialization with default values."""
        with patch("src.services.vector_store.settings") as mock_settings:
            mock_settings.chroma_persist_directory = "./data/chroma"
            mock_settings.chroma_collection_name = "default_collection"

            store = VectorStore(embedding_service=mock_embedding_service)

            assert store.collection_name == "default_collection"

    @patch("chromadb.PersistentClient")
    def test_client_property_lazy_initialization(
        self, mock_persistent_client, vector_store
    ):
        """Test ChromaDB client lazy initialization."""
        mock_client = Mock()
        mock_persistent_client.return_value = mock_client

        # First access initializes client
        client = vector_store.client
        assert client == mock_client
        mock_persistent_client.assert_called_once()

        # Second access returns cached client
        client2 = vector_store.client
        assert client2 == mock_client
        assert mock_persistent_client.call_count == 1

    @patch("chromadb.PersistentClient")
    def test_client_initialization_error(
        self, mock_persistent_client, vector_store
    ):
        """Test ChromaDB client initialization error handling."""
        mock_persistent_client.side_effect = Exception("ChromaDB connection failed")

        with pytest.raises(VectorStoreError) as exc_info:
            _ = vector_store.client

        assert "Failed to initialize ChromaDB client" in str(exc_info.value)

    @patch("chromadb.PersistentClient")
    def test_collection_property_lazy_initialization(
        self, mock_persistent_client, vector_store
    ):
        """Test ChromaDB collection lazy initialization."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_persistent_client.return_value = mock_client

        # First access initializes collection
        collection = vector_store.collection
        assert collection == mock_collection
        mock_client.get_or_create_collection.assert_called_once_with(
            name="test_collection",
            metadata={"hnsw:space": "cosine"},
        )

        # Second access returns cached collection
        collection2 = vector_store.collection
        assert collection2 == mock_collection
        assert mock_client.get_or_create_collection.call_count == 1

    @patch("chromadb.PersistentClient")
    def test_collection_initialization_error(
        self, mock_persistent_client, vector_store
    ):
        """Test ChromaDB collection initialization error handling."""
        mock_client = Mock()
        mock_client.get_or_create_collection.side_effect = Exception(
            "Collection creation failed"
        )
        mock_persistent_client.return_value = mock_client

        with pytest.raises(VectorStoreError) as exc_info:
            _ = vector_store.collection

        assert "Failed to access ChromaDB collection" in str(exc_info.value)

    @patch("chromadb.PersistentClient")
    def test_add_documents_success(self, mock_persistent_client, vector_store):
        """Test successful document addition."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_persistent_client.return_value = mock_client

        documents = ["Document 1", "Document 2"]
        metadatas = [{"source": "test1"}, {"source": "test2"}]
        ids = ["id1", "id2"]

        result = vector_store.add_documents(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

        assert result["success"] is True
        assert result["count"] == 2
        assert result["ids"] == ids
        mock_collection.add.assert_called_once()

    @patch("chromadb.PersistentClient")
    def test_add_documents_auto_generate_ids(
        self, mock_persistent_client, vector_store
    ):
        """Test document addition with auto-generated IDs."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_persistent_client.return_value = mock_client

        documents = ["Document 1", "Document 2"]

        result = vector_store.add_documents(documents=documents)

        assert result["success"] is True
        assert result["count"] == 2
        assert len(result["ids"]) == 2
        # Check that IDs are UUIDs (valid format)
        for doc_id in result["ids"]:
            assert isinstance(doc_id, str)
            assert len(doc_id) == 36  # UUID string length

    @patch("chromadb.PersistentClient")
    def test_add_documents_auto_generate_metadatas(
        self, mock_persistent_client, vector_store
    ):
        """Test document addition with auto-generated metadatas."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_persistent_client.return_value = mock_client

        documents = ["Document 1", "Document 2"]
        ids = ["id1", "id2"]

        result = vector_store.add_documents(documents=documents, ids=ids)

        assert result["success"] is True
        # Verify default metadatas were used
        call_args = mock_collection.add.call_args
        assert call_args[1]["metadatas"] == [
            {"source": "unknown"},
            {"source": "unknown"},
        ]

    @patch("chromadb.PersistentClient")
    def test_add_documents_length_mismatch(
        self, mock_persistent_client, vector_store
    ):
        """Test document addition with mismatched lengths."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_persistent_client.return_value = mock_client

        documents = ["Document 1", "Document 2"]
        metadatas = [{"source": "test1"}]  # Only 1 metadata

        with pytest.raises(VectorStoreError) as exc_info:
            vector_store.add_documents(
                documents=documents,
                metadatas=metadatas,
                ids=["id1", "id2"],
            )

        assert "Mismatched lengths" in str(exc_info.value)

    @patch("chromadb.PersistentClient")
    def test_add_documents_with_embeddings(
        self, mock_persistent_client, vector_store, mock_embedding_service
    ):
        """Test document addition with pre-computed embeddings."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_persistent_client.return_value = mock_client

        documents = ["Document 1", "Document 2"]
        embeddings = [[0.1] * 384, [0.2] * 384]

        result = vector_store.add_documents(
            documents=documents,
            embeddings=embeddings,
        )

        assert result["success"] is True
        # Verify embedding service was NOT called
        mock_embedding_service.embed_texts.assert_not_called()

    @patch("chromadb.PersistentClient")
    def test_add_documents_generate_embeddings(
        self, mock_persistent_client, vector_store, mock_embedding_service
    ):
        """Test document addition with auto-generated embeddings."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_persistent_client.return_value = mock_client

        documents = ["Document 1", "Document 2"]

        result = vector_store.add_documents(documents=documents)

        assert result["success"] is True
        # Verify embedding service WAS called
        mock_embedding_service.embed_texts.assert_called_once_with(documents)

    @patch("chromadb.PersistentClient")
    def test_add_documents_error(self, mock_persistent_client, vector_store):
        """Test document addition error handling."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.add.side_effect = Exception("ChromaDB add failed")
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_persistent_client.return_value = mock_client

        with pytest.raises(VectorStoreError) as exc_info:
            vector_store.add_documents(documents=["Document 1"])

        assert "Failed to add documents" in str(exc_info.value)

    @patch("chromadb.PersistentClient")
    def test_query_success(
        self, mock_persistent_client, vector_store, mock_embedding_service
    ):
        """Test successful vector query."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [["Doc 1", "Doc 2"]],
            "metadatas": [[{"source": "test1"}, {"source": "test2"}]],
            "distances": [[0.1, 0.2]],
            "ids": [["id1", "id2"]],
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_persistent_client.return_value = mock_client

        result = vector_store.query(query_text="test query", top_k=2)

        assert result["documents"] == ["Doc 1", "Doc 2"]
        assert result["metadatas"] == [{"source": "test1"}, {"source": "test2"}]
        assert result["distances"] == [0.1, 0.2]
        assert result["ids"] == ["id1", "id2"]
        mock_embedding_service.embed_text.assert_called_once_with("test query")

    @patch("chromadb.PersistentClient")
    def test_query_with_filter(
        self, mock_persistent_client, vector_store, mock_embedding_service
    ):
        """Test vector query with metadata filter."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [["Doc 1"]],
            "metadatas": [[{"source": "test1"}]],
            "distances": [[0.1]],
            "ids": [["id1"]],
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_persistent_client.return_value = mock_client

        filter_dict = {"source": "test1"}
        result = vector_store.query(
            query_text="test query", top_k=1, filter=filter_dict
        )

        assert len(result["documents"]) == 1
        # Verify filter was passed to ChromaDB
        call_args = mock_collection.query.call_args
        assert call_args[1]["where"] == filter_dict

    @patch("chromadb.PersistentClient")
    def test_query_error(
        self, mock_persistent_client, vector_store, mock_embedding_service
    ):
        """Test query error handling."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.query.side_effect = Exception("ChromaDB query failed")
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_persistent_client.return_value = mock_client

        with pytest.raises(VectorStoreError) as exc_info:
            vector_store.query(query_text="test query")

        assert "Vector search query failed" in str(exc_info.value)

    @patch("chromadb.PersistentClient")
    def test_get_collection_stats_success(
        self, mock_persistent_client, vector_store
    ):
        """Test getting collection statistics."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.count.return_value = 42
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_persistent_client.return_value = mock_client

        stats = vector_store.get_collection_stats()

        assert stats["collection_name"] == "test_collection"
        assert stats["document_count"] == 42
        assert "persist_directory" in stats

    @patch("chromadb.PersistentClient")
    def test_get_collection_stats_error(
        self, mock_persistent_client, vector_store
    ):
        """Test collection stats error handling."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.count.side_effect = Exception("ChromaDB count failed")
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_persistent_client.return_value = mock_client

        with pytest.raises(VectorStoreError) as exc_info:
            vector_store.get_collection_stats()

        assert "Failed to retrieve collection statistics" in str(exc_info.value)

    @patch("chromadb.PersistentClient")
    def test_delete_collection_success(
        self, mock_persistent_client, vector_store
    ):
        """Test successful collection deletion."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_persistent_client.return_value = mock_client

        # Initialize collection first
        _ = vector_store.collection

        # Delete collection
        vector_store.delete_collection()

        mock_client.delete_collection.assert_called_once_with(
            name="test_collection"
        )
        assert vector_store._collection is None

    @patch("chromadb.PersistentClient")
    def test_delete_collection_error(
        self, mock_persistent_client, vector_store
    ):
        """Test collection deletion error handling."""
        mock_client = Mock()
        mock_client.delete_collection.side_effect = Exception(
            "ChromaDB delete failed"
        )
        mock_persistent_client.return_value = mock_client

        with pytest.raises(VectorStoreError) as exc_info:
            vector_store.delete_collection()

        assert "Failed to delete collection" in str(exc_info.value)

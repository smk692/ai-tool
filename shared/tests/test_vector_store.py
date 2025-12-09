"""Tests for vector_store module."""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from shared.vector_store import VectorStore, get_vector_store


class TestVectorStore:
    """Tests for VectorStore class."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create a mock QdrantClient."""
        with patch("shared.vector_store.QdrantClient") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance

    def test_initialization_with_defaults(self):
        """Test VectorStore initializes with default values."""
        store = VectorStore()
        assert store.host == "localhost"
        assert store.port == 6333
        assert store.collection_name == "rag_documents"
        assert store._client is None

    def test_initialization_with_custom_values(self):
        """Test VectorStore initializes with custom values."""
        store = VectorStore(
            host="custom-host",
            port=6334,
            api_key="test-key",
            collection_name="custom_collection",
        )
        assert store.host == "custom-host"
        assert store.port == 6334
        assert store.collection_name == "custom_collection"
        assert store._api_key == "test-key"

    def test_client_lazy_loading(self, mock_qdrant_client):
        """Test Qdrant client is lazy-loaded."""
        with patch("shared.vector_store.QdrantClient") as mock_constructor:
            mock_constructor.return_value = mock_qdrant_client
            store = VectorStore()
            assert store._client is None

            # Access client property
            _ = store.client
            mock_constructor.assert_called_once()

    def test_ensure_collection_creates_new(self, mock_qdrant_client):
        """Test ensure_collection creates collection when not exists."""
        mock_qdrant_client.get_collections.return_value.collections = []

        with patch("shared.vector_store.QdrantClient", return_value=mock_qdrant_client):
            store = VectorStore()
            result = store.ensure_collection()

            assert result is True
            mock_qdrant_client.create_collection.assert_called_once()

    def test_ensure_collection_exists(self, mock_qdrant_client):
        """Test ensure_collection returns False when collection exists."""
        mock_collection = MagicMock()
        mock_collection.name = "rag_documents"
        mock_qdrant_client.get_collections.return_value.collections = [mock_collection]

        with patch("shared.vector_store.QdrantClient", return_value=mock_qdrant_client):
            store = VectorStore()
            result = store.ensure_collection()

            assert result is False
            mock_qdrant_client.create_collection.assert_not_called()

    def test_upsert_empty_list(self, mock_qdrant_client):
        """Test upsert with empty list returns 0."""
        with patch("shared.vector_store.QdrantClient", return_value=mock_qdrant_client):
            store = VectorStore()
            result = store.upsert([])
            assert result == 0

    def test_upsert_single_point(self, mock_qdrant_client):
        """Test upsert single point."""
        with patch("shared.vector_store.QdrantClient", return_value=mock_qdrant_client):
            store = VectorStore()
            points = [
                {
                    "id": str(uuid4()),
                    "vector": [0.1] * 1024,
                    "payload": {"text": "test"},
                }
            ]
            result = store.upsert(points)

            assert result == 1
            mock_qdrant_client.upsert.assert_called_once()

    def test_upsert_batched(self, mock_qdrant_client):
        """Test upsert batches large point lists."""
        with patch("shared.vector_store.QdrantClient", return_value=mock_qdrant_client):
            store = VectorStore()
            # Create 150 points (should be split into 2 batches with batch_size=100)
            points = [
                {"id": str(uuid4()), "vector": [0.1] * 1024, "payload": {}}
                for _ in range(150)
            ]
            result = store.upsert(points, batch_size=100)

            assert result == 150
            assert mock_qdrant_client.upsert.call_count == 2

    def test_delete_empty_list(self, mock_qdrant_client):
        """Test delete with empty list returns 0."""
        with patch("shared.vector_store.QdrantClient", return_value=mock_qdrant_client):
            store = VectorStore()
            result = store.delete([])
            assert result == 0

    def test_delete_by_ids(self, mock_qdrant_client):
        """Test delete by point IDs."""
        with patch("shared.vector_store.QdrantClient", return_value=mock_qdrant_client):
            store = VectorStore()
            point_ids = [str(uuid4()), str(uuid4())]
            result = store.delete(point_ids)

            assert result == 2
            mock_qdrant_client.delete.assert_called_once()

    def test_delete_by_filter(self, mock_qdrant_client):
        """Test delete by filter condition."""
        with patch("shared.vector_store.QdrantClient", return_value=mock_qdrant_client):
            store = VectorStore()
            result = store.delete_by_filter("source_id", "test-source")

            assert result is True
            mock_qdrant_client.delete.assert_called_once()

    def test_search_basic(self, mock_qdrant_client):
        """Test basic search returns formatted results."""
        mock_result = MagicMock()
        mock_result.id = str(uuid4())
        mock_result.score = 0.95
        mock_result.payload = {"text": "test result"}
        mock_qdrant_client.search.return_value = [mock_result]

        with patch("shared.vector_store.QdrantClient", return_value=mock_qdrant_client):
            store = VectorStore()
            results = store.search([0.1] * 1024, limit=5)

            assert len(results) == 1
            assert results[0]["score"] == 0.95
            assert results[0]["payload"]["text"] == "test result"

    def test_search_with_filter(self, mock_qdrant_client):
        """Test search with filter conditions."""
        mock_qdrant_client.search.return_value = []

        with patch("shared.vector_store.QdrantClient", return_value=mock_qdrant_client):
            store = VectorStore()
            store.search(
                [0.1] * 1024,
                limit=10,
                filter_conditions={"source_type": "notion"},
            )

            call_kwargs = mock_qdrant_client.search.call_args[1]
            assert call_kwargs["query_filter"] is not None

    def test_search_with_score_threshold(self, mock_qdrant_client):
        """Test search with score threshold."""
        mock_qdrant_client.search.return_value = []

        with patch("shared.vector_store.QdrantClient", return_value=mock_qdrant_client):
            store = VectorStore()
            store.search([0.1] * 1024, score_threshold=0.7)

            call_kwargs = mock_qdrant_client.search.call_args[1]
            assert call_kwargs["score_threshold"] == 0.7

    def test_get_point_found(self, mock_qdrant_client):
        """Test get_point returns point data when found."""
        mock_point = MagicMock()
        mock_point.id = "test-id"
        mock_point.vector = [0.1] * 1024
        mock_point.payload = {"text": "test"}
        mock_qdrant_client.retrieve.return_value = [mock_point]

        with patch("shared.vector_store.QdrantClient", return_value=mock_qdrant_client):
            store = VectorStore()
            result = store.get_point("test-id")

            assert result is not None
            assert result["id"] == "test-id"
            assert result["payload"]["text"] == "test"

    def test_get_point_not_found(self, mock_qdrant_client):
        """Test get_point returns None when not found."""
        mock_qdrant_client.retrieve.return_value = []

        with patch("shared.vector_store.QdrantClient", return_value=mock_qdrant_client):
            store = VectorStore()
            result = store.get_point("nonexistent-id")

            assert result is None

    def test_get_point_exception(self, mock_qdrant_client):
        """Test get_point handles exceptions gracefully."""
        mock_qdrant_client.retrieve.side_effect = Exception("Connection error")

        with patch("shared.vector_store.QdrantClient", return_value=mock_qdrant_client):
            store = VectorStore()
            result = store.get_point("test-id")

            assert result is None

    def test_point_exists_true(self, mock_qdrant_client):
        """Test point_exists returns True when point exists."""
        mock_point = MagicMock()
        mock_point.id = "test-id"
        mock_point.vector = [0.1] * 1024
        mock_point.payload = {}
        mock_qdrant_client.retrieve.return_value = [mock_point]

        with patch("shared.vector_store.QdrantClient", return_value=mock_qdrant_client):
            store = VectorStore()
            result = store.point_exists("test-id")

            assert result is True

    def test_point_exists_false(self, mock_qdrant_client):
        """Test point_exists returns False when point doesn't exist."""
        mock_qdrant_client.retrieve.return_value = []

        with patch("shared.vector_store.QdrantClient", return_value=mock_qdrant_client):
            store = VectorStore()
            result = store.point_exists("nonexistent-id")

            assert result is False

    def test_count_without_filter(self, mock_qdrant_client):
        """Test count without filter returns total count."""
        mock_result = MagicMock()
        mock_result.count = 100
        mock_qdrant_client.count.return_value = mock_result

        with patch("shared.vector_store.QdrantClient", return_value=mock_qdrant_client):
            store = VectorStore()
            result = store.count()

            assert result == 100
            call_kwargs = mock_qdrant_client.count.call_args[1]
            assert call_kwargs["count_filter"] is None

    def test_count_with_filter(self, mock_qdrant_client):
        """Test count with filter returns filtered count."""
        mock_result = MagicMock()
        mock_result.count = 25
        mock_qdrant_client.count.return_value = mock_result

        with patch("shared.vector_store.QdrantClient", return_value=mock_qdrant_client):
            store = VectorStore()
            result = store.count(filter_conditions={"source_type": "swagger"})

            assert result == 25
            call_kwargs = mock_qdrant_client.count.call_args[1]
            assert call_kwargs["count_filter"] is not None

    def test_get_collection_info(self, mock_qdrant_client):
        """Test get_collection_info returns formatted info."""
        mock_info = MagicMock()
        mock_info.vectors_count = 1000
        mock_info.points_count = 1000
        mock_info.status = "green"
        mock_qdrant_client.get_collection.return_value = mock_info

        with patch("shared.vector_store.QdrantClient", return_value=mock_qdrant_client):
            store = VectorStore()
            result = store.get_collection_info()

            assert result["name"] == "rag_documents"
            assert result["vectors_count"] == 1000
            assert result["points_count"] == 1000

    def test_normalize_id_string(self):
        """Test _normalize_id handles string IDs."""
        store = VectorStore()
        result = store._normalize_id("test-id")
        assert result == "test-id"

    def test_normalize_id_uuid(self):
        """Test _normalize_id handles UUID objects."""
        store = VectorStore()
        test_uuid = uuid4()
        result = store._normalize_id(test_uuid)
        assert result == str(test_uuid)


class TestGetVectorStore:
    """Tests for get_vector_store function."""

    def test_get_vector_store_returns_instance(self):
        """Test get_vector_store returns VectorStore instance."""
        store = get_vector_store()
        assert isinstance(store, VectorStore)

    def test_get_vector_store_cached_default(self):
        """Test get_vector_store returns cached instance for default params."""
        import shared.vector_store

        # Reset the global cache
        shared.vector_store._default_store = None

        store1 = get_vector_store()
        store2 = get_vector_store()
        assert store1 is store2

    def test_get_vector_store_custom_params_not_cached(self):
        """Test get_vector_store returns new instance for custom params."""
        store1 = get_vector_store(host="custom-host")
        store2 = get_vector_store(host="custom-host")
        # Custom params should create new instances
        assert store1 is not store2

    def test_get_vector_store_with_custom_collection(self):
        """Test get_vector_store with custom collection name."""
        store = get_vector_store(collection_name="custom_collection")
        assert store.collection_name == "custom_collection"

    def test_get_vector_store_with_api_key(self):
        """Test get_vector_store with API key."""
        store = get_vector_store(api_key="test-api-key")
        assert store._api_key == "test-api-key"


class TestVectorStoreEdgeCases:
    """Edge case tests for VectorStore."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create a mock QdrantClient."""
        with patch("shared.vector_store.QdrantClient") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance

    def test_search_empty_results(self, mock_qdrant_client):
        """Test search handles empty results."""
        mock_qdrant_client.search.return_value = []

        with patch("shared.vector_store.QdrantClient", return_value=mock_qdrant_client):
            store = VectorStore()
            results = store.search([0.1] * 1024)

            assert results == []

    def test_search_result_with_none_payload(self, mock_qdrant_client):
        """Test search handles results with None payload."""
        mock_result = MagicMock()
        mock_result.id = "test-id"
        mock_result.score = 0.9
        mock_result.payload = None
        mock_qdrant_client.search.return_value = [mock_result]

        with patch("shared.vector_store.QdrantClient", return_value=mock_qdrant_client):
            store = VectorStore()
            results = store.search([0.1] * 1024)

            assert results[0]["payload"] == {}

    def test_upsert_point_without_payload(self, mock_qdrant_client):
        """Test upsert handles points without payload."""
        with patch("shared.vector_store.QdrantClient", return_value=mock_qdrant_client):
            store = VectorStore()
            points = [
                {"id": str(uuid4()), "vector": [0.1] * 1024}
                # No payload field
            ]
            result = store.upsert(points)

            assert result == 1

    def test_get_point_with_none_payload(self, mock_qdrant_client):
        """Test get_point handles point with None payload."""
        mock_point = MagicMock()
        mock_point.id = "test-id"
        mock_point.vector = [0.1] * 1024
        mock_point.payload = None
        mock_qdrant_client.retrieve.return_value = [mock_point]

        with patch("shared.vector_store.QdrantClient", return_value=mock_qdrant_client):
            store = VectorStore()
            result = store.get_point("test-id")

            assert result["payload"] == {}

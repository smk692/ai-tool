"""Smoke tests for mock fixtures to validate mock infrastructure.

These tests ensure that mock implementations behave correctly and
provide the expected interfaces for use in other test modules.
"""

import numpy as np
import pytest


class TestMockSentenceTransformer:
    """Validate MockSentenceTransformer mock implementation."""

    def test_mock_initialization(self, mock_embedding_model):
        """Test: Mock model initializes with correct defaults."""
        assert mock_embedding_model is not None
        assert mock_embedding_model.model_name == "mock-model"
        assert mock_embedding_model.dimension == 384
        assert mock_embedding_model.device == "cpu"

    def test_mock_encode_single_text(self, mock_embedding_model):
        """Test: Generate embedding for single text."""
        text = "Hello world"
        embedding = mock_embedding_model.encode(text)

        assert embedding.shape == (1, 384)
        assert embedding.dtype == np.float32

    def test_mock_encode_batch(self, mock_embedding_model):
        """Test: Generate embeddings for batch of texts."""
        texts = ["Hello", "World", "Test"]
        embeddings = mock_embedding_model.encode(texts)

        assert embeddings.shape == (3, 384)
        assert embeddings.dtype == np.float32

    def test_mock_encode_deterministic(self, mock_embedding_model):
        """Test: Same text produces same embedding (deterministic)."""
        text = "Test determinism"

        embedding1 = mock_embedding_model.encode(text)
        embedding2 = mock_embedding_model.encode(text)

        np.testing.assert_array_equal(embedding1, embedding2)

    def test_mock_encode_different_texts(self, mock_embedding_model):
        """Test: Different texts produce different embeddings."""
        text1 = "First text"
        text2 = "Second text"

        embedding1 = mock_embedding_model.encode(text1)
        embedding2 = mock_embedding_model.encode(text2)

        # Embeddings should not be identical
        assert not np.array_equal(embedding1, embedding2)

    def test_mock_encode_empty_input_raises(self, mock_embedding_model):
        """Test: Empty input raises ValueError."""
        with pytest.raises(ValueError, match="Empty input"):
            mock_embedding_model.encode([])

    def test_mock_encode_empty_string_raises(self, mock_embedding_model):
        """Test: Empty string in list raises ValueError."""
        with pytest.raises(ValueError, match="Empty input"):
            mock_embedding_model.encode([""])

    def test_mock_device_property(self, mock_embedding_model):
        """Test: Device property returns correct value."""
        assert mock_embedding_model.device == "cpu"

    def test_mock_to_device(self, mock_embedding_model):
        """Test: Change device with to() method."""
        model = mock_embedding_model.to("cuda")

        assert model.device == "cuda"
        assert model is mock_embedding_model  # Method chaining

    def test_mock_get_dimension(self, mock_embedding_model):
        """Test: Get embedding dimension."""
        dim = mock_embedding_model.get_sentence_embedding_dimension()

        assert dim == 384

    def test_mock_encode_normalize(self, mock_embedding_model):
        """Test: Normalize embeddings to unit length."""
        text = "Test normalization"
        embedding = mock_embedding_model.encode(
            text, normalize_embeddings=True
        )

        # Check unit length (L2 norm ≈ 1.0)
        norm = np.linalg.norm(embedding)
        np.testing.assert_almost_equal(norm, 1.0, decimal=5)


class TestMockChromaClient:
    """Validate MockChromaClient and MockCollection implementation."""

    def test_mock_client_initialization(self, mock_vector_store):
        """Test: Mock client initializes correctly."""
        assert mock_vector_store is not None

    def test_mock_create_collection(self, mock_vector_store):
        """Test: Create new collection."""
        collection = mock_vector_store.create_collection("test_collection")

        assert collection is not None
        assert collection.name == "test_collection"
        assert collection.count() == 0

    def test_mock_get_or_create_collection(self, mock_vector_store):
        """Test: Get existing or create new collection."""
        # First call creates
        collection1 = mock_vector_store.get_or_create_collection("test")

        # Second call retrieves existing
        collection2 = mock_vector_store.get_or_create_collection("test")

        assert collection1 is collection2

    def test_mock_collection_add_single(self, mock_vector_store):
        """Test: Add single document to collection."""
        collection = mock_vector_store.create_collection("test")
        embedding = [0.1] * 384

        collection.add(
            ids="doc1",
            embeddings=embedding,
            metadatas={"key": "value"},
            documents="Test document"
        )

        assert collection.count() == 1

    def test_mock_collection_add_batch(self, mock_vector_store):
        """Test: Add batch of documents to collection."""
        collection = mock_vector_store.create_collection("test")
        embeddings = [[0.1] * 384, [0.2] * 384, [0.3] * 384]

        collection.add(
            ids=["doc1", "doc2", "doc3"],
            embeddings=embeddings,
            metadatas=[{"id": 1}, {"id": 2}, {"id": 3}],
            documents=["Doc 1", "Doc 2", "Doc 3"]
        )

        assert collection.count() == 3

    def test_mock_collection_add_validation(self, mock_vector_store):
        """Test: Add validates list lengths match."""
        collection = mock_vector_store.create_collection("test")

        # Mismatched lengths should raise ValueError
        with pytest.raises(ValueError, match="Embedding count"):
            collection.add(
                ids=["doc1", "doc2"],
                embeddings=[[0.1] * 384]  # Only 1 embedding for 2 ids
            )

    def test_mock_collection_query_empty_raises(self, mock_vector_store):
        """Test: Query empty collection raises ValueError."""
        collection = mock_vector_store.create_collection("test")

        with pytest.raises(ValueError, match="Cannot query empty collection"):
            collection.query(query_embeddings=[0.1] * 384)

    def test_mock_collection_query_single(self, mock_vector_store):
        """Test: Query returns similar documents."""
        collection = mock_vector_store.create_collection("test")

        # Add documents
        embeddings = [[0.1] * 384, [0.2] * 384, [0.9] * 384]
        collection.add(
            ids=["doc1", "doc2", "doc3"],
            embeddings=embeddings,
            documents=["Doc 1", "Doc 2", "Doc 3"]
        )

        # Query with vector similar to doc1
        results = collection.query(
            query_embeddings=[0.1] * 384,
            n_results=2
        )

        assert len(results["ids"]) == 1  # Single query
        assert len(results["ids"][0]) == 2  # Top 2 results
        assert results["ids"][0][0] == "doc1"  # doc1 should be closest

    def test_mock_collection_query_batch(self, mock_vector_store):
        """Test: Batch query returns results for each query."""
        collection = mock_vector_store.create_collection("test")

        # Add documents
        embeddings = [[0.1] * 384, [0.5] * 384, [0.9] * 384]
        collection.add(
            ids=["doc1", "doc2", "doc3"],
            embeddings=embeddings
        )

        # Batch query
        results = collection.query(
            query_embeddings=[[0.1] * 384, [0.9] * 384],
            n_results=1
        )

        assert len(results["ids"]) == 2  # Two queries
        assert results["ids"][0][0] == "doc1"  # First query → doc1
        assert results["ids"][1][0] == "doc3"  # Second query → doc3

    def test_mock_collection_get_by_ids(self, mock_vector_store):
        """Test: Get documents by IDs."""
        collection = mock_vector_store.create_collection("test")

        # Add documents
        collection.add(
            ids=["doc1", "doc2"],
            embeddings=[[0.1] * 384, [0.2] * 384],
            documents=["Doc 1", "Doc 2"]
        )

        # Get specific IDs
        results = collection.get(ids=["doc1"])

        assert results["ids"] == ["doc1"]
        assert results["documents"] == ["Doc 1"]

    def test_mock_collection_get_pagination(self, mock_vector_store):
        """Test: Get with limit and offset."""
        collection = mock_vector_store.create_collection("test")

        # Add documents
        collection.add(
            ids=["doc1", "doc2", "doc3"],
            embeddings=[[0.1] * 384, [0.2] * 384, [0.3] * 384]
        )

        # Get with pagination
        results = collection.get(limit=2, offset=1)

        assert len(results["ids"]) == 2
        assert results["ids"] == ["doc2", "doc3"]

    def test_mock_collection_delete_specific(self, mock_vector_store):
        """Test: Delete specific documents by IDs."""
        collection = mock_vector_store.create_collection("test")

        # Add and delete
        collection.add(
            ids=["doc1", "doc2"],
            embeddings=[[0.1] * 384, [0.2] * 384]
        )

        collection.delete(ids=["doc1"])

        assert collection.count() == 1

    def test_mock_collection_delete_all(self, mock_vector_store):
        """Test: Delete all documents."""
        collection = mock_vector_store.create_collection("test")

        # Add and delete all
        collection.add(
            ids=["doc1", "doc2"],
            embeddings=[[0.1] * 384, [0.2] * 384]
        )

        collection.delete()  # No IDs = delete all

        assert collection.count() == 0

    def test_mock_client_list_collections(self, mock_vector_store):
        """Test: List all collections."""
        mock_vector_store.create_collection("col1")
        mock_vector_store.create_collection("col2")

        collections = mock_vector_store.list_collections()

        assert len(collections) == 2
        assert any(c.name == "col1" for c in collections)
        assert any(c.name == "col2" for c in collections)

    def test_mock_client_delete_collection(self, mock_vector_store):
        """Test: Delete collection."""
        mock_vector_store.create_collection("test")
        mock_vector_store.delete_collection("test")

        collections = mock_vector_store.list_collections()
        assert len(collections) == 0

    def test_mock_client_reset(self, mock_vector_store):
        """Test: Reset client deletes all collections."""
        mock_vector_store.create_collection("col1")
        mock_vector_store.create_collection("col2")

        mock_vector_store.reset()

        assert len(mock_vector_store.list_collections()) == 0

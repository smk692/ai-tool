"""Tests for embedding module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from shared.embedding import EmbeddingModel, get_embedding_model


class TestEmbeddingModel:
    """Tests for EmbeddingModel class."""

    @pytest.fixture
    def mock_sentence_transformer(self):
        """Create a mock SentenceTransformer."""
        with patch("shared.embedding.SentenceTransformer") as mock:
            mock_instance = MagicMock()
            mock_instance.get_sentence_embedding_dimension.return_value = 1024
            mock_instance.encode.return_value = np.array([[0.1] * 1024])
            mock.return_value = mock_instance
            yield mock

    def test_initialization_with_defaults(self):
        """Test EmbeddingModel initializes with default values."""
        model = EmbeddingModel()
        assert model.model_name == "intfloat/multilingual-e5-large-instruct"
        assert model._model is None
        assert model._device is None

    def test_initialization_with_custom_values(self):
        """Test EmbeddingModel initializes with custom values."""
        model = EmbeddingModel(model_name="custom-model", device="cpu")
        assert model.model_name == "custom-model"
        assert model._device == "cpu"

    def test_model_lazy_loading(self, mock_sentence_transformer):
        """Test model is lazy-loaded on first access."""
        model = EmbeddingModel()
        assert model._model is None

        # Access the model property
        _ = model.model
        mock_sentence_transformer.assert_called_once()
        assert model._model is not None

    def test_dimension_property(self, mock_sentence_transformer):
        """Test dimension property returns correct value."""
        model = EmbeddingModel()
        dimension = model.dimension
        assert dimension == 1024

    def test_embed_documents_empty_list(self, mock_sentence_transformer):
        """Test embed_documents with empty list returns empty list."""
        model = EmbeddingModel()
        result = model.embed_documents([])
        assert result == []

    def test_embed_documents_adds_passage_prefix(self, mock_sentence_transformer):
        """Test embed_documents adds 'passage: ' prefix to texts."""
        model = EmbeddingModel()
        texts = ["test document"]
        model.embed_documents(texts)

        # Check the encode was called with prefixed texts
        call_args = mock_sentence_transformer.return_value.encode.call_args
        assert call_args[0][0] == ["passage: test document"]

    def test_embed_documents_returns_list(self, mock_sentence_transformer):
        """Test embed_documents returns list of embeddings."""
        mock_sentence_transformer.return_value.encode.return_value = np.array(
            [[0.1] * 1024, [0.2] * 1024]
        )
        model = EmbeddingModel()
        result = model.embed_documents(["doc1", "doc2"])
        assert len(result) == 2
        assert len(result[0]) == 1024

    def test_embed_documents_batch_size(self, mock_sentence_transformer):
        """Test embed_documents passes batch_size parameter."""
        model = EmbeddingModel()
        model.embed_documents(["test"], batch_size=64)

        call_kwargs = mock_sentence_transformer.return_value.encode.call_args[1]
        assert call_kwargs["batch_size"] == 64

    def test_embed_query_adds_query_prefix(self, mock_sentence_transformer):
        """Test embed_query adds 'query: ' prefix."""
        mock_sentence_transformer.return_value.encode.return_value = np.array([0.1] * 1024)
        model = EmbeddingModel()
        model.embed_query("test query")

        call_args = mock_sentence_transformer.return_value.encode.call_args
        assert call_args[0][0] == "query: test query"

    def test_embed_query_returns_list(self, mock_sentence_transformer):
        """Test embed_query returns embedding as list."""
        mock_sentence_transformer.return_value.encode.return_value = np.array([0.1] * 1024)
        model = EmbeddingModel()
        result = model.embed_query("test query")
        assert isinstance(result, list)
        assert len(result) == 1024

    def test_compute_similarity_identical_vectors(self):
        """Test compute_similarity returns 1.0 for identical vectors."""
        model = EmbeddingModel()
        vec = [0.1, 0.2, 0.3]
        similarity = model.compute_similarity(vec, vec)
        assert abs(similarity - 1.0) < 0.0001

    def test_compute_similarity_orthogonal_vectors(self):
        """Test compute_similarity returns ~0 for orthogonal vectors."""
        model = EmbeddingModel()
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = model.compute_similarity(vec1, vec2)
        assert abs(similarity) < 0.0001

    def test_compute_similarity_zero_vector(self):
        """Test compute_similarity handles zero vectors."""
        model = EmbeddingModel()
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        similarity = model.compute_similarity(vec1, vec2)
        assert similarity == 0.0

    def test_compute_similarity_negative_correlation(self):
        """Test compute_similarity returns negative for opposite vectors."""
        model = EmbeddingModel()
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        similarity = model.compute_similarity(vec1, vec2)
        assert abs(similarity - (-1.0)) < 0.0001


class TestGetEmbeddingModel:
    """Tests for get_embedding_model function."""

    def test_get_embedding_model_returns_instance(self):
        """Test get_embedding_model returns EmbeddingModel instance."""
        model = get_embedding_model()
        assert isinstance(model, EmbeddingModel)

    def test_get_embedding_model_cached_default(self):
        """Test get_embedding_model returns cached instance for default params."""
        import shared.embedding

        # Reset the global cache
        shared.embedding._default_model = None

        model1 = get_embedding_model()
        model2 = get_embedding_model()
        assert model1 is model2

    def test_get_embedding_model_custom_params_not_cached(self):
        """Test get_embedding_model returns new instance for custom params."""
        model1 = get_embedding_model(model_name="custom-model")
        model2 = get_embedding_model(model_name="custom-model")
        # Custom params should create new instances
        assert model1 is not model2

    def test_get_embedding_model_with_device(self):
        """Test get_embedding_model with custom device."""
        model = get_embedding_model(device="cpu")
        assert model._device == "cpu"


class TestEmbeddingModelIntegration:
    """Integration-style tests for embedding model (using mocks)."""

    @pytest.fixture
    def mock_model_for_korean(self):
        """Create mock for Korean text tests."""
        with patch("shared.embedding.SentenceTransformer") as mock:
            mock_instance = MagicMock()
            mock_instance.get_sentence_embedding_dimension.return_value = 1024
            # Korean text should get slightly different but similar embeddings
            def encode_side_effect(texts, **kwargs):
                if isinstance(texts, str):
                    return np.array([0.15] * 1024 if "한글" in texts else [0.1] * 1024)
                # Return array with correct number of embeddings
                return np.array([
                    [0.15] * 1024 if "한글" in t else [0.1] * 1024
                    for t in texts
                ])
            mock_instance.encode.side_effect = encode_side_effect
            mock.return_value = mock_instance
            yield mock

    def test_korean_text_embedding(self, mock_model_for_korean):
        """Test embedding Korean text works correctly."""
        model = EmbeddingModel()
        korean_text = "한글 테스트 문서입니다."
        result = model.embed_documents([korean_text])

        assert len(result) == 1
        assert len(result[0]) == 1024

    def test_mixed_language_embedding(self, mock_model_for_korean):
        """Test embedding mixed Korean/English text."""
        model = EmbeddingModel()
        mixed_texts = ["This is English", "한글 문서", "Mixed 혼합 text"]
        result = model.embed_documents(mixed_texts)

        assert len(result) == 3
        for embedding in result:
            assert len(embedding) == 1024

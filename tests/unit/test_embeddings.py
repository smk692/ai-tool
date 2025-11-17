"""
Unit Tests for HuggingFaceEmbedding Service

Tests Korean language support, multilingual capabilities, and core functionality
of the paraphrase-multilingual-MiniLM-L12-v2 embedding model.
"""

import pytest
import numpy as np

from src.models.embedding import EmbeddingConfiguration, DeviceType
from src.services.embeddings import HuggingFaceEmbedding


@pytest.fixture
def embedding_service():
    """Create HuggingFaceEmbedding service instance."""
    config = EmbeddingConfiguration()
    return HuggingFaceEmbedding(config)


class TestEmbeddingInitialization:
    """Test embedding service initialization and configuration."""

    def test_initialization_default_config(self):
        """Test service initialization with default configuration."""
        config = EmbeddingConfiguration()
        service = HuggingFaceEmbedding(config)

        assert service.embedding_dim == 384
        assert service.config.model_name == "paraphrase-multilingual-MiniLM-L12-v2"
        assert service.config.batch_size == 100

    def test_initialization_custom_config(self):
        """Test service initialization with custom configuration."""
        config = EmbeddingConfiguration(
            batch_size=50,
            max_seq_length=256
        )
        service = HuggingFaceEmbedding(config)

        assert service.config.batch_size == 50
        assert service.config.max_seq_length == 256

    def test_get_embedding_dimension(self, embedding_service):
        """Test get_embedding_dimension() returns correct value."""
        assert embedding_service.get_embedding_dimension() == 384


class TestKoreanTextEmbedding:
    """Test Korean language text embedding capabilities."""

    def test_single_korean_text(self, embedding_service):
        """Test embedding single Korean text."""
        text = "안녕하세요"
        embedding = embedding_service.embed_text(text)

        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

    def test_korean_database_terminology(self, embedding_service):
        """Test embedding Korean database terminology."""
        texts = [
            "PostgreSQL 데이터베이스",
            "인덱스 최적화",
            "트랜잭션 격리 수준",
            "쿼리 성능 튜닝"
        ]

        for text in texts:
            embedding = embedding_service.embed_text(text)
            assert len(embedding) == 384

    def test_korean_sentence_embedding(self, embedding_service):
        """Test embedding complete Korean sentences."""
        sentence = "관계형 데이터베이스는 ACID 속성을 보장합니다."
        embedding = embedding_service.embed_text(sentence)

        assert len(embedding) == 384
        # Check L2 normalization (magnitude should be ~1.0)
        magnitude = np.linalg.norm(embedding)
        assert abs(magnitude - 1.0) < 1e-5

    def test_korean_mixed_content(self, embedding_service):
        """Test embedding Korean text with mixed English."""
        text = "PostgreSQL은 오픈소스 RDBMS입니다"
        embedding = embedding_service.embed_text(text)

        assert len(embedding) == 384


class TestMultilingualSupport:
    """Test multilingual embedding capabilities."""

    def test_english_text(self, embedding_service):
        """Test embedding English text."""
        text = "PostgreSQL is a powerful database system"
        embedding = embedding_service.embed_text(text)

        assert len(embedding) == 384

    def test_japanese_text(self, embedding_service):
        """Test embedding Japanese text."""
        text = "データベース管理システム"
        embedding = embedding_service.embed_text(text)

        assert len(embedding) == 384

    def test_chinese_text(self, embedding_service):
        """Test embedding Chinese text."""
        text = "数据库索引优化"
        embedding = embedding_service.embed_text(text)

        assert len(embedding) == 384

    def test_multilingual_batch(self, embedding_service):
        """Test batch embedding with multiple languages."""
        texts = [
            "한국어 텍스트",
            "English text",
            "日本語テキスト",
            "中文文本"
        ]
        embeddings = embedding_service.embed_texts(texts)

        assert len(embeddings) == 4
        assert all(len(emb) == 384 for emb in embeddings)


class TestBatchEmbedding:
    """Test batch embedding functionality."""

    def test_batch_embedding_small(self, embedding_service):
        """Test batch embedding with small number of texts."""
        texts = [
            "첫 번째 문서",
            "두 번째 문서",
            "세 번째 문서"
        ]
        embeddings = embedding_service.embed_texts(texts)

        assert len(embeddings) == 3
        assert all(len(emb) == 384 for emb in embeddings)

    def test_batch_embedding_large(self, embedding_service):
        """Test batch embedding with larger number of texts."""
        texts = [f"문서 번호 {i+1}" for i in range(100)]
        embeddings = embedding_service.embed_texts(texts)

        assert len(embeddings) == 100
        assert all(len(emb) == 384 for emb in embeddings)

    def test_batch_embedding_custom_batch_size(self, embedding_service):
        """Test batch embedding with custom batch size."""
        texts = [f"텍스트 {i+1}" for i in range(50)]
        embeddings = embedding_service.embed_texts(texts, batch_size=10)

        assert len(embeddings) == 50

    def test_batch_embedding_single_text(self, embedding_service):
        """Test batch embedding with single text (edge case)."""
        texts = ["단일 문서"]
        embeddings = embedding_service.embed_texts(texts)

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 384


class TestL2Normalization:
    """Test L2 normalization of embeddings."""

    def test_single_embedding_normalized(self, embedding_service):
        """Test single embedding is L2 normalized."""
        text = "정규화 테스트"
        embedding = embedding_service.embed_text(text)

        magnitude = np.linalg.norm(embedding)
        assert abs(magnitude - 1.0) < 1e-5

    def test_batch_embeddings_normalized(self, embedding_service):
        """Test all batch embeddings are L2 normalized."""
        texts = ["텍스트 1", "텍스트 2", "텍스트 3"]
        embeddings = embedding_service.embed_texts(texts)

        for embedding in embeddings:
            magnitude = np.linalg.norm(embedding)
            assert abs(magnitude - 1.0) < 1e-5


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_text_raises_error(self, embedding_service):
        """Test empty text raises ValueError."""
        with pytest.raises(ValueError, match="Empty text cannot be embedded"):
            embedding_service.embed_text("")

    def test_whitespace_only_raises_error(self, embedding_service):
        """Test whitespace-only text raises ValueError."""
        with pytest.raises(ValueError, match="Empty text cannot be embedded"):
            embedding_service.embed_text("   ")

    def test_empty_text_list_raises_error(self, embedding_service):
        """Test empty text list raises ValueError."""
        with pytest.raises(ValueError, match="Empty text list cannot be embedded"):
            embedding_service.embed_texts([])

    def test_very_long_text_truncation(self, embedding_service):
        """Test very long text is handled (auto-truncated to 512 tokens)."""
        long_text = "긴 텍스트입니다. " * 200  # ~400 words, >512 tokens
        embedding = embedding_service.embed_text(long_text)

        # Should still return valid embedding
        assert len(embedding) == 384


class TestSemanticSimilarity:
    """Test semantic similarity of embeddings."""

    def test_similar_korean_texts_close_embeddings(self, embedding_service):
        """Test semantically similar Korean texts have close embeddings."""
        text1 = "데이터베이스 성능 최적화"
        text2 = "DB 성능 향상"

        emb1 = np.array(embedding_service.embed_text(text1))
        emb2 = np.array(embedding_service.embed_text(text2))

        # Cosine similarity (dot product of L2 normalized vectors)
        similarity = np.dot(emb1, emb2)

        # Similar texts should have high similarity (>0.5)
        assert similarity > 0.5

    def test_dissimilar_korean_texts_distant_embeddings(self, embedding_service):
        """Test semantically dissimilar Korean texts have distant embeddings."""
        text1 = "데이터베이스 관리"
        text2 = "날씨가 좋습니다"

        emb1 = np.array(embedding_service.embed_text(text1))
        emb2 = np.array(embedding_service.embed_text(text2))

        similarity = np.dot(emb1, emb2)

        # Dissimilar texts should have lower similarity (<0.5)
        assert similarity < 0.5

    def test_cross_language_semantic_similarity(self, embedding_service):
        """Test semantic similarity across languages."""
        ko_text = "데이터베이스"
        en_text = "database"
        ja_text = "データベース"

        ko_emb = np.array(embedding_service.embed_text(ko_text))
        en_emb = np.array(embedding_service.embed_text(en_text))
        ja_emb = np.array(embedding_service.embed_text(ja_text))

        # Cross-language similarity should be relatively high for same concept
        ko_en_sim = np.dot(ko_emb, en_emb)
        ko_ja_sim = np.dot(ko_emb, ja_emb)

        assert ko_en_sim > 0.4  # Multilingual model should capture semantic similarity
        assert ko_ja_sim > 0.4


class TestModelValidation:
    """Test model validation functionality."""

    def test_validate_model_returns_true(self, embedding_service):
        """Test validate_model() returns True for properly loaded model."""
        assert embedding_service.validate_model() == True

    def test_validate_model_checks_dimension(self, embedding_service):
        """Test validate_model() checks embedding dimension."""
        # This test validates the validation method itself
        result = embedding_service.validate_model()
        assert result == True

        # Test embedding from validation has correct dimension
        test_embedding = embedding_service.embed_text("테스트")
        assert len(test_embedding) == embedding_service.embedding_dim

    def test_validate_model_checks_normalization(self, embedding_service):
        """Test validate_model() checks L2 normalization."""
        result = embedding_service.validate_model()
        assert result == True

        # Verify normalization
        test_embedding = embedding_service.embed_text("정규화 확인")
        magnitude = np.linalg.norm(test_embedding)
        assert abs(magnitude - 1.0) < 1e-6


class TestSpecialCharacters:
    """Test handling of special characters and edge cases."""

    def test_korean_with_numbers(self, embedding_service):
        """Test Korean text with numbers."""
        text = "PostgreSQL 13 버전 업데이트"
        embedding = embedding_service.embed_text(text)
        assert len(embedding) == 384

    def test_korean_with_punctuation(self, embedding_service):
        """Test Korean text with punctuation."""
        text = "데이터베이스란 무엇인가? 구조화된 데이터의 집합입니다!"
        embedding = embedding_service.embed_text(text)
        assert len(embedding) == 384

    def test_korean_with_special_characters(self, embedding_service):
        """Test Korean text with special characters."""
        text = "DB 성능 향상 (10% → 20%)"
        embedding = embedding_service.embed_text(text)
        assert len(embedding) == 384

    def test_urls_in_text(self, embedding_service):
        """Test text containing URLs."""
        text = "https://www.postgresql.org 에서 문서를 확인하세요"
        embedding = embedding_service.embed_text(text)
        assert len(embedding) == 384


class TestExtendedMultilingualSupport:
    """Test extended multilingual support and special character handling."""

    def test_multilingual_embedding_korean(self, embedding_service):
        """Test Korean text embedding."""
        text = "데이터베이스 인덱스"
        embedding = embedding_service.embed_text(text)

        assert len(embedding) == 384
        print(f"Korean: {text} → embedding generated")

    def test_multilingual_embedding_english(self, embedding_service):
        """Test English text embedding."""
        text = "database index"
        embedding = embedding_service.embed_text(text)

        assert len(embedding) == 384
        print(f"English: {text} → embedding generated")

    def test_multilingual_embedding_mixed(self, embedding_service):
        """Test mixed Korean-English text embedding."""
        text = "PostgreSQL의 B-tree 인덱스"
        embedding = embedding_service.embed_text(text)

        assert len(embedding) == 384
        print(f"Mixed: {text} → embedding generated")

    def test_multilingual_embedding_japanese(self, embedding_service):
        """Test Japanese text embedding."""
        text = "日本語のテキスト"
        embedding = embedding_service.embed_text(text)

        assert len(embedding) == 384
        print(f"Japanese: {text} → embedding generated")

    def test_multilingual_embedding_chinese(self, embedding_service):
        """Test Chinese text embedding."""
        text = "中文文本"
        embedding = embedding_service.embed_text(text)

        assert len(embedding) == 384
        print(f"Chinese: {text} → embedding generated")

    def test_special_characters_sql(self, embedding_service):
        """Test SQL special characters."""
        text = "SQL의 WHERE 조건절 (condition)"
        embedding = embedding_service.embed_text(text)
        assert len(embedding) == 384

    def test_special_characters_python(self, embedding_service):
        """Test Python f-string syntax."""
        text = "Python f-string {variable}"
        embedding = embedding_service.embed_text(text)
        assert len(embedding) == 384

    def test_special_characters_regex(self, embedding_service):
        """Test regex patterns."""
        text = "정규표현식 [a-zA-Z]+"
        embedding = embedding_service.embed_text(text)
        assert len(embedding) == 384

    def test_special_characters_emoji(self, embedding_service):
        """Test emoji characters."""
        text = "이모지 포함 😀 텍스트"
        embedding = embedding_service.embed_text(text)
        assert len(embedding) == 384

    def test_unicode_normalization(self, embedding_service):
        """Test Unicode normalization (NFD vs NFC)."""
        # Both forms should produce similar embeddings
        text_nfd = "한글"  # NFD form
        text_nfc = "한글"  # NFC form

        embedding_nfd = embedding_service.embed_text(text_nfd)
        embedding_nfc = embedding_service.embed_text(text_nfc)

        # Calculate similarity
        similarity = np.dot(embedding_nfd, embedding_nfc)
        assert similarity > 0.99, f"Unicode normalization similarity {similarity:.4f} too low"

    def test_encoding_edge_case_newlines(self, embedding_service):
        """Test text with multiple newlines."""
        text = "\n\n\n텍스트\n\n"
        embedding = embedding_service.embed_text(text)
        assert len(embedding) == 384

    def test_encoding_edge_case_tabs(self, embedding_service):
        """Test text with tabs."""
        text = "\t\t텍스트\t\t"
        embedding = embedding_service.embed_text(text)
        assert len(embedding) == 384

    def test_encoding_edge_case_spaces(self, embedding_service):
        """Test text with multiple spaces."""
        text = "   텍스트   "
        embedding = embedding_service.embed_text(text)
        assert len(embedding) == 384

    def test_encoding_edge_case_crlf(self, embedding_service):
        """Test Windows CRLF line endings."""
        text = "텍스트\r\n윈도우"
        embedding = embedding_service.embed_text(text)
        assert len(embedding) == 384

    def test_multilingual_batch_processing(self, embedding_service):
        """Test batch processing with multilingual texts."""
        texts = [
            "한국어 텍스트",
            "English text",
            "日本語テキスト",
            "中文文本",
            "PostgreSQL의 B-tree 인덱스"
        ]

        embeddings = embedding_service.embed_texts(texts)

        assert len(embeddings) == 5
        assert all(len(emb) == 384 for emb in embeddings)
        print(f"Multilingual batch: {len(texts)} texts processed")

    def test_special_characters_comprehensive(self, embedding_service):
        """Test comprehensive special character handling."""
        texts = [
            "SQL의 WHERE 조건절 (condition)",
            "Python f-string {variable}",
            "정규표현식 [a-zA-Z]+",
            "이모지 포함 😀 텍스트",
        ]

        for text in texts:
            embedding = embedding_service.embed_text(text)
            assert len(embedding) == 384

        print(f"Special characters: {len(texts)} variants tested")

    def test_cross_language_semantic_similarity(self, embedding_service):
        """Test semantic similarity across languages."""
        korean_text = "데이터베이스"
        english_text = "database"
        japanese_text = "データベース"
        chinese_text = "数据库"

        ko_emb = np.array(embedding_service.embed_text(korean_text))
        en_emb = np.array(embedding_service.embed_text(english_text))
        ja_emb = np.array(embedding_service.embed_text(japanese_text))
        zh_emb = np.array(embedding_service.embed_text(chinese_text))

        # Cross-language similarity should be relatively high for same concept
        ko_en_sim = np.dot(ko_emb, en_emb)
        ko_ja_sim = np.dot(ko_emb, ja_emb)
        ko_zh_sim = np.dot(ko_emb, zh_emb)

        print(f"\nCross-language similarity:")
        print(f"  Korean-English: {ko_en_sim:.3f}")
        print(f"  Korean-Japanese: {ko_ja_sim:.3f}")
        print(f"  Korean-Chinese: {ko_zh_sim:.3f}")

        # Multilingual model should capture semantic similarity
        assert ko_en_sim > 0.4, f"Korean-English similarity {ko_en_sim:.3f} too low"
        assert ko_ja_sim > 0.4, f"Korean-Japanese similarity {ko_ja_sim:.3f} too low"
        assert ko_zh_sim > 0.4, f"Korean-Chinese similarity {ko_zh_sim:.3f} too low"

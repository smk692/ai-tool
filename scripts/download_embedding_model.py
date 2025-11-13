#!/usr/bin/env python3
"""
Download and cache the embedding model.

This script downloads paraphrase-multilingual-MiniLM-L12-v2 from Hugging Face
and caches it locally for faster startup.
"""

from pathlib import Path

from sentence_transformers import SentenceTransformer

from config.settings import settings


def download_model() -> None:
    """Download and cache the embedding model."""
    model_name = settings.embedding_model_name
    print(f"Downloading {model_name}...")

    try:
        # Download model (will be cached in ~/.cache/huggingface/)
        model = SentenceTransformer(model_name)

        print(f"✅ Model downloaded successfully")
        print(f"   Model: {model_name}")
        print(f"   Dimensions: {model.get_sentence_embedding_dimension()}")
        print(f"   Max sequence length: {model.max_seq_length}")

        # Test Korean language support
        print("\nTesting Korean language support...")
        test_text = "안녕하세요. 이것은 테스트 문장입니다."
        embedding = model.encode(test_text)
        print(f"✅ Korean test passed")
        print(f"   Embedding shape: {embedding.shape}")

    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        raise


if __name__ == "__main__":
    download_model()

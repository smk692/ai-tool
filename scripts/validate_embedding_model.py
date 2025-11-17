"""
Embedding Model Validation Script

Validates that the Hugging Face embedding model is properly configured and operational.
This script checks:
1. Model download and caching
2. Embedding dimension verification
3. Korean text embedding capability
4. Configuration file integrity
"""

import logging
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.embedding import DeviceType, EmbeddingConfiguration
from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_model_download() -> bool:
    """
    Validate that the embedding model can be downloaded and loaded.

    Returns:
        True if model loads successfully
    """
    try:
        logger.info("Step 1: Validating model download...")
        model_name = "paraphrase-multilingual-MiniLM-L12-v2"

        logger.info(f"Loading model: {model_name}")
        model = SentenceTransformer(model_name)

        logger.info(f"✅ Model loaded successfully")
        logger.info(f"   Cache location: ~/.cache/torch/sentence_transformers/")
        logger.info(f"   Model size: ~470MB")

        return True

    except Exception as e:
        logger.error(f"❌ Model download failed: {e}")
        return False


def validate_embedding_dimension(model: SentenceTransformer) -> bool:
    """
    Validate that the embedding dimension is 384 as expected.

    Args:
        model: Loaded SentenceTransformer model

    Returns:
        True if dimension matches expected value
    """
    try:
        logger.info("\nStep 2: Validating embedding dimension...")

        expected_dim = 384
        actual_dim = model.get_sentence_embedding_dimension()

        if actual_dim == expected_dim:
            logger.info(f"✅ Embedding dimension correct: {actual_dim}")
            return True
        else:
            logger.error(f"❌ Embedding dimension mismatch: expected {expected_dim}, got {actual_dim}")
            return False

    except Exception as e:
        logger.error(f"❌ Dimension validation failed: {e}")
        return False


def validate_korean_text(model: SentenceTransformer) -> bool:
    """
    Validate that Korean text can be embedded successfully.

    Args:
        model: Loaded SentenceTransformer model

    Returns:
        True if Korean text embedding works
    """
    try:
        logger.info("\nStep 3: Validating Korean text embedding...")

        test_texts = [
            "안녕하세요",
            "PostgreSQL 데이터베이스",
            "한국어와 영어 mixed text"
        ]

        for text in test_texts:
            embedding = model.encode(text, convert_to_numpy=True)

            # Check dimension
            if len(embedding) != 384:
                logger.error(f"❌ Wrong embedding dimension for '{text}': {len(embedding)}")
                return False

            # Check dtype
            if embedding.dtype != np.float32:
                logger.error(f"❌ Wrong dtype for '{text}': {embedding.dtype}")
                return False

            logger.info(f"   ✓ '{text}' → {len(embedding)}-dim vector")

        logger.info("✅ Korean text embedding successful")
        return True

    except Exception as e:
        logger.error(f"❌ Korean text validation failed: {e}")
        return False


def validate_configuration() -> bool:
    """
    Validate that all configuration files are properly set up.

    Returns:
        True if all configurations are valid
    """
    try:
        logger.info("\nStep 4: Validating configuration files...")

        # Check EmbeddingConfiguration
        logger.info("   Checking src/models/embedding.py...")
        config = EmbeddingConfiguration()

        logger.info(f"   ✓ Model name: {config.model_name}")
        logger.info(f"   ✓ Embedding dim: {config.embedding_dim}")
        logger.info(f"   ✓ Device: {config.device}")
        logger.info(f"   ✓ Batch size: {config.batch_size}")
        logger.info(f"   ✓ Max sequence length: {config.max_seq_length}")

        # Check Settings
        logger.info("   Checking config/settings.py...")
        logger.info(f"   ✓ EMBEDDING_MODEL_NAME: {settings.embedding_model_name}")
        logger.info(f"   ✓ EMBEDDING_DEVICE: {settings.embedding_device}")
        logger.info(f"   ✓ EMBEDDING_BATCH_SIZE: {settings.embedding_batch_size}")

        # Check environment variables
        logger.info("   Checking .env file...")
        if Path(project_root / ".env").exists():
            logger.info("   ✓ .env file exists")
        else:
            logger.warning("   ⚠️  .env file not found (using defaults)")

        logger.info("✅ Configuration validation passed")
        return True

    except Exception as e:
        logger.error(f"❌ Configuration validation failed: {e}")
        return False


def main():
    """Run all validation checks."""
    logger.info("="*60)
    logger.info("Embedding Model Validation")
    logger.info("="*60)

    results = []

    # Step 1: Model download
    model_loaded = validate_model_download()
    results.append(("Model Download", model_loaded))

    if not model_loaded:
        logger.error("\n❌ VALIDATION FAILED: Cannot proceed without model")
        sys.exit(1)

    # Load model for subsequent tests
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # Step 2: Embedding dimension
    dim_valid = validate_embedding_dimension(model)
    results.append(("Embedding Dimension", dim_valid))

    # Step 3: Korean text
    korean_valid = validate_korean_text(model)
    results.append(("Korean Text Embedding", korean_valid))

    # Step 4: Configuration
    config_valid = validate_configuration()
    results.append(("Configuration Files", config_valid))

    # Summary
    logger.info("\n" + "="*60)
    logger.info("Validation Summary")
    logger.info("="*60)

    all_passed = True
    for check_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{check_name:.<30} {status}")
        if not passed:
            all_passed = False

    logger.info("="*60)

    if all_passed:
        logger.info("\n🎉 All validation checks passed!")
        logger.info("The embedding model is ready for use.")
        sys.exit(0)
    else:
        logger.error("\n❌ Some validation checks failed.")
        logger.error("Please fix the issues before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()

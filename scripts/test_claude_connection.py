#!/usr/bin/env python
"""Test script for Claude API connection."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from src.services.llm_client import LLMClient
from src.utils.logging import logger


def main():
    """Test Claude API connection with simple query."""
    print("=" * 60)
    print("Claude API Connection Test")
    print("=" * 60)

    # Check API key
    print("\n1. Checking API Key Configuration...")
    if not settings.anthropic_api_key:
        print("❌ ANTHROPIC_API_KEY not found in environment variables")
        print("   Please set ANTHROPIC_API_KEY in your .env file")
        return False

    print(f"✅ API Key configured: {settings.anthropic_api_key[:10]}...")

    # Check model configuration
    print(f"\n2. Model Configuration:")
    print(f"   - Model: {settings.claude_model}")
    print(f"   - Temperature: {settings.claude_temperature}")
    print(f"   - Max Tokens: {settings.claude_max_tokens}")
    print(f"   - Timeout: {settings.claude_timeout}s")
    print(f"   - Max Retries: {settings.claude_max_retries}")

    # Initialize LLM client
    print("\n3. Initializing LLM Client...")
    try:
        llm_client = LLMClient()
        print("✅ LLM Client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize LLM Client: {e}")
        return False

    # Test connection with simple query
    print("\n4. Testing API Connection...")
    try:
        success = llm_client.test_connection()
        if success:
            print("✅ API connection test PASSED")
        else:
            print("❌ API connection test FAILED")
            return False
    except Exception as e:
        print(f"❌ API connection test error: {e}")
        return False

    # Test with Korean query
    print("\n5. Testing Korean Language Support...")
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content="당신은 친절한 AI 어시스턴트입니다."),
            HumanMessage(content="안녕하세요! 오늘 날씨는 어떤가요?"),
        ]

        result = llm_client.invoke(messages)

        print("✅ Korean query successful")
        print(f"   Response preview: {result['content'][:100]}...")
        print(f"   Token usage: {result['token_usage'].total_tokens} tokens")
        print(f"   Execution time: {result['execution_time']:.2f}s")

    except Exception as e:
        print(f"❌ Korean query test failed: {e}")
        return False

    # Test error handling
    print("\n6. Testing Error Handling...")
    try:
        # Test with invalid configuration
        from src.models.llm_config import LLMConfiguration

        invalid_config = LLMConfiguration(
            api_key="invalid_key",
            model_name=settings.claude_model,
            temperature=settings.claude_temperature,
            max_tokens=settings.claude_max_tokens,
            timeout=5,  # Short timeout for quick test
            max_retries=1,  # Only 1 retry for speed
        )

        test_client = LLMClient(config=invalid_config)

        messages = [
            SystemMessage(content="Test"),
            HumanMessage(content="Test"),
        ]

        try:
            test_client.invoke(messages)
            print("⚠️  Expected authentication error but succeeded")
        except Exception as expected_error:
            print("✅ Error handling works correctly")
            print(f"   Caught expected error: {type(expected_error).__name__}")

    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print("✅ All tests PASSED")
    print("\nClaude API is properly configured and working!")
    print("\nNext steps:")
    print("1. Run unit tests: pytest tests/")
    print("2. Run E2E tests: pytest tests/e2e/")
    print("3. Start the application")

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        logger.exception("Test script failed")
        sys.exit(1)

"""
AI Assistant 사용 예제 스크립트

이 스크립트는 각 체인의 실제 사용법을 보여줍니다.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.chains.router import RouterChain
from src.chains.text_to_sql import TextToSQLChain
from src.chains.knowledge import KnowledgeChain
from src.chains.multi_turn import MultiTurnChain
from src.models.query_response import QueryRequest, QueryType
from src.services.llm_client import LLMClient
from src.services.embedding import HuggingFaceEmbedding
from src.services.memory import SQLiteConversationMemory
from src.utils.logging import logger


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


def example_router_chain():
    """Example: Intent Classification using Router Chain."""
    print_section("1. Intent Classification (Router Chain)")

    llm_client = LLMClient()
    router = RouterChain(llm_client)

    # Test queries in Korean
    test_queries = [
        ("지난달 신규 가입자 수는?", QueryType.TEXT_TO_SQL),
        ("회원가입 절차가 어떻게 되나요?", QueryType.KNOWLEDGE),
        ("안녕하세요, 오늘 날씨가 좋네요", QueryType.ASSISTANT),
        ("최근 1주일간 일별 주문 금액을 조회해주세요", QueryType.TEXT_TO_SQL),
        ("비밀번호 재설정하는 방법 알려주세요", QueryType.KNOWLEDGE),
    ]

    for query_text, expected_type in test_queries:
        query = QueryRequest(user_id="example_user", query_text=query_text)

        try:
            query_type = router.classify(query)
            status = "✅" if query_type == expected_type else "⚠️"
            print(f"{status} Query: {query_text}")
            print(f"   Classified as: {query_type.value}")
            print(f"   Expected: {expected_type.value}")
            print()
        except Exception as e:
            print(f"❌ Error classifying query: {e}\n")


def example_text_to_sql_chain():
    """Example: Text-to-SQL generation."""
    print_section("2. Text-to-SQL Generation")

    llm_client = LLMClient()
    text_to_sql = TextToSQLChain(llm_client)

    # Test SQL generation queries
    sql_queries = [
        "지난 7일간 일별 신규 가입자 수를 조회해주세요",
        "이번 달 총 주문 금액은 얼마인가요?",
        "가장 많이 팔린 상품 10개를 보여주세요",
    ]

    for query_text in sql_queries:
        query = QueryRequest(user_id="example_user", query_text=query_text)

        try:
            response = text_to_sql.generate_sql(query)
            print(f"Query: {query_text}")
            print(f"Generated SQL:")
            print("-" * 60)
            print(response.sql_query)
            print("-" * 60)
            print(f"Confidence: {response.confidence_score:.2f}")
            print(f"Token Usage: {response.token_usage.total_tokens} tokens")
            print()
        except Exception as e:
            print(f"❌ Error generating SQL: {e}\n")


def example_knowledge_chain():
    """Example: RAG-based Knowledge Discovery."""
    print_section("3. Knowledge Discovery (RAG)")

    llm_client = LLMClient()
    embedding_service = HuggingFaceEmbedding()
    knowledge_chain = KnowledgeChain(llm_client, embedding_service)

    # Test knowledge queries
    knowledge_queries = [
        "회원가입할 때 이메일 인증이 필요한가요?",
        "비밀번호는 어떤 조건을 만족해야 하나요?",
        "계정을 삭제하려면 어떻게 해야 하나요?",
    ]

    for query_text in knowledge_queries:
        query = QueryRequest(user_id="example_user", query_text=query_text)

        try:
            response = knowledge_chain.search(query, top_k=3)
            print(f"Query: {query_text}")
            print(f"Answer:")
            print("-" * 60)
            print(response.answer)
            print("-" * 60)
            print(f"Confidence: {response.confidence_score:.2f}")
            print(f"Source Documents: {len(response.source_documents)}")
            for i, doc in enumerate(response.source_documents, 1):
                print(f"  {i}. {doc.title} (relevance: {doc.relevance_score:.2f})")
            print(f"Token Usage: {response.token_usage.total_tokens} tokens")
            print()
        except Exception as e:
            print(f"❌ Error searching knowledge: {e}\n")


def example_multi_turn_chat():
    """Example: Multi-turn conversation with history."""
    print_section("4. Multi-turn Conversation")

    llm_client = LLMClient()
    memory = SQLiteConversationMemory()
    chat = MultiTurnChain(llm_client, memory)

    session_id = "example_session_001"

    # Conversation turns
    conversation = [
        "안녕하세요!",
        "주문 내역을 확인하고 싶어요",
        "지난달 주문 내역이요",
        "감사합니다!",
    ]

    for i, query_text in enumerate(conversation, 1):
        query = QueryRequest(
            user_id="example_user", session_id=session_id, query_text=query_text
        )

        try:
            response = chat.chat(query)
            print(f"Turn {i}:")
            print(f"User: {query_text}")
            print(f"Assistant: {response.answer}")
            print(f"Confidence: {response.confidence_score:.2f}")
            print(f"Token Usage: {response.token_usage.total_tokens} tokens")
            print()
        except Exception as e:
            print(f"❌ Error in conversation: {e}\n")

    # Display conversation history
    print("-" * 60)
    print("Conversation History:")
    print("-" * 60)
    history = memory.get_conversation_history(session_id, limit=10)
    for i, turn in enumerate(history, 1):
        print(f"Turn {i}:")
        print(f"  User: {turn['user_message']}")
        print(f"  Assistant: {turn['assistant_message']}")
        print()


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print(" AI Assistant - Usage Examples")
    print(" Claude Code + Hugging Face Embeddings")
    print("=" * 80)

    try:
        # Check LLM connection
        print("\n🔄 Testing Claude API connection...")
        llm_client = LLMClient()
        if llm_client.test_connection():
            print("✅ Claude API connection successful!\n")
        else:
            print("❌ Claude API connection failed. Check your API key.\n")
            return

        # Run examples
        example_router_chain()
        example_text_to_sql_chain()
        example_knowledge_chain()
        example_multi_turn_chat()

        print("\n" + "=" * 80)
        print(" All examples completed!")
        print("=" * 80 + "\n")

    except KeyboardInterrupt:
        print("\n\n⚠️ Examples interrupted by user.\n")
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()

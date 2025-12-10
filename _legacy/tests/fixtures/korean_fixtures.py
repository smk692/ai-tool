"""Korean language test fixtures for LLM operations."""

import pytest
from decimal import Decimal
from uuid import uuid4

from src.models.query_request import QueryRequest, QueryType


@pytest.fixture
def korean_text_to_sql_queries():
    """Sample Korean Text-to-SQL queries."""
    return [
        {
            "query": "지난달 신규 가입자 수는?",
            "expected_type": QueryType.TEXT_TO_SQL,
            "description": "Last month's new user count",
        },
        {
            "query": "이번주 매출 총액은 얼마인가요?",
            "expected_type": QueryType.TEXT_TO_SQL,
            "description": "This week's total revenue",
        },
        {
            "query": "상위 10명의 고객과 그들의 총 구매액을 알려주세요",
            "expected_type": QueryType.TEXT_TO_SQL,
            "description": "Top 10 customers by purchase amount",
        },
        {
            "query": "어제 주문한 고객 목록을 보여주세요",
            "expected_type": QueryType.TEXT_TO_SQL,
            "description": "Yesterday's order customers",
        },
        {
            "query": "평균 주문 금액이 가장 높은 카테고리는?",
            "expected_type": QueryType.TEXT_TO_SQL,
            "description": "Category with highest average order value",
        },
    ]


@pytest.fixture
def korean_knowledge_queries():
    """Sample Korean Knowledge Discovery queries."""
    return [
        {
            "query": "회원가입 절차가 어떻게 되나요?",
            "expected_type": QueryType.KNOWLEDGE,
            "description": "Membership registration process",
        },
        {
            "query": "비밀번호 재설정하는 방법을 알려주세요",
            "expected_type": QueryType.KNOWLEDGE,
            "description": "Password reset instructions",
        },
        {
            "query": "환불 정책은 어떻게 되나요?",
            "expected_type": QueryType.KNOWLEDGE,
            "description": "Refund policy information",
        },
        {
            "query": "배송은 얼마나 걸리나요?",
            "expected_type": QueryType.KNOWLEDGE,
            "description": "Delivery timeframe",
        },
        {
            "query": "고객센터 운영 시간이 궁금합니다",
            "expected_type": QueryType.KNOWLEDGE,
            "description": "Customer service hours",
        },
    ]


@pytest.fixture
def korean_assistant_queries():
    """Sample Korean general assistant queries."""
    return [
        {
            "query": "안녕하세요",
            "expected_type": QueryType.ASSISTANT,
            "description": "Greeting",
        },
        {
            "query": "도움이 필요해요",
            "expected_type": QueryType.ASSISTANT,
            "description": "Request for help",
        },
        {
            "query": "감사합니다",
            "expected_type": QueryType.ASSISTANT,
            "description": "Thank you",
        },
        {
            "query": "다시 설명해줄 수 있나요?",
            "expected_type": QueryType.ASSISTANT,
            "description": "Request for clarification",
        },
        {
            "query": "좋아요, 알겠습니다",
            "expected_type": QueryType.ASSISTANT,
            "description": "Acknowledgment",
        },
    ]


@pytest.fixture
def sample_query_request():
    """Create a sample QueryRequest."""
    return QueryRequest(
        query_id=uuid4(),
        user_id="test_user_001",
        query_text="지난달 매출은 얼마인가요?",
        query_language="ko",
        query_type=QueryType.TEXT_TO_SQL,
        session_id=uuid4(),
    )


@pytest.fixture
def korean_documents():
    """Sample Korean documents for knowledge discovery."""
    return [
        {
            "title": "회원가입 가이드",
            "content": """회원가입 절차는 다음과 같습니다:
1. 이메일 주소 입력
2. 비밀번호 설정 (8자 이상, 특수문자 포함)
3. 이메일 인증
4. 프로필 정보 입력
5. 가입 완료

가입 후에는 즉시 서비스를 이용하실 수 있습니다.""",
        },
        {
            "title": "환불 정책",
            "content": """환불 정책:
- 구매 후 7일 이내 전액 환불 가능
- 제품 개봉 시 환불 불가
- 배송비는 고객 부담
- 환불 처리 기간: 영업일 기준 3-5일
- 환불 신청은 고객센터 또는 앱에서 가능합니다.""",
        },
        {
            "title": "배송 안내",
            "content": """배송 정보:
- 일반 배송: 주문 후 2-3일 소요
- 빠른 배송: 주문 후 1일 (추가 요금 발생)
- 제주도/도서산간 지역: 추가 1-2일 소요
- 배송 추적은 마이페이지에서 확인 가능
- 배송비: 30,000원 이상 구매 시 무료"""
        },
    ]


@pytest.fixture
def database_schema():
    """Sample database schema for SQL generation tests."""
    return {
        "users": [
            "user_id (INTEGER, PRIMARY KEY)",
            "email (VARCHAR(255), UNIQUE)",
            "name (VARCHAR(100))",
            "created_at (TIMESTAMP)",
            "updated_at (TIMESTAMP)",
        ],
        "orders": [
            "order_id (INTEGER, PRIMARY KEY)",
            "user_id (INTEGER, FOREIGN KEY -> users.user_id)",
            "total_amount (DECIMAL(10,2))",
            "status (VARCHAR(50))",
            "created_at (TIMESTAMP)",
        ],
        "order_items": [
            "item_id (INTEGER, PRIMARY KEY)",
            "order_id (INTEGER, FOREIGN KEY -> orders.order_id)",
            "product_id (INTEGER)",
            "quantity (INTEGER)",
            "unit_price (DECIMAL(10,2))",
        ],
        "products": [
            "product_id (INTEGER, PRIMARY KEY)",
            "name (VARCHAR(200))",
            "category (VARCHAR(100))",
            "price (DECIMAL(10,2))",
            "stock_quantity (INTEGER)",
        ],
    }


@pytest.fixture
def expected_sql_queries():
    """Expected SQL queries for Korean questions."""
    return {
        "지난달 신규 가입자 수는?": "SELECT COUNT(*) FROM users WHERE created_at >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month') AND created_at < DATE_TRUNC('month', CURRENT_DATE)",
        "이번주 매출 총액은 얼마인가요?": "SELECT SUM(total_amount) FROM orders WHERE created_at >= DATE_TRUNC('week', CURRENT_DATE)",
        "상위 10명의 고객과 그들의 총 구매액을 알려주세요": "SELECT user_id, SUM(total_amount) as total_purchase FROM orders GROUP BY user_id ORDER BY total_purchase DESC LIMIT 10",
    }


@pytest.fixture
def sample_conversation_history():
    """Sample conversation history for multi-turn tests."""
    return [
        {
            "role": "user",
            "content": "안녕하세요",
        },
        {
            "role": "assistant",
            "content": "안녕하세요! 무엇을 도와드릴까요?",
        },
        {
            "role": "user",
            "content": "주문 내역을 확인하고 싶어요",
        },
        {
            "role": "assistant",
            "content": "주문 내역 확인을 도와드리겠습니다. 주문 번호나 주문일자를 알려주시겠어요?",
        },
    ]

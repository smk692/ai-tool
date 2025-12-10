"""
Vector Search Latency Performance Tests

Tests vector search performance to ensure:
- Search latency ≤0.5s for standard queries
- Consistent performance across different collection sizes
- Korean language query performance meets SLA targets
"""

import time
import pytest
import numpy as np
from typing import List, Dict

from src.models.embedding import EmbeddingConfiguration
from src.services.embeddings import HuggingFaceEmbedding
from src.services.vector_store import VectorStore


@pytest.fixture(scope="module")
def embedding_service():
    """Create shared embedding service for all tests."""
    config = EmbeddingConfiguration()
    return HuggingFaceEmbedding(config)


@pytest.fixture(scope="module")
def small_collection(embedding_service):
    """Create small collection (100 documents) for baseline testing."""
    vector_store = VectorStore(
        embedding_service=embedding_service,
        collection_name="perf_small_100"
    )

    # Generate 100 Korean database-related documents
    documents = [
        f"PostgreSQL 데이터베이스의 {i}번째 인덱스 최적화 방법입니다."
        for i in range(100)
    ]

    vector_store.add_documents(documents)
    yield vector_store

    # Cleanup
    vector_store.delete_collection()


@pytest.fixture(scope="module")
def medium_collection(embedding_service):
    """Create medium collection (1000 documents) for standard testing."""
    vector_store = VectorStore(
        embedding_service=embedding_service,
        collection_name="perf_medium_1000"
    )

    # Generate 1000 varied Korean documents
    base_topics = [
        "데이터베이스 인덱스 구조",
        "트랜잭션 격리 수준",
        "쿼리 실행 계획",
        "데이터 정규화 기법",
        "성능 모니터링 방법"
    ]

    documents = [
        f"{base_topics[i % 5]} - 문서 번호 {i+1}: 추가 설명 내용입니다."
        for i in range(1000)
    ]

    vector_store.add_documents(documents)
    yield vector_store

    # Cleanup
    vector_store.delete_collection()


@pytest.fixture(scope="module")
def large_collection(embedding_service):
    """Create large collection (5000 documents) for stress testing."""
    vector_store = VectorStore(
        embedding_service=embedding_service,
        collection_name="perf_large_5000"
    )

    # Generate 5000 documents
    documents = [
        f"데이터베이스 성능 튜닝 가이드 {i+1}: "
        f"인덱스, 쿼리 최적화, 캐싱 전략에 대한 상세 설명입니다."
        for i in range(5000)
    ]

    vector_store.add_documents(documents)
    yield vector_store

    # Cleanup
    vector_store.delete_collection()


class TestSearchLatencySLA:
    """Test vector search latency against SLA targets (≤0.5s)."""

    def test_single_query_latency_small_collection(self, small_collection):
        """Test single query latency on small collection (100 docs)."""
        query = "데이터베이스 인덱스 최적화"

        start_time = time.perf_counter()
        results = small_collection.query(query, top_k=5)
        latency = time.perf_counter() - start_time

        assert len(results["documents"]) == 5
        assert latency < 0.5, f"Search latency {latency:.3f}s exceeds SLA target 0.5s"
        print(f"\n  Small collection (100 docs): {latency:.3f}s")

    def test_single_query_latency_medium_collection(self, medium_collection):
        """Test single query latency on medium collection (1000 docs)."""
        query = "트랜잭션 처리 방법"

        start_time = time.perf_counter()
        results = medium_collection.query(query, top_k=5)
        latency = time.perf_counter() - start_time

        assert len(results["documents"]) == 5
        assert latency < 0.5, f"Search latency {latency:.3f}s exceeds SLA target 0.5s"
        print(f"\n  Medium collection (1000 docs): {latency:.3f}s")

    def test_single_query_latency_large_collection(self, large_collection):
        """Test single query latency on large collection (5000 docs)."""
        query = "성능 튜닝 가이드"

        start_time = time.perf_counter()
        results = large_collection.query(query, top_k=5)
        latency = time.perf_counter() - start_time

        assert len(results["documents"]) == 5
        assert latency < 0.5, f"Search latency {latency:.3f}s exceeds SLA target 0.5s"
        print(f"\n  Large collection (5000 docs): {latency:.3f}s")

    def test_multiple_queries_average_latency(self, medium_collection):
        """Test average latency across multiple queries."""
        queries = [
            "데이터베이스 최적화",
            "인덱스 구조",
            "트랜잭션 격리",
            "쿼리 성능",
            "데이터 정규화"
        ]

        latencies = []
        for query in queries:
            start_time = time.perf_counter()
            medium_collection.query(query, top_k=5)
            latency = time.perf_counter() - start_time
            latencies.append(latency)

        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)

        assert avg_latency < 0.5, f"Average latency {avg_latency:.3f}s exceeds SLA"
        assert max_latency < 0.5, f"Max latency {max_latency:.3f}s exceeds SLA"

        print(f"\n  Average latency: {avg_latency:.3f}s")
        print(f"  Max latency: {max_latency:.3f}s")
        print(f"  Min latency: {np.min(latencies):.3f}s")


class TestKoreanQueryPerformance:
    """Test Korean language query performance specifically."""

    def test_korean_short_query(self, medium_collection):
        """Test short Korean query performance."""
        query = "데이터베이스"

        start_time = time.perf_counter()
        results = medium_collection.query(query, top_k=5)
        latency = time.perf_counter() - start_time

        assert len(results["documents"]) > 0
        assert latency < 0.5
        print(f"\n  Short Korean query: {latency:.3f}s")

    def test_korean_long_query(self, medium_collection):
        """Test long Korean query performance."""
        query = "PostgreSQL 데이터베이스에서 인덱스를 사용하여 쿼리 성능을 최적화하는 방법"

        start_time = time.perf_counter()
        results = medium_collection.query(query, top_k=5)
        latency = time.perf_counter() - start_time

        assert len(results["documents"]) > 0
        assert latency < 0.5
        print(f"\n  Long Korean query: {latency:.3f}s")

    def test_korean_technical_terms(self, medium_collection):
        """Test Korean technical terminology query."""
        queries = [
            "B-Tree 인덱스",
            "MVCC 트랜잭션",
            "ACID 속성",
            "정규화 3NF",
            "쿼리 플랜"
        ]

        for query in queries:
            start_time = time.perf_counter()
            results = medium_collection.query(query, top_k=3)
            latency = time.perf_counter() - start_time

            assert len(results["documents"]) > 0
            assert latency < 0.5, f"Query '{query}' latency {latency:.3f}s exceeds SLA"


class TestDifferentTopKValues:
    """Test latency with different top_k values."""

    def test_top_k_1(self, medium_collection):
        """Test with top_k=1."""
        start_time = time.perf_counter()
        results = medium_collection.query("데이터베이스", top_k=1)
        latency = time.perf_counter() - start_time

        assert len(results["documents"]) == 1
        assert latency < 0.5
        print(f"\n  top_k=1: {latency:.3f}s")

    def test_top_k_5(self, medium_collection):
        """Test with top_k=5."""
        start_time = time.perf_counter()
        results = medium_collection.query("데이터베이스", top_k=5)
        latency = time.perf_counter() - start_time

        assert len(results["documents"]) == 5
        assert latency < 0.5
        print(f"\n  top_k=5: {latency:.3f}s")

    def test_top_k_10(self, medium_collection):
        """Test with top_k=10."""
        start_time = time.perf_counter()
        results = medium_collection.query("데이터베이스", top_k=10)
        latency = time.perf_counter() - start_time

        assert len(results["documents"]) == 10
        assert latency < 0.5
        print(f"\n  top_k=10: {latency:.3f}s")

    def test_top_k_20(self, medium_collection):
        """Test with top_k=20."""
        start_time = time.perf_counter()
        results = medium_collection.query("데이터베이스", top_k=20)
        latency = time.perf_counter() - start_time

        assert len(results["documents"]) == 20
        assert latency < 0.5
        print(f"\n  top_k=20: {latency:.3f}s")


class TestConsecutiveQueries:
    """Test performance of consecutive queries (cache effects)."""

    def test_consecutive_identical_queries(self, medium_collection):
        """Test consecutive identical queries (potential caching)."""
        query = "데이터베이스 최적화"
        latencies = []

        for i in range(5):
            start_time = time.perf_counter()
            medium_collection.query(query, top_k=5)
            latency = time.perf_counter() - start_time
            latencies.append(latency)

        # All queries should meet SLA
        assert all(lat < 0.5 for lat in latencies)

        print(f"\n  1st query: {latencies[0]:.3f}s")
        print(f"  2nd query: {latencies[1]:.3f}s")
        print(f"  5th query: {latencies[4]:.3f}s")
        print(f"  Average: {np.mean(latencies):.3f}s")

    def test_consecutive_different_queries(self, medium_collection):
        """Test consecutive different queries."""
        queries = [
            "인덱스 최적화",
            "트랜잭션 처리",
            "쿼리 성능",
            "데이터 정규화",
            "성능 모니터링"
        ]

        latencies = []
        for query in queries:
            start_time = time.perf_counter()
            medium_collection.query(query, top_k=5)
            latency = time.perf_counter() - start_time
            latencies.append(latency)

        assert all(lat < 0.5 for lat in latencies)
        print(f"\n  Average latency: {np.mean(latencies):.3f}s")
        print(f"  Std deviation: {np.std(latencies):.3f}s")


class TestPerformanceStatistics:
    """Collect comprehensive performance statistics."""

    def test_performance_percentiles(self, medium_collection):
        """Test latency percentiles (p50, p95, p99)."""
        queries = [f"데이터베이스 문서 {i}" for i in range(100)]
        latencies = []

        for query in queries:
            start_time = time.perf_counter()
            medium_collection.query(query, top_k=5)
            latency = time.perf_counter() - start_time
            latencies.append(latency)

        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)

        # SLA: p95 should be < 0.5s
        assert p95 < 0.5, f"p95 latency {p95:.3f}s exceeds SLA"

        print(f"\n  p50 (median): {p50:.3f}s")
        print(f"  p95: {p95:.3f}s")
        print(f"  p99: {p99:.3f}s")
        print(f"  Mean: {np.mean(latencies):.3f}s")
        print(f"  Std: {np.std(latencies):.3f}s")

    def test_throughput(self, medium_collection):
        """Test query throughput (queries per second)."""
        num_queries = 50
        queries = [f"검색 쿼리 {i}" for i in range(num_queries)]

        start_time = time.perf_counter()
        for query in queries:
            medium_collection.query(query, top_k=5)
        total_time = time.perf_counter() - start_time

        throughput = num_queries / total_time

        # Should handle at least 2 queries per second
        assert throughput >= 2.0, f"Throughput {throughput:.2f} q/s too low"

        print(f"\n  Throughput: {throughput:.2f} queries/second")
        print(f"  Total time for {num_queries} queries: {total_time:.2f}s")
        print(f"  Average latency: {total_time/num_queries:.3f}s")

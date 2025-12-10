"""
Top-5 Embedding Accuracy Benchmark Tests

Tests Korean search accuracy ≥90% (Top-5) using Hit@5 metric.
Validates that relevant documents appear in top-5 search results.
"""

import json
import pytest
from pathlib import Path
from typing import List, Dict, Tuple

from src.models.embedding import EmbeddingConfiguration
from src.services.embeddings import HuggingFaceEmbedding
from src.services.vector_store import VectorStore


@pytest.fixture(scope="class")
def benchmark_data():
    """Load benchmark queries and documents."""
    data_dir = Path(__file__).parent / "data"

    with open(data_dir / "queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)

    with open(data_dir / "documents.json", "r", encoding="utf-8") as f:
        documents = json.load(f)

    return queries, documents


@pytest.fixture(scope="class")
def setup_benchmark(benchmark_data):
    """Setup vector store with benchmark documents."""
    queries, documents = benchmark_data

    # Initialize embedding service
    config = EmbeddingConfiguration()
    embedding_service = HuggingFaceEmbedding(config)

    # Initialize vector store
    vector_store = VectorStore(
        embedding_service=embedding_service,
        collection_name="benchmark_accuracy"
    )

    # Index documents with IDs
    texts = [doc["text"] for doc in documents]
    ids = [doc["id"] for doc in documents]
    metadatas = [
        {
            "category": doc["category"],
            "language": doc["language"]
        }
        for doc in documents
    ]

    vector_store.add_documents(
        documents=texts,
        ids=ids,
        metadatas=metadatas
    )

    yield vector_store, queries

    # Cleanup
    vector_store.delete_collection()


def calculate_hit_at_k(
    result_ids: List[str],
    answer_ids: List[str],
    k: int = 5
) -> bool:
    """
    Calculate Hit@K metric.

    Returns True if any answer_id appears in top-k results.

    Args:
        result_ids: List of retrieved document IDs (top-k)
        answer_ids: List of relevant document IDs
        k: Top-k results to consider

    Returns:
        True if hit, False otherwise
    """
    top_k_results = set(result_ids[:k])
    relevant_docs = set(answer_ids)

    return len(top_k_results & relevant_docs) > 0


def calculate_accuracy(hits: List[bool]) -> float:
    """Calculate accuracy from list of hit results."""
    if not hits:
        return 0.0
    return sum(hits) / len(hits)


class TestEmbeddingAccuracy:
    """임베딩 검색 정확도 벤치마크"""

    def test_overall_top5_accuracy(self, setup_benchmark):
        """전체 Top-5 정확도 테스트 (목표: ≥90%)"""
        vector_store, queries = setup_benchmark

        hits = []

        for query_data in queries:
            query_text = query_data["query"]
            answer_ids = query_data["answer_ids"]

            # Search
            results = vector_store.query(query_text, top_k=5)
            result_ids = results["ids"]

            # Calculate Hit@5
            hit = calculate_hit_at_k(result_ids, answer_ids, k=5)
            hits.append(hit)

        accuracy = calculate_accuracy(hits)

        print(f"\n  Overall Top-5 Accuracy: {accuracy:.1%} ({sum(hits)}/{len(hits)})")
        print(f"  Target: ≥90%")

        assert accuracy >= 0.90, f"Overall accuracy {accuracy:.1%} below target 90%"

    def test_korean_query_accuracy(self, setup_benchmark):
        """한국어 쿼리 Top-5 정확도 (목표: ≥90%)"""
        vector_store, queries = setup_benchmark

        # Filter Korean queries
        korean_queries = [q for q in queries if q["language"] == "korean"]

        hits = []

        for query_data in korean_queries:
            query_text = query_data["query"]
            answer_ids = query_data["answer_ids"]

            results = vector_store.query(query_text, top_k=5)
            result_ids = results["ids"]

            hit = calculate_hit_at_k(result_ids, answer_ids, k=5)
            hits.append(hit)

        accuracy = calculate_accuracy(hits)

        print(f"\n  Korean Query Top-5 Accuracy: {accuracy:.1%} ({sum(hits)}/{len(hits)})")
        print(f"  Total Korean queries: {len(korean_queries)}")

        assert accuracy >= 0.90, f"Korean accuracy {accuracy:.1%} below target 90%"

    def test_english_query_accuracy(self, setup_benchmark):
        """영어 쿼리 Top-5 정확도 (참고용)"""
        vector_store, queries = setup_benchmark

        # Filter English queries
        english_queries = [q for q in queries if q["language"] == "english"]

        if not english_queries:
            pytest.skip("No English queries in benchmark data")

        hits = []

        for query_data in english_queries:
            query_text = query_data["query"]
            answer_ids = query_data["answer_ids"]

            results = vector_store.query(query_text, top_k=5)
            result_ids = results["ids"]

            hit = calculate_hit_at_k(result_ids, answer_ids, k=5)
            hits.append(hit)

        accuracy = calculate_accuracy(hits)

        print(f"\n  English Query Top-5 Accuracy: {accuracy:.1%} ({sum(hits)}/{len(hits)})")
        print(f"  Total English queries: {len(english_queries)}")

    def test_category_based_accuracy(self, setup_benchmark):
        """카테고리별 Top-5 정확도 분석"""
        vector_store, queries = setup_benchmark

        # Group by category
        category_results = {}

        for query_data in queries:
            category = query_data["category"]
            query_text = query_data["query"]
            answer_ids = query_data["answer_ids"]

            results = vector_store.query(query_text, top_k=5)
            result_ids = results["ids"]

            hit = calculate_hit_at_k(result_ids, answer_ids, k=5)

            if category not in category_results:
                category_results[category] = []
            category_results[category].append(hit)

        print(f"\n  Category-based Accuracy:")
        for category, hits in sorted(category_results.items()):
            accuracy = calculate_accuracy(hits)
            print(f"  - {category}: {accuracy:.1%} ({sum(hits)}/{len(hits)})")

    def test_top1_accuracy(self, setup_benchmark):
        """Top-1 정확도 (참고용 - 가장 엄격한 기준)"""
        vector_store, queries = setup_benchmark

        hits = []

        for query_data in queries:
            query_text = query_data["query"]
            answer_ids = query_data["answer_ids"]

            results = vector_store.query(query_text, top_k=1)
            result_ids = results["ids"]

            hit = calculate_hit_at_k(result_ids, answer_ids, k=1)
            hits.append(hit)

        accuracy = calculate_accuracy(hits)

        print(f"\n  Top-1 Accuracy (strict): {accuracy:.1%} ({sum(hits)}/{len(hits)})")

    def test_top10_accuracy(self, setup_benchmark):
        """Top-10 정확도 (참고용 - 완화된 기준)"""
        vector_store, queries = setup_benchmark

        hits = []

        for query_data in queries:
            query_text = query_data["query"]
            answer_ids = query_data["answer_ids"]

            results = vector_store.query(query_text, top_k=10)
            result_ids = results["ids"]

            hit = calculate_hit_at_k(result_ids, answer_ids, k=10)
            hits.append(hit)

        accuracy = calculate_accuracy(hits)

        print(f"\n  Top-10 Accuracy (relaxed): {accuracy:.1%} ({sum(hits)}/{len(hits)})")

    def test_multilingual_cross_language_search(self, setup_benchmark):
        """다국어 교차 검색 성능 (한국어 쿼리 → 영어 문서)"""
        vector_store, queries = setup_benchmark

        # Korean queries that might match English documents
        korean_queries = [q for q in queries if q["language"] == "korean"]

        cross_language_hits = []

        for query_data in korean_queries:
            query_text = query_data["query"]
            answer_ids = query_data["answer_ids"]

            results = vector_store.query(query_text, top_k=10)

            # Check if any English documents are in results
            english_docs_found = any(
                meta.get("language") == "english"
                for meta in results["metadatas"]
            )

            # Check if semantic match (any answer found)
            hit = calculate_hit_at_k(results["ids"], answer_ids, k=10)

            if hit:
                cross_language_hits.append(1)

        if cross_language_hits:
            cross_lang_accuracy = len(cross_language_hits) / len(korean_queries)
            print(f"\n  Cross-language retrieval rate: {cross_lang_accuracy:.1%}")
            print(f"  (Korean queries finding semantically related documents)")

    def test_accuracy_by_answer_count(self, setup_benchmark):
        """정답 문서 개수별 정확도 분석"""
        vector_store, queries = setup_benchmark

        # Group by number of answer documents
        answer_count_results = {}

        for query_data in queries:
            query_text = query_data["query"]
            answer_ids = query_data["answer_ids"]
            answer_count = len(answer_ids)

            results = vector_store.query(query_text, top_k=5)
            result_ids = results["ids"]

            hit = calculate_hit_at_k(result_ids, answer_ids, k=5)

            if answer_count not in answer_count_results:
                answer_count_results[answer_count] = []
            answer_count_results[answer_count].append(hit)

        print(f"\n  Accuracy by Answer Count:")
        for count, hits in sorted(answer_count_results.items()):
            accuracy = calculate_accuracy(hits)
            print(f"  - {count} answer(s): {accuracy:.1%} ({sum(hits)}/{len(hits)} queries)")

    def test_detailed_failure_analysis(self, setup_benchmark):
        """정확도 실패 케이스 상세 분석"""
        vector_store, queries = setup_benchmark

        failures = []

        for query_data in queries:
            query_text = query_data["query"]
            answer_ids = query_data["answer_ids"]
            category = query_data["category"]
            language = query_data["language"]

            results = vector_store.query(query_text, top_k=5)
            result_ids = results["ids"]

            hit = calculate_hit_at_k(result_ids, answer_ids, k=5)

            if not hit:
                failures.append({
                    "query": query_text,
                    "category": category,
                    "language": language,
                    "expected": answer_ids,
                    "got": result_ids[:5]
                })

        if failures:
            print(f"\n  Failure Analysis:")
            print(f"  Total failures: {len(failures)}")
            print(f"  Failure rate: {len(failures)/len(queries):.1%}")

            # Group failures by category
            category_failures = {}
            for fail in failures:
                cat = fail["category"]
                category_failures[cat] = category_failures.get(cat, 0) + 1

            print(f"  Failures by category:")
            for cat, count in sorted(category_failures.items(), key=lambda x: -x[1]):
                print(f"    - {cat}: {count}")

            # Show sample failures
            print(f"\n  Sample failure cases (first 3):")
            for i, fail in enumerate(failures[:3], 1):
                print(f"\n  {i}. Query: {fail['query']}")
                print(f"     Expected IDs: {fail['expected']}")
                print(f"     Got IDs: {fail['got']}")

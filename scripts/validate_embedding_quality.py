"""
Embedding 품질 검증 스크립트

임베딩 모델의 검색 정확도와 성능을 평가합니다.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.services.embedding import HuggingFaceEmbedding
from src.utils.logging import logger


# 평가용 테스트 쿼리 및 기대 결과
TEST_QUERIES = [
    {
        "query": "회원가입하는 방법을 알려주세요",
        "expected_doc_ids": ["doc_001"],  # 회원가입 가이드
        "expected_categories": ["계정관리"],
        "min_relevance": 0.7,
    },
    {
        "query": "비밀번호를 잊어버렸어요",
        "expected_doc_ids": ["doc_002"],  # 비밀번호 재설정 방법
        "expected_categories": ["계정관리"],
        "min_relevance": 0.7,
    },
    {
        "query": "주문 취소하고 반품하려면 어떻게 해야 하나요?",
        "expected_doc_ids": ["doc_004"],  # 반품 및 교환 정책
        "expected_categories": ["반품/교환"],
        "min_relevance": 0.6,
    },
    {
        "query": "상품 결제 방법이 궁금합니다",
        "expected_doc_ids": ["doc_003"],  # 주문 및 결제 가이드
        "expected_categories": ["주문/결제"],
        "min_relevance": 0.6,
    },
    {
        "query": "고객센터에 문의하고 싶어요",
        "expected_doc_ids": ["doc_005"],  # 고객센터 이용 안내
        "expected_categories": ["고객지원"],
        "min_relevance": 0.7,
    },
    {
        "query": "이메일 인증은 어떻게 하나요?",
        "expected_doc_ids": ["doc_001"],  # 회원가입 가이드 (이메일 인증 포함)
        "expected_categories": ["계정관리"],
        "min_relevance": 0.5,
    },
    {
        "query": "신용카드로 결제할 수 있나요?",
        "expected_doc_ids": ["doc_003"],  # 주문 및 결제 가이드
        "expected_categories": ["주문/결제"],
        "min_relevance": 0.5,
    },
]


class EmbeddingQualityValidator:
    """임베딩 품질 검증 클래스"""

    def __init__(self):
        """임베딩 서비스 초기화"""
        self.embedding_service = HuggingFaceEmbedding()
        self.results: List[Dict] = []

    def validate_retrieval_accuracy(self, top_k: int = 3) -> Dict:
        """
        검색 정확도 평가

        Args:
            top_k: 상위 몇 개의 문서를 검색할지

        Returns:
            정확도 평가 결과
        """
        print("=" * 80)
        print(f" Retrieval Accuracy Validation (top_k={top_k})")
        print("=" * 80)
        print()

        total_queries = len(TEST_QUERIES)
        correct_retrievals = 0
        relevance_scores = []

        for i, test_case in enumerate(TEST_QUERIES, 1):
            query = test_case["query"]
            expected_doc_ids = test_case["expected_doc_ids"]
            expected_categories = test_case["expected_categories"]
            min_relevance = test_case["min_relevance"]

            print(f"Test {i}/{total_queries}: {query}")

            # Perform search
            start_time = time.time()
            results = self.embedding_service.search(query_text=query, top_k=top_k)
            search_time = time.time() - start_time

            if not results or not results.get("documents") or not results["documents"][0]:
                print("  ❌ No results found")
                self.results.append(
                    {
                        "query": query,
                        "success": False,
                        "reason": "No results",
                        "search_time": search_time,
                    }
                )
                print()
                continue

            # Check if expected document is in top results
            retrieved_ids = results["ids"][0]
            retrieved_metadatas = results["metadatas"][0]
            distances = results["distances"][0] if results.get("distances") else []

            # Calculate relevance scores (1 - distance)
            relevance_scores_query = [1.0 - d for d in distances] if distances else []

            # Check if any expected document is retrieved
            found_expected = any(doc_id in retrieved_ids for doc_id in expected_doc_ids)

            # Check category match
            retrieved_categories = [
                meta.get("category", "") for meta in retrieved_metadatas
            ]
            category_match = any(cat in retrieved_categories for cat in expected_categories)

            # Check relevance threshold
            max_relevance = max(relevance_scores_query) if relevance_scores_query else 0.0
            relevance_ok = max_relevance >= min_relevance

            # Overall success
            success = found_expected and relevance_ok

            if success:
                correct_retrievals += 1
                status = "✅"
            else:
                status = "⚠️"

            print(f"  {status} Expected: {expected_doc_ids[0]} (min relevance: {min_relevance:.2f})")
            print(f"     Retrieved:")
            for j, (doc_id, metadata, score) in enumerate(
                zip(retrieved_ids, retrieved_metadatas, relevance_scores_query), 1
            ):
                title = metadata.get("title", "Unknown")
                category = metadata.get("category", "Unknown")
                match_indicator = "✓" if doc_id in expected_doc_ids else " "
                print(f"       {j}. [{match_indicator}] {title} ({category}) - score: {score:.3f}")

            print(f"     Search time: {search_time:.3f}s")
            print(f"     Category match: {category_match}")

            self.results.append(
                {
                    "query": query,
                    "success": success,
                    "expected_ids": expected_doc_ids,
                    "retrieved_ids": retrieved_ids,
                    "max_relevance": max_relevance,
                    "category_match": category_match,
                    "search_time": search_time,
                }
            )

            if relevance_scores_query:
                relevance_scores.extend(relevance_scores_query)

            print()

        # Calculate overall metrics
        accuracy = correct_retrievals / total_queries
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        avg_search_time = sum(r["search_time"] for r in self.results) / len(self.results)

        print("-" * 80)
        print(" Summary")
        print("-" * 80)
        print(f"Accuracy: {correct_retrievals}/{total_queries} ({accuracy:.1%})")
        print(f"Average Relevance Score: {avg_relevance:.3f}")
        print(f"Average Search Time: {avg_search_time:.3f}s")
        print()

        return {
            "accuracy": accuracy,
            "correct_retrievals": correct_retrievals,
            "total_queries": total_queries,
            "avg_relevance": avg_relevance,
            "avg_search_time": avg_search_time,
        }

    def benchmark_performance(self, num_iterations: int = 10) -> Dict:
        """
        임베딩 및 검색 성능 벤치마크

        Args:
            num_iterations: 반복 횟수

        Returns:
            성능 벤치마크 결과
        """
        print("=" * 80)
        print(f" Performance Benchmark ({num_iterations} iterations)")
        print("=" * 80)
        print()

        # Test embedding speed
        print("🔄 Testing embedding speed...")
        test_texts = [tc["query"] for tc in TEST_QUERIES]
        embedding_times = []

        for i in range(num_iterations):
            start_time = time.time()
            embeddings = self.embedding_service.embed_texts(test_texts)
            embedding_time = time.time() - start_time
            embedding_times.append(embedding_time)

        avg_embedding_time = sum(embedding_times) / len(embedding_times)
        texts_per_second = len(test_texts) / avg_embedding_time

        print(f"✅ Embedding speed:")
        print(f"   Average time: {avg_embedding_time:.3f}s for {len(test_texts)} texts")
        print(f"   Throughput: {texts_per_second:.1f} texts/second")
        print()

        # Test search speed
        print("🔄 Testing search speed...")
        search_times = []

        for i in range(num_iterations):
            start_time = time.time()
            for test_case in TEST_QUERIES:
                self.embedding_service.search(query_text=test_case["query"], top_k=3)
            search_time = time.time() - start_time
            search_times.append(search_time)

        avg_search_time = sum(search_times) / len(search_times)
        searches_per_second = len(TEST_QUERIES) / avg_search_time

        print(f"✅ Search speed:")
        print(f"   Average time: {avg_search_time:.3f}s for {len(TEST_QUERIES)} searches")
        print(f"   Throughput: {searches_per_second:.1f} searches/second")
        print()

        return {
            "embedding": {
                "avg_time": avg_embedding_time,
                "texts_per_second": texts_per_second,
            },
            "search": {
                "avg_time": avg_search_time,
                "searches_per_second": searches_per_second,
            },
        }

    def validate_model_info(self) -> Dict:
        """
        임베딩 모델 정보 검증

        Returns:
            모델 정보
        """
        print("=" * 80)
        print(" Embedding Model Information")
        print("=" * 80)
        print()

        model_name = self.embedding_service.model_name
        embedding_dim = self.embedding_service.embedding_dim
        device = self.embedding_service.device

        print(f"Model Name: {model_name}")
        print(f"Embedding Dimensions: {embedding_dim}")
        print(f"Device: {device}")
        print()

        # Test single embedding
        test_text = "테스트 텍스트입니다"
        embedding = self.embedding_service.embed_text(test_text)

        print(f"Sample Embedding:")
        print(f"  Input: '{test_text}'")
        print(f"  Output shape: {len(embedding)}")
        print(f"  First 5 values: {embedding[:5]}")
        print()

        return {
            "model_name": model_name,
            "embedding_dim": embedding_dim,
            "device": device,
        }


def main():
    """메인 실행 함수"""
    print("\n" + "=" * 80)
    print(" Embedding Quality Validation")
    print(" Hugging Face sentence-transformers")
    print("=" * 80 + "\n")

    try:
        validator = EmbeddingQualityValidator()

        # 1. Model information
        model_info = validator.validate_model_info()

        # 2. Retrieval accuracy
        accuracy_results = validator.validate_retrieval_accuracy(top_k=3)

        # 3. Performance benchmark
        performance_results = validator.benchmark_performance(num_iterations=5)

        # Final report
        print("=" * 80)
        print(" Final Report")
        print("=" * 80)
        print()
        print(f"Model: {model_info['model_name']}")
        print(f"Dimensions: {model_info['embedding_dim']}")
        print(f"Device: {model_info['device']}")
        print()
        print(f"Retrieval Accuracy: {accuracy_results['accuracy']:.1%}")
        print(f"Average Relevance: {accuracy_results['avg_relevance']:.3f}")
        print(f"Average Search Time: {accuracy_results['avg_search_time']:.3f}s")
        print()
        print(f"Embedding Speed: {performance_results['embedding']['texts_per_second']:.1f} texts/s")
        print(f"Search Speed: {performance_results['search']['searches_per_second']:.1f} searches/s")
        print()

        # Quality assessment
        if accuracy_results["accuracy"] >= 0.8:
            print("✅ PASS: Embedding quality is good (≥80% accuracy)")
        elif accuracy_results["accuracy"] >= 0.6:
            print("⚠️ WARNING: Embedding quality needs improvement (60-80% accuracy)")
        else:
            print("❌ FAIL: Embedding quality is poor (<60% accuracy)")

        print("\n" + "=" * 80 + "\n")

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        print(f"\n❌ Error: {e}\n")
        raise


if __name__ == "__main__":
    main()

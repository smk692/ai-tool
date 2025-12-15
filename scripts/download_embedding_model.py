#!/usr/bin/env python3
"""임베딩 모델 사전 다운로드 스크립트.

HuggingFace 모델을 로컬 캐시에 미리 다운로드하여
챗봇 시작 시간을 단축합니다.

Usage:
    python scripts/download_embedding_model.py
    python scripts/download_embedding_model.py [custom-model-name]
"""

import sys
import time

# 기본 모델 (shared/embedding.py와 동일)
DEFAULT_MODEL = "intfloat/multilingual-e5-large-instruct"


def download_model(model_name: str = DEFAULT_MODEL) -> None:
    """임베딩 모델을 다운로드합니다.

    Args:
        model_name: HuggingFace 모델 이름
    """
    print("=" * 60)
    print(f"임베딩 모델 다운로드: {model_name}")
    print("=" * 60)
    print()

    # sentence-transformers 임포트
    print("[1/3] sentence-transformers 라이브러리 로드 중...")
    start = time.time()
    from sentence_transformers import SentenceTransformer

    print(f"      완료 ({time.time() - start:.2f}초)")
    print()

    # 모델 다운로드/로드
    print("[2/3] 모델 다운로드 중... (최초 실행 시 2-5분 소요)")
    print("      모델 크기: ~2.2GB")
    print("      저장 위치: ~/.cache/huggingface/")
    print()
    start = time.time()
    model = SentenceTransformer(model_name)
    download_time = time.time() - start
    print(f"      완료 ({download_time:.2f}초)")
    print()

    # 모델 테스트
    print("[3/3] 모델 테스트 중...")
    start = time.time()

    # E5 모델은 prefix 필요
    test_texts = [
        "query: 안녕하세요. 이것은 테스트 문장입니다.",
        "passage: This is a test sentence in English.",
    ]
    embeddings = model.encode(test_texts, normalize_embeddings=True)

    print(f"      완료 ({time.time() - start:.2f}초)")
    print(f"      임베딩 차원: {model.get_sentence_embedding_dimension()}")
    print(f"      최대 시퀀스 길이: {model.max_seq_length}")
    print()

    print("=" * 60)
    print("✅ 모델 다운로드 완료!")
    print()
    print("이제 'make run-chatbot' 실행 시 빠르게 시작됩니다.")
    print("=" * 60)


def main() -> None:
    """메인 함수."""
    # 커스텀 모델명을 인자로 받을 수 있음
    model_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL

    try:
        download_model(model_name)
    except KeyboardInterrupt:
        print("\n\n중단됨.")
        sys.exit(1)
    except Exception as e:
        print(f"\n오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

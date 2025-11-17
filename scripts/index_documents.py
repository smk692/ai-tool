#!/usr/bin/env python
"""
Document Indexing CLI Utility

Indexes documents from JSON/CSV files into ChromaDB vector store using
Hugging Face embeddings for semantic search.

Usage:
    python scripts/index_documents.py --file data/documents.json
    python scripts/index_documents.py --file data/documents.csv --collection my_docs
    python scripts/index_documents.py --clear documents  # Clear collection
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.embedding import EmbeddingConfiguration
from src.services.embeddings import HuggingFaceEmbedding
from src.services.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_documents_from_json(file_path: Path) -> List[Dict]:
    """
    Load documents from JSON file.

    Expected format:
    [
        {"text": "document content", "metadata": {"key": "value"}},
        {"text": "another document", "metadata": {"category": "test"}}
    ]

    Args:
        file_path: Path to JSON file

    Returns:
        List of document dictionaries with 'text' and optional 'metadata' keys
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of documents")

    logger.info(f"Loaded {len(data)} documents from {file_path}")
    return data


def load_documents_from_csv(file_path: Path) -> List[Dict]:
    """
    Load documents from CSV file.

    Expected columns:
    - 'text' (required): Document content
    - Other columns become metadata

    Args:
        file_path: Path to CSV file

    Returns:
        List of document dictionaries with 'text' and 'metadata' keys
    """
    df = pd.read_csv(file_path)

    if 'text' not in df.columns:
        raise ValueError("CSV file must have a 'text' column")

    documents = []
    for _, row in df.iterrows():
        text = row['text']
        metadata = {k: v for k, v in row.items() if k != 'text' and pd.notna(v)}
        documents.append({'text': text, 'metadata': metadata})

    logger.info(f"Loaded {len(documents)} documents from {file_path}")
    return documents


def index_documents(
    file_path: Path,
    collection_name: str,
    batch_size: int = 100
) -> Dict:
    """
    Index documents from file into ChromaDB vector store.

    Args:
        file_path: Path to input file (JSON or CSV)
        collection_name: Name of ChromaDB collection
        batch_size: Number of documents to process per batch

    Returns:
        Dictionary with indexing results
    """
    # Load documents based on file extension
    if file_path.suffix == '.json':
        documents_data = load_documents_from_json(file_path)
    elif file_path.suffix == '.csv':
        documents_data = load_documents_from_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Use .json or .csv")

    # Extract texts and metadatas
    texts = [doc['text'] for doc in documents_data]
    metadatas = [
        doc.get('metadata', {'source': str(file_path)})
        for doc in documents_data
    ]

    # Ensure all metadata dicts have at least 'source' key
    for i, metadata in enumerate(metadatas):
        if not metadata or len(metadata) == 0:
            metadatas[i] = {'source': str(file_path)}

    # Initialize services
    logger.info("Initializing embedding service...")
    config = EmbeddingConfiguration(batch_size=batch_size)
    embedding_service = HuggingFaceEmbedding(config)

    logger.info(f"Initializing vector store (collection: {collection_name})...")
    vector_store = VectorStore(
        embedding_service=embedding_service,
        collection_name=collection_name
    )

    # Index documents
    logger.info(f"Indexing {len(texts)} documents...")
    result = vector_store.add_documents(
        documents=texts,
        metadatas=metadatas
    )

    # Get collection stats
    stats = vector_store.get_collection_stats()

    logger.info(f"✅ Indexing complete!")
    logger.info(f"   Documents indexed: {result['count']}")
    logger.info(f"   Total documents in collection: {stats['document_count']}")
    logger.info(f"   Collection: {stats['collection_name']}")
    logger.info(f"   Persist directory: {stats['persist_directory']}")

    return {
        'indexed_count': result['count'],
        'total_count': stats['document_count'],
        'collection': stats['collection_name'],
        'ids': result['ids']
    }


def clear_collection(collection_name: str):
    """
    Clear (delete) a ChromaDB collection.

    Args:
        collection_name: Name of collection to delete
    """
    # Initialize minimal services
    config = EmbeddingConfiguration()
    embedding_service = HuggingFaceEmbedding(config)
    vector_store = VectorStore(
        embedding_service=embedding_service,
        collection_name=collection_name
    )

    # Delete collection
    vector_store.delete_collection()
    logger.info(f"✅ Collection '{collection_name}' deleted successfully")


def query_test(collection_name: str, query_text: str, top_k: int = 5):
    """
    Test query against indexed collection.

    Args:
        collection_name: Name of collection to query
        query_text: Query text
        top_k: Number of results to return
    """
    # Initialize services
    config = EmbeddingConfiguration()
    embedding_service = HuggingFaceEmbedding(config)
    vector_store = VectorStore(
        embedding_service=embedding_service,
        collection_name=collection_name
    )

    # Query
    logger.info(f"Query: {query_text}")
    results = vector_store.query(query_text, top_k=top_k)

    logger.info(f"\n✅ Found {len(results['documents'])} results:")
    for i, (doc, dist, meta) in enumerate(zip(
        results['documents'],
        results['distances'],
        results['metadatas']
    )):
        logger.info(f"\n{i+1}. Distance: {dist:.4f}")
        logger.info(f"   Text: {doc[:100]}...")
        logger.info(f"   Metadata: {meta}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Index documents into ChromaDB vector store'
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Index command
    index_parser = subparsers.add_parser('index', help='Index documents from file')
    index_parser.add_argument(
        '--file',
        type=Path,
        required=True,
        help='Path to input file (JSON or CSV)'
    )
    index_parser.add_argument(
        '--collection',
        type=str,
        default='documents',
        help='ChromaDB collection name (default: documents)'
    )
    index_parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for embedding generation (default: 100)'
    )

    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear (delete) a collection')
    clear_parser.add_argument(
        'collection',
        type=str,
        help='Collection name to delete'
    )

    # Query command
    query_parser = subparsers.add_parser('query', help='Test query against collection')
    query_parser.add_argument(
        '--collection',
        type=str,
        default='documents',
        help='Collection name to query (default: documents)'
    )
    query_parser.add_argument(
        '--text',
        type=str,
        required=True,
        help='Query text'
    )
    query_parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of results to return (default: 5)'
    )

    args = parser.parse_args()

    # Execute command
    try:
        if args.command == 'index':
            if not args.file.exists():
                logger.error(f"File not found: {args.file}")
                sys.exit(1)

            index_documents(
                file_path=args.file,
                collection_name=args.collection,
                batch_size=args.batch_size
            )

        elif args.command == 'clear':
            confirm = input(f"Are you sure you want to delete collection '{args.collection}'? (yes/no): ")
            if confirm.lower() == 'yes':
                clear_collection(args.collection)
            else:
                logger.info("Cancelled")

        elif args.command == 'query':
            query_test(
                collection_name=args.collection,
                query_text=args.text,
                top_k=args.top_k
            )

        else:
            parser.print_help()
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

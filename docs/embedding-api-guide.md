# Embedding API Usage Guide

## Quick Start

### Installation

```bash
pip install sentence-transformers>=2.2.0 chromadb>=0.4.0
```

### Basic Usage

```python
from src.services.embeddings import HuggingFaceEmbedding
from src.models.embedding import EmbeddingConfiguration

# Initialize with default configuration
config = EmbeddingConfiguration()
embedding_service = HuggingFaceEmbedding(config)

# Embed single text
vector = embedding_service.embed_text("한국어 텍스트")
print(f"Vector dimension: {len(vector)}")  # 384

# Embed multiple texts (batch processing)
texts = ["데이터베이스", "machine learning", "人工知能"]
vectors = embedding_service.embed_texts(texts)
print(f"Batch size: {len(vectors)}")  # 3
```

## API Reference

### EmbeddingConfiguration

Configuration model for embedding service.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | "paraphrase-multilingual-MiniLM-L12-v2" | Hugging Face model name |
| `model_path` | Path | None | Local model path (optional) |
| `embedding_dim` | int | 384 | Embedding vector dimension |
| `max_seq_length` | int | 512 | Maximum sequence length |
| `device` | DeviceType | DeviceType.CPU | Device for inference (CPU/CUDA/MPS) |
| `batch_size` | int | 100 | Default batch size for processing |
| `normalize_embeddings` | bool | True | L2 normalize output vectors |
| `show_progress_bar` | bool | False | Display progress during batch processing |

#### Example

```python
from src.models.embedding import EmbeddingConfiguration, DeviceType

# Custom configuration
config = EmbeddingConfiguration(
    device=DeviceType.MPS,  # Use Apple Silicon GPU
    batch_size=50,
    show_progress_bar=True
)
```

### HuggingFaceEmbedding

Main embedding service class.

#### `__init__(config: EmbeddingConfiguration)`

Initialize the embedding service with configuration.

```python
embedding_service = HuggingFaceEmbedding(config)
```

#### `embed_text(text: str) -> List[float]`

Generate embedding for a single text.

**Parameters**:
- `text` (str): Input text to embed

**Returns**:
- `List[float]`: 384-dimensional embedding vector (L2 normalized)

**Example**:

```python
# Korean text
korean_vector = embedding_service.embed_text("데이터베이스 최적화")

# English text
english_vector = embedding_service.embed_text("database optimization")

# Calculate similarity
import numpy as np
similarity = np.dot(korean_vector, english_vector)
print(f"Cross-language similarity: {similarity:.3f}")  # ~0.97
```

#### `embed_texts(texts: List[str]) -> List[List[float]]`

Generate embeddings for multiple texts (batch processing).

**Parameters**:
- `texts` (List[str]): List of input texts

**Returns**:
- `List[List[float]]`: List of 384-dimensional embedding vectors

**Example**:

```python
documents = [
    "PostgreSQL 트랜잭션 격리 수준",
    "B-Tree 인덱스 구조",
    "MVCC 동시성 제어"
]

vectors = embedding_service.embed_texts(documents)
print(f"Generated {len(vectors)} vectors")
```

#### `get_model_info() -> Dict[str, Any]`

Get information about the loaded model.

**Returns**:
- `Dict[str, Any]`: Model metadata

**Example**:

```python
info = embedding_service.get_model_info()
print(f"Model: {info['model_name']}")
print(f"Dimensions: {info['embedding_dim']}")
print(f"Max tokens: {info['max_seq_length']}")
```

## Integration with VectorStore

### Basic Integration

```python
from src.services.embeddings import HuggingFaceEmbedding
from src.services.vector_store import VectorStore
from src.models.embedding import EmbeddingConfiguration

# Initialize embedding service
config = EmbeddingConfiguration()
embedding_service = HuggingFaceEmbedding(config)

# Initialize vector store with embedding service
vector_store = VectorStore(
    embedding_service=embedding_service,
    collection_name="my_documents"
)

# Add documents
documents = [
    "PostgreSQL은 ACID 속성을 완벽하게 지원합니다.",
    "B-Tree 인덱스는 범위 검색에 최적화되어 있습니다."
]

vector_store.add_documents(
    documents=documents,
    ids=["doc_001", "doc_002"],
    metadatas=[
        {"category": "database", "language": "korean"},
        {"category": "performance", "language": "korean"}
    ]
)

# Search
results = vector_store.query("트랜잭션 격리 수준", top_k=5)
print(f"Found {len(results['documents'])} results")
```

### Advanced Usage with Metadata Filtering

```python
# Search with metadata filter
results = vector_store.query(
    "데이터베이스 인덱스",
    top_k=10,
    where={"category": "performance"}
)

# Get collection statistics
stats = vector_store.get_collection_stats()
print(f"Total documents: {stats['count']}")
```

## Best Practices

### 1. Batch Processing

**Always use `embed_texts()` for multiple documents:**

```python
# ❌ BAD: Sequential processing
vectors = []
for doc in documents:
    vectors.append(embedding_service.embed_text(doc))

# ✅ GOOD: Batch processing
vectors = embedding_service.embed_texts(documents)
```

**Performance improvement**: 5-10x faster for batches of 100+ documents.

### 2. Error Handling

**Validate input text length and handle errors:**

```python
def safe_embed(embedding_service, text: str, max_length: int = 10000):
    """Safely embed text with validation."""
    if not text or len(text) == 0:
        raise ValueError("Empty text provided")

    if len(text) > max_length:
        # Truncate or split long text
        text = text[:max_length]

    try:
        return embedding_service.embed_text(text)
    except Exception as e:
        print(f"Embedding failed: {e}")
        return None
```

### 3. Performance Optimization

**Adjust batch size based on available memory:**

```python
import psutil

# Calculate optimal batch size
available_memory_gb = psutil.virtual_memory().available / (1024**3)

if available_memory_gb > 16:
    batch_size = 200
elif available_memory_gb > 8:
    batch_size = 100
else:
    batch_size = 50

config = EmbeddingConfiguration(batch_size=batch_size)
```

### 4. GPU Acceleration

**Enable GPU if available:**

```python
import torch
from src.models.embedding import DeviceType

# Auto-detect best device
if torch.cuda.is_available():
    device = DeviceType.CUDA
elif torch.backends.mps.is_available():  # Apple Silicon
    device = DeviceType.MPS
else:
    device = DeviceType.CPU

config = EmbeddingConfiguration(device=device)
print(f"Using device: {device}")
```

### 5. Caching Embeddings

**Cache embeddings for frequently used texts:**

```python
from functools import lru_cache

class CachedEmbeddingService:
    def __init__(self, embedding_service):
        self.service = embedding_service
        self._cache = {}

    def embed_text(self, text: str):
        if text not in self._cache:
            self._cache[text] = self.service.embed_text(text)
        return self._cache[text]

    def clear_cache(self):
        self._cache.clear()
```

## Common Use Cases

### 1. Semantic Search

```python
# Index documents
docs = [
    "PostgreSQL ACID properties ensure data consistency",
    "B-Tree indexes optimize range queries",
    "MVCC enables concurrent transactions"
]

vector_store.add_documents(docs)

# Search with natural language query
query = "How does PostgreSQL handle concurrent access?"
results = vector_store.query(query, top_k=3)

for i, doc in enumerate(results['documents'], 1):
    print(f"{i}. {doc}")
```

### 2. Document Similarity

```python
# Find similar documents
doc1 = "데이터베이스 정규화"
doc2 = "Database normalization"

v1 = embedding_service.embed_text(doc1)
v2 = embedding_service.embed_text(doc2)

similarity = np.dot(v1, v2)
print(f"Similarity: {similarity:.3f}")  # ~0.97 (high similarity)
```

### 3. Multilingual Search

```python
# Add multilingual documents
docs = [
    "데이터베이스 인덱스",  # Korean
    "database index",       # English
    "データベースインデックス",  # Japanese
    "数据库索引"            # Chinese
]

vector_store.add_documents(docs, ids=[f"doc_{i}" for i in range(4)])

# Search in any language
query = "索引"  # Chinese query
results = vector_store.query(query, top_k=4)
# Returns all related documents regardless of language
```

### 4. Duplicate Detection

```python
def find_duplicates(documents, threshold=0.95):
    """Find duplicate documents using cosine similarity."""
    vectors = embedding_service.embed_texts(documents)
    duplicates = []

    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            similarity = np.dot(vectors[i], vectors[j])
            if similarity >= threshold:
                duplicates.append((i, j, similarity))

    return duplicates

# Example
docs = [
    "PostgreSQL database",
    "PostgreSQL database system",  # Near duplicate
    "MongoDB NoSQL database"
]

dupes = find_duplicates(docs)
for i, j, sim in dupes:
    print(f"Doc {i} ~ Doc {j}: {sim:.3f}")
```

### 5. Clustering Documents

```python
from sklearn.cluster import KMeans

# Embed documents
documents = [...]  # Your documents
vectors = embedding_service.embed_texts(documents)

# Cluster
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(vectors)

# Group by cluster
clusters = {}
for idx, label in enumerate(labels):
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(documents[idx])

for cluster_id, docs in clusters.items():
    print(f"\nCluster {cluster_id}:")
    for doc in docs[:3]:  # Show first 3
        print(f"  - {doc}")
```

## Performance Tips

### Optimizing for Large Datasets

```python
# Process large datasets in chunks
def embed_large_dataset(documents, chunk_size=1000):
    all_vectors = []

    for i in range(0, len(documents), chunk_size):
        chunk = documents[i:i + chunk_size]
        vectors = embedding_service.embed_texts(chunk)
        all_vectors.extend(vectors)

        print(f"Processed {min(i + chunk_size, len(documents))}/{len(documents)}")

    return all_vectors
```

### Memory Management

```python
import gc

# Clear memory after large operations
vectors = embedding_service.embed_texts(large_document_list)
# ... use vectors ...

# Clean up
del large_document_list
gc.collect()
```

## Testing Your Integration

```python
def test_embedding_service():
    """Basic integration test."""
    config = EmbeddingConfiguration()
    service = HuggingFaceEmbedding(config)

    # Test single embedding
    vector = service.embed_text("test")
    assert len(vector) == 384
    assert abs(np.linalg.norm(vector) - 1.0) < 1e-6  # L2 normalized

    # Test batch embedding
    vectors = service.embed_texts(["test1", "test2"])
    assert len(vectors) == 2

    # Test multilingual
    korean = service.embed_text("테스트")
    english = service.embed_text("test")
    similarity = np.dot(korean, english)
    assert similarity > 0.5  # Should be similar

    print("✅ All tests passed")

test_embedding_service()
```

## Next Steps

- Review [Embedding Model Specification](embedding-model.md) for technical details
- Check [Troubleshooting Guide](embedding-troubleshooting.md) for common issues
- Read [FAQ](embedding-faq.md) for frequently asked questions

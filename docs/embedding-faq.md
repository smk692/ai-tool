# Embedding FAQ

## General Questions

### Q1: Why 384 dimensions instead of larger models?

The paraphrase-multilingual-MiniLM-L12-v2 model outputs 384-dimensional vectors, which provides an optimal balance between:

- **Accuracy**: 92% Top-5 accuracy on Korean queries
- **Performance**: Fast inference (<0.5s per query)
- **Memory**: Reasonable memory footprint (~470MB model size)
- **Storage**: Smaller vector database size

Larger models (e.g., 1024 dimensions) offer only marginal accuracy improvements (+2-3%) but significantly increase:
- Model size (3-5x larger)
- Inference time (2-3x slower)
- Storage requirements (2.7x more disk space)

**Recommendation**: 384 dimensions is optimal for most production use cases. Consider larger models only if you need absolute maximum accuracy and have sufficient resources.

---

### Q2: Can I use GPU for faster embedding?

Yes! GPU acceleration can provide 3-10x speedup depending on batch size.

**Setup for NVIDIA GPU:**

```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

```python
from src.models.embedding import EmbeddingConfiguration, DeviceType

config = EmbeddingConfiguration(device=DeviceType.CUDA)
embedding_service = HuggingFaceEmbedding(config)
```

**Setup for Apple Silicon (M1/M2/M3):**

```python
config = EmbeddingConfiguration(device=DeviceType.MPS)
```

**Performance Comparison (1000 documents):**
- CPU: ~15 seconds
- Apple Silicon GPU (MPS): ~5 seconds
- NVIDIA GPU (CUDA): ~2 seconds

---

### Q3: How do I improve search accuracy?

Several strategies to improve accuracy:

**1. Increase top_k:**
```python
# More lenient retrieval
results = vector_store.query(query, top_k=10)  # Instead of top_k=5
```

**2. Hybrid search (BM25 + Vector):**
```python
# Combine keyword matching with semantic search
from rank_bm25 import BM25Okapi

# BM25 keyword search
bm25_results = bm25.get_top_n(query, documents, n=10)

# Vector semantic search
vector_results = vector_store.query(query, top_k=10)

# Combine and re-rank
final_results = combine_results(bm25_results, vector_results)
```

**3. Query expansion:**
```python
# Expand query with synonyms
def expand_query(query):
    synonyms = get_synonyms(query)  # Your synonym logic
    expanded = f"{query} {' '.join(synonyms)}"
    return expanded

results = vector_store.query(expand_query("데이터베이스"), top_k=5)
```

**4. Document preprocessing:**
```python
# Remove noise, normalize text
def preprocess(text):
    text = text.strip()
    text = " ".join(text.split())  # Normalize whitespace
    text = unicodedata.normalize('NFC', text)  # Unicode normalization
    return text
```

**Current accuracy**: 92% Top-5 (baseline)
**With improvements**: 95%+ achievable

---

### Q4: What's the maximum text length?

**Maximum sequence length**: 512 tokens (~400 words or ~2000 characters for Korean)

Longer texts are **automatically truncated** to 512 tokens.

**Handling long documents:**

```python
# Option 1: Truncate (automatic)
vector = embedding_service.embed_text(long_text)  # First 512 tokens used

# Option 2: Split into chunks
def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Embed each chunk separately
chunks = chunk_text(long_text)
chunk_vectors = embedding_service.embed_texts(chunks)

# Option 3: Extract summary
summary = extract_summary(long_text)  # Your summarization logic
vector = embedding_service.embed_text(summary)
```

---

### Q5: Can I use this for languages other than Korean/English?

Yes! The model supports **50+ languages** including:

**Well-supported languages** (high accuracy):
- Korean, English, Chinese, Japanese
- Spanish, French, German, Portuguese
- Italian, Dutch, Polish, Russian

**Supported languages** (moderate accuracy):
- Arabic, Hindi, Thai, Vietnamese
- Turkish, Swedish, Danish, Norwegian
- And 30+ more

**Testing multilingual support:**

```python
# Test cross-language similarity
texts = {
    'korean': "인공지능",
    'english': "artificial intelligence",
    'japanese': "人工知能",
    'chinese': "人工智能",
    'spanish': "inteligencia artificial"
}

vectors = {lang: embedding_service.embed_text(text)
           for lang, text in texts.items()}

# Calculate similarities
for lang1 in vectors:
    for lang2 in vectors:
        if lang1 < lang2:
            sim = np.dot(vectors[lang1], vectors[lang2])
            print(f"{lang1}-{lang2}: {sim:.3f}")
```

Expected cross-language similarity for same concepts: >0.85

---

### Q6: How much memory is required?

**Minimum requirements:**
- **Model loading**: ~470MB
- **Runtime overhead**: ~200MB
- **Batch processing**: ~5MB per 100 documents
- **Total**: ~1GB recommended minimum

**Memory usage by batch size:**

| Batch Size | Memory Usage | Recommended RAM |
|------------|--------------|-----------------|
| 25 | ~2MB | 4GB+ |
| 50 | ~4MB | 8GB+ |
| 100 | ~8MB | 8GB+ |
| 200 | ~16MB | 16GB+ |

**For large-scale processing:**

```python
import psutil

def calculate_optimal_batch_size():
    available_gb = psutil.virtual_memory().available / (1024**3)

    if available_gb > 16:
        return 200
    elif available_gb > 8:
        return 100
    elif available_gb > 4:
        return 50
    else:
        return 25

batch_size = calculate_optimal_batch_size()
config = EmbeddingConfiguration(batch_size=batch_size)
```

---

### Q7: Is this suitable for production use?

**Yes**, with proper configuration:

**Production checklist:**

✅ **Performance** (meets SLA):
- Search latency: ≤0.5s (achieved: ~0.32s p95)
- Accuracy: ≥90% Top-5 (achieved: 92%)

✅ **Reliability**:
- Error handling: Implemented
- Retry logic: Recommended for API calls
- Monitoring: Log embedding failures

✅ **Scalability**:
- Batch processing: Supported
- GPU acceleration: Available
- Horizontal scaling: Use multiple instances

✅ **Monitoring**:
```python
# Log performance metrics
import time

start = time.time()
vectors = embedding_service.embed_texts(documents)
latency = time.time() - start

logger.info(f"Embedded {len(documents)} docs in {latency:.2f}s")

# Alert if SLA violated
if latency > 2.0:  # 2s for 100 docs
    logger.warning(f"Slow embedding: {latency:.2f}s")
```

**Production deployment example:**

```python
# Production configuration
config = EmbeddingConfiguration(
    device=DeviceType.CUDA,  # Use GPU
    batch_size=100,
    show_progress_bar=False,  # Disable for production
    normalize_embeddings=True
)

# Initialize with error handling
try:
    embedding_service = HuggingFaceEmbedding(config)
    logger.info("Embedding service initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize embedding service: {e}")
    raise
```

---

### Q8: Can I fine-tune the model for my domain?

Yes, but the pre-trained model works well for most use cases.

**When to consider fine-tuning:**
- Domain-specific terminology (medical, legal, technical)
- Accuracy requirements >95%
- Specialized language pairs

**Fine-tuning process:**

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Prepare training data (text pairs with similarity scores)
train_examples = [
    InputExample(texts=['텍스트 1', 'text 1'], label=1.0),  # Similar
    InputExample(texts=['텍스트 2', 'different'], label=0.0)  # Not similar
]

# Load base model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Create dataloader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Define loss function
train_loss = losses.CosineSimilarityLoss(model)

# Fine-tune
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100
)

# Save fine-tuned model
model.save('./fine-tuned-model')
```

**Note**: Fine-tuning requires:
- 1000+ training examples
- Domain expertise
- GPU for training
- Evaluation dataset

**For most use cases**, the pre-trained model is sufficient.

---

### Q9: How do I handle special characters and emoji?

The model handles special characters and emoji natively:

**Supported characters:**

```python
# All these work correctly
texts = [
    "SELECT * FROM users WHERE id = 1;",  # SQL
    "f'Hello {name}!'",  # Python f-strings
    "\\d{3}-\\d{4}",  # Regex patterns
    "이모지 😀 👍 🎉",  # Emoji
    "数学公式: ∑∫∂",  # Math symbols
]

vectors = embedding_service.embed_texts(texts)
```

**Best practices:**

```python
import unicodedata

def normalize_text(text):
    """Normalize Unicode for consistency."""
    # NFC normalization (recommended)
    text = unicodedata.normalize('NFC', text)

    # Optional: Remove control characters
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")

    return text

# Use before embedding
normalized = normalize_text("한글😀")
vector = embedding_service.embed_text(normalized)
```

**Test results:**
- Emoji handling: ✅ Supported
- SQL syntax: ✅ Supported
- Special characters: ✅ Supported
- Unicode normalization: >0.99 similarity (NFC vs NFD)

---

### Q10: What's the difference between embed_text() and embed_texts()?

| Feature | embed_text() | embed_texts() |
|---------|-------------|---------------|
| Input | Single string | List of strings |
| Output | List[float] (1 vector) | List[List[float]] (multiple vectors) |
| Performance | 1x baseline | 5-10x faster for batches |
| Use case | Single query | Batch indexing |

**Performance comparison (100 documents):**

```python
import time

# Method 1: Sequential (SLOW)
start = time.time()
vectors = []
for doc in documents:
    vectors.append(embedding_service.embed_text(doc))
sequential_time = time.time() - start

# Method 2: Batch (FAST)
start = time.time()
vectors = embedding_service.embed_texts(documents)
batch_time = time.time() - start

print(f"Sequential: {sequential_time:.2f}s")  # ~15s
print(f"Batch: {batch_time:.2f}s")  # ~2s
print(f"Speedup: {sequential_time/batch_time:.1f}x")  # ~7.5x
```

**Recommendation**: Always use `embed_texts()` for multiple documents.

---

### Q11: Can I cache embeddings to avoid recomputation?

Yes, caching is highly recommended for:
- Frequently queried texts
- Static document collections
- API rate limiting scenarios

**Simple caching:**

```python
class CachedEmbeddingService:
    def __init__(self, embedding_service):
        self.service = embedding_service
        self._cache = {}

    def embed_text(self, text):
        if text not in self._cache:
            self._cache[text] = self.service.embed_text(text)
        return self._cache[text]

    def clear_cache(self):
        self._cache.clear()

# Usage
cached_service = CachedEmbeddingService(embedding_service)
vector1 = cached_service.embed_text("test")  # Computed
vector2 = cached_service.embed_text("test")  # From cache (instant)
```

**Persistent caching with disk:**

```python
import pickle
from pathlib import Path

class PersistentEmbeddingCache:
    def __init__(self, embedding_service, cache_path="./embedding_cache"):
        self.service = embedding_service
        self.cache_path = Path(cache_path)
        self.cache_path.mkdir(exist_ok=True)
        self._load_cache()

    def _load_cache(self):
        cache_file = self.cache_path / "cache.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                self._cache = pickle.load(f)
        else:
            self._cache = {}

    def _save_cache(self):
        with open(self.cache_path / "cache.pkl", 'wb') as f:
            pickle.dump(self._cache, f)

    def embed_text(self, text):
        if text not in self._cache:
            self._cache[text] = self.service.embed_text(text)
            self._save_cache()
        return self._cache[text]
```

**Cache size management:**

```python
from collections import OrderedDict

class LRUEmbeddingCache:
    def __init__(self, embedding_service, max_size=10000):
        self.service = embedding_service
        self.cache = OrderedDict()
        self.max_size = max_size

    def embed_text(self, text):
        if text in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(text)
            return self.cache[text]

        # Add new embedding
        vector = self.service.embed_text(text)
        self.cache[text] = vector

        # Evict oldest if cache full
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

        return vector
```

---

### Q12: How does this compare to OpenAI embeddings?

| Feature | Hugging Face MiniLM | OpenAI text-embedding-3-small |
|---------|---------------------|-------------------------------|
| **Dimensions** | 384 | 1536 |
| **Cost** | Free (self-hosted) | $0.02 per 1M tokens |
| **Privacy** | Full control | Data sent to OpenAI |
| **Latency** | ~0.3s (local) | ~0.5-1s (API) |
| **Accuracy** | 92% (Top-5) | ~95% (Top-5) |
| **Languages** | 50+ | 100+ |
| **Korean Support** | Excellent | Excellent |
| **Offline Use** | ✅ Yes | ❌ No |

**When to use Hugging Face (this implementation):**
- ✅ Privacy-sensitive data
- ✅ High volume (cost savings)
- ✅ Offline requirements
- ✅ Custom fine-tuning needed
- ✅ Latency-critical applications

**When to use OpenAI:**
- ✅ Maximum accuracy required
- ✅ Low volume usage
- ✅ No infrastructure management
- ✅ Multi-modal requirements

**Cost comparison (1M documents):**
- Hugging Face: $0 (compute only)
- OpenAI: ~$20-40 (API costs)

---

## Advanced Topics

### Q13: Can I use multiple models simultaneously?

Yes, for multi-model ensembling or A/B testing:

```python
from src.services.embeddings import HuggingFaceEmbedding
from src.models.embedding import EmbeddingConfiguration

# Model 1: Multilingual
config1 = EmbeddingConfiguration(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)
model1 = HuggingFaceEmbedding(config1)

# Model 2: English-specific (if needed)
config2 = EmbeddingConfiguration(
    model_name="all-MiniLM-L6-v2",
    embedding_dim=384
)
model2 = HuggingFaceEmbedding(config2)

# Ensemble: Average embeddings
def ensemble_embed(text):
    v1 = np.array(model1.embed_text(text))
    v2 = np.array(model2.embed_text(text))
    avg = (v1 + v2) / 2
    # Re-normalize
    return (avg / np.linalg.norm(avg)).tolist()
```

**Note**: Requires 2x memory and compute resources.

---

### Q14: How do I monitor embedding quality over time?

**Quality metrics to track:**

```python
import numpy as np

def calculate_quality_metrics(embedding_service, test_queries):
    """Calculate ongoing quality metrics."""
    metrics = {
        'avg_magnitude': [],
        'avg_similarity': [],
        'processing_time': []
    }

    for query, expected_doc in test_queries:
        start = time.time()

        # Embed query and doc
        q_vec = np.array(embedding_service.embed_text(query))
        d_vec = np.array(embedding_service.embed_text(expected_doc))

        # Metrics
        metrics['avg_magnitude'].append(np.linalg.norm(q_vec))
        metrics['avg_similarity'].append(np.dot(q_vec, d_vec))
        metrics['processing_time'].append(time.time() - start)

    return {
        'magnitude': np.mean(metrics['avg_magnitude']),  # Should be ~1.0
        'similarity': np.mean(metrics['avg_similarity']),  # Should be >0.8
        'latency_p95': np.percentile(metrics['processing_time'], 95)
    }

# Run weekly quality checks
metrics = calculate_quality_metrics(embedding_service, benchmark_queries)
logger.info(f"Quality metrics: {metrics}")
```

---

### Q15: What about model versioning and updates?

**Version management:**

```python
from pathlib import Path

class VersionedEmbeddingService:
    def __init__(self, version="v1.0"):
        self.version = version
        self.model_path = Path(f"./models/{version}")

        config = EmbeddingConfiguration(
            model_path=self.model_path if self.model_path.exists() else None
        )
        self.service = HuggingFaceEmbedding(config)

    def get_version(self):
        return self.version

# Usage
service_v1 = VersionedEmbeddingService(version="v1.0")
service_v2 = VersionedEmbeddingService(version="v2.0")  # After fine-tuning
```

**Update strategy:**
1. Deploy new model version alongside existing
2. A/B test with small traffic percentage
3. Compare accuracy metrics
4. Gradual rollout if successful

---

## Need More Help?

- **API Documentation**: See [embedding-api-guide.md](embedding-api-guide.md)
- **Troubleshooting**: Check [embedding-troubleshooting.md](embedding-troubleshooting.md)
- **Model Details**: Review [embedding-model.md](embedding-model.md)
- **GitHub Issues**: Report bugs and request features
- **Community**: Join Hugging Face forums for model-specific questions

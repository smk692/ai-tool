# Embedding Troubleshooting Guide

## Common Issues

### Issue 1: Slow Embedding Generation

#### Symptoms
- Embedding takes >2s for 100 documents
- Progress bar moves very slowly
- High CPU usage during embedding

#### Diagnosis

```python
import time

# Measure embedding time
start = time.time()
vectors = embedding_service.embed_texts(documents[:100])
elapsed = time.time() - start

print(f"Time for 100 documents: {elapsed:.2f}s")
# Expected: <2s on modern hardware
```

#### Solutions

1. **Reduce batch size**:
```python
config = EmbeddingConfiguration(batch_size=50)  # Default is 100
```

2. **Disable progress bar**:
```python
config = EmbeddingConfiguration(show_progress_bar=False)
```

3. **Enable GPU acceleration** (if available):
```python
from src.models.embedding import DeviceType
config = EmbeddingConfiguration(device=DeviceType.CUDA)
```

4. **Check CPU usage**:
```python
import psutil
print(f"CPU Usage: {psutil.cpu_percent()}%")
# If >90%, reduce concurrent processes
```

5. **Process in chunks**:
```python
def embed_in_chunks(texts, chunk_size=100):
    vectors = []
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        vectors.extend(embedding_service.embed_texts(chunk))
    return vectors
```

---

### Issue 2: Out of Memory (OOM)

#### Symptoms
- `MemoryError` during batch processing
- Process killed by OS
- System becomes unresponsive

#### Diagnosis

```python
import psutil

mem = psutil.virtual_memory()
print(f"Available memory: {mem.available / (1024**3):.2f} GB")
print(f"Memory usage: {mem.percent}%")
```

#### Solutions

1. **Reduce batch size**:
```python
# For low memory systems
config = EmbeddingConfiguration(batch_size=25)
```

2. **Process in smaller chunks**:
```python
def embed_with_memory_check(texts, max_memory_percent=80):
    vectors = []
    batch_size = 100

    for i in range(0, len(texts), batch_size):
        # Check memory before processing
        if psutil.virtual_memory().percent > max_memory_percent:
            batch_size = batch_size // 2
            print(f"Reducing batch size to {batch_size}")

        chunk = texts[i:i + batch_size]
        vectors.extend(embedding_service.embed_texts(chunk))

    return vectors
```

3. **Clear unused variables**:
```python
import gc

# After processing large batches
del large_document_list
gc.collect()
```

4. **Use generator pattern**:
```python
def embed_generator(texts, batch_size=100):
    """Generate embeddings without loading all in memory."""
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        yield embedding_service.embed_texts(chunk)

# Usage
for batch_vectors in embed_generator(huge_document_list):
    # Process each batch
    vector_store.add_vectors(batch_vectors)
```

---

### Issue 3: Model Download Failure

#### Symptoms
- `OSError: Can't load model`
- `ConnectionError` during initialization
- Model files not found

#### Diagnosis

```python
from pathlib import Path

model_path = Path.home() / ".cache" / "huggingface"
print(f"Model cache: {model_path}")
print(f"Exists: {model_path.exists()}")
```

#### Solutions

1. **Check internet connection**:
```bash
ping huggingface.co
```

2. **Manual download**:
```python
from sentence_transformers import SentenceTransformer

# Download model explicitly
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print(f"Model downloaded to: {model._model_config.name_or_path}")
```

3. **Use local model path**:
```python
config = EmbeddingConfiguration(
    model_path="/path/to/local/model"
)
```

4. **Set proxy** (if behind firewall):
```bash
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
```

---

### Issue 4: ChromaDB Persistence Error

#### Symptoms
- `ValueError: Chroma is running in in-memory mode`
- Collection not found after restart
- Data loss between sessions

#### Diagnosis

```python
# Check if using persistent client
print(f"Vector store client: {type(vector_store._client)}")
# Should be: chromadb.api.client.Client (PersistentClient)
```

#### Solutions

1. **Use PersistentClient**:
```python
import chromadb

# Correct initialization
client = chromadb.PersistentClient(path="./chroma_db")
```

2. **Verify persistence path**:
```python
from pathlib import Path

db_path = Path("./chroma_db")
db_path.mkdir(exist_ok=True)
print(f"Database path: {db_path.absolute()}")
```

3. **Check file permissions**:
```bash
ls -la ./chroma_db
chmod 755 ./chroma_db
```

---

### Issue 5: Poor Search Accuracy

#### Symptoms
- Irrelevant results in top-5
- Expected documents not returned
- Low similarity scores (<0.5)

#### Diagnosis

```python
# Test with known similar texts
query = "데이터베이스"
doc = "database"

query_vec = embedding_service.embed_text(query)
doc_vec = embedding_service.embed_text(doc)

similarity = np.dot(query_vec, doc_vec)
print(f"Similarity: {similarity:.3f}")
# Expected: >0.9 for translations
```

#### Solutions

1. **Verify text preprocessing**:
```python
# Remove extra whitespace
text = " ".join(text.split())

# Check for encoding issues
print(text.encode('utf-8'))
```

2. **Increase top_k**:
```python
# Try top-10 instead of top-5
results = vector_store.query(query, top_k=10)
```

3. **Check metadata filtering**:
```python
# Remove filters temporarily
results = vector_store.query(query, top_k=5, where=None)
```

4. **Verify collection contents**:
```python
stats = vector_store.get_collection_stats()
print(f"Total documents: {stats['count']}")

# Check sample documents
sample = vector_store.collection.peek(limit=5)
print(f"Sample docs: {sample['documents']}")
```

5. **Re-index documents**:
```python
# Delete and recreate collection
vector_store.delete_collection()
vector_store = VectorStore(
    embedding_service=embedding_service,
    collection_name="my_documents"
)
# Re-add all documents
```

---

### Issue 6: Unicode Encoding Errors

#### Symptoms
- `UnicodeDecodeError` or `UnicodeEncodeError`
- Garbled text in results
- Special characters not preserved

#### Diagnosis

```python
# Check text encoding
text = "한국어 텍스트 😀"
print(f"Encoding: {text.encode('utf-8')}")
```

#### Solutions

1. **Ensure UTF-8 encoding**:
```python
# Read files with explicit encoding
with open('documents.txt', 'r', encoding='utf-8') as f:
    documents = f.readlines()
```

2. **Normalize Unicode**:
```python
import unicodedata

def normalize_text(text):
    # NFC normalization (recommended)
    return unicodedata.normalize('NFC', text)

text = normalize_text("한글")
```

3. **Handle encoding errors gracefully**:
```python
def safe_read(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Try alternate encoding
        with open(filepath, 'r', encoding='latin-1') as f:
            return f.read()
```

---

### Issue 7: GPU Not Detected

#### Symptoms
- Model runs on CPU despite GPU available
- `device=cuda` not working
- Slow performance on GPU machine

#### Diagnosis

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
```

#### Solutions

1. **Install CUDA-enabled PyTorch**:
```bash
# For NVIDIA GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For Apple Silicon (M1/M2)
pip install torch torchvision torchaudio
```

2. **Verify GPU configuration**:
```python
from src.models.embedding import DeviceType

# Auto-detect
if torch.cuda.is_available():
    device = DeviceType.CUDA
elif torch.backends.mps.is_available():
    device = DeviceType.MPS
else:
    device = DeviceType.CPU

config = EmbeddingConfiguration(device=device)
print(f"Using device: {config.device}")
```

3. **Check CUDA version compatibility**:
```bash
nvidia-smi  # Check CUDA version
python -c "import torch; print(torch.version.cuda)"
```

---

### Issue 8: Collection Already Exists Error

#### Symptoms
- `ValueError: Collection already exists`
- Cannot create new collection with same name

#### Diagnosis

```python
# List all collections
collections = vector_store._client.list_collections()
print(f"Existing collections: {[c.name for c in collections]}")
```

#### Solutions

1. **Delete existing collection**:
```python
vector_store.delete_collection()
```

2. **Use unique collection names**:
```python
import datetime

collection_name = f"documents_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
```

3. **Get or create pattern**:
```python
try:
    vector_store = VectorStore(
        embedding_service=embedding_service,
        collection_name="my_documents"
    )
except ValueError:
    # Collection exists, delete and recreate
    client = chromadb.PersistentClient(path="./chroma_db")
    client.delete_collection("my_documents")
    vector_store = VectorStore(
        embedding_service=embedding_service,
        collection_name="my_documents"
    )
```

---

## Debugging Tips

### Enable Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enable sentence-transformers logging
logging.getLogger('sentence_transformers').setLevel(logging.DEBUG)
```

### Profile Performance

```python
import cProfile
import pstats

def profile_embedding():
    profiler = cProfile.Profile()
    profiler.enable()

    # Your code
    vectors = embedding_service.embed_texts(documents)

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumtime')
    stats.print_stats(10)  # Top 10 functions

profile_embedding()
```

### Memory Profiling

```python
from memory_profiler import profile

@profile
def embed_large_batch():
    return embedding_service.embed_texts(large_document_list)

embed_large_batch()
```

## Getting Help

### Check System Requirements

```python
import sys
import torch
import sentence_transformers
import chromadb

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"Sentence Transformers: {sentence_transformers.__version__}")
print(f"ChromaDB: {chromadb.__version__}")
```

Minimum requirements:
- Python: >=3.10
- PyTorch: >=2.0.0
- sentence-transformers: >=2.2.0
- chromadb: >=0.4.0

### Verify Installation

```bash
# Run test suite
python -m pytest tests/unit/test_embeddings.py -v

# Run benchmarks
python -m pytest tests/benchmarks/test_embedding_accuracy.py -v
```

### Report Issues

When reporting issues, include:

1. **Error message and traceback**
2. **System information** (output of system requirements check)
3. **Minimal reproducible example**
4. **Expected vs. actual behavior**

## Additional Resources

- [API Usage Guide](embedding-api-guide.md)
- [Model Specification](embedding-model.md)
- [FAQ](embedding-faq.md)
- [Hugging Face Documentation](https://huggingface.co/sentence-transformers)
- [ChromaDB Documentation](https://docs.trychroma.com/)

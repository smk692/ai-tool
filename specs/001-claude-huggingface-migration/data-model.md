# Data Model: LLM Migration to Claude Code

**Date**: 2025-01-13
**Status**: Complete
**Purpose**: Define core entities, relationships, and validation rules for Claude Code migration

---

## Entity Definitions

### 1. LLM Configuration

**Purpose**: Represents language model settings and API connection parameters

**Attributes**:

| Field | Type | Required | Default | Validation | Description |
|-------|------|----------|---------|------------|-------------|
| `provider` | Enum(String) | Yes | "anthropic" | One of: "anthropic", "openai" (legacy) | LLM provider identifier |
| `model_name` | String | Yes | "claude-3-5-sonnet-20241022" | Non-empty string | Specific model version identifier |
| `api_key` | String (Secret) | Yes | (from env: ANTHROPIC_API_KEY) | Non-empty, secure storage | API authentication credential |
| `temperature` | Float | No | 0.0 | 0.0 ≤ temperature ≤ 1.0 | Sampling temperature (0 = deterministic) |
| `max_tokens` | Integer | No | 4096 | 1 ≤ max_tokens ≤ 200000 | Maximum response tokens |
| `timeout` | Integer | No | 60 | 1 ≤ timeout ≤ 300 | Request timeout in seconds |
| `max_retries` | Integer | No | 3 | 0 ≤ max_retries ≤ 5 | Automatic retry attempts |
| `streaming` | Boolean | No | False | True/False | Enable streaming responses |

**Relationships**:
- Used by: Text-to-SQL Chain, Knowledge Discovery Chain, Router Chain, Multi-turn Chain
- References: None (independent configuration)

**State Transitions**:
```
[Uninitialized] → [Validated] → [Active] → [Error] ↔ [Retry] → [Active]
                       ↓
                   [Deprecated]
```

**Validation Rules**:
1. API key must be valid format (starts with "sk-ant-" for Anthropic)
2. Model name must exist in Anthropic model catalog
3. Timeout must align with SLA requirements (≤60s for Text-to-SQL)
4. Temperature = 0 enforced for SQL generation (deterministic output)

**Example**:
```python
llm_config = LLMConfiguration(
    provider="anthropic",
    model_name="claude-3-5-sonnet-20241022",
    api_key=os.environ["ANTHROPIC_API_KEY"],
    temperature=0.0,
    max_tokens=4096,
    timeout=60,
    max_retries=3,
    streaming=False
)
```

---

### 2. Embedding Configuration

**Purpose**: Represents vector embedding model settings for document retrieval

**Attributes**:

| Field | Type | Required | Default | Validation | Description |
|-------|------|----------|---------|------------|-------------|
| `model_name` | String | Yes | "paraphrase-multilingual-MiniLM-L12-v2" | Non-empty, valid Hugging Face model | Embedding model identifier |
| `model_path` | String | No | (auto-download from HF) | Valid file path or HF repo | Local model path or HF repository |
| `embedding_dim` | Integer | Yes | 384 | Positive integer | Embedding vector dimensions |
| `device` | Enum(String) | No | "cpu" | One of: "cpu", "cuda", "mps" | Inference device |
| `batch_size` | Integer | No | 100 | 1 ≤ batch_size ≤ 1000 | Batch processing size |
| `max_seq_length` | Integer | No | 512 | 1 ≤ length ≤ 8192 | Maximum input sequence length |

**Relationships**:
- Used by: RAG System (document indexing, query embedding)
- References: ChromaDB Collection (embedding storage)

**Validation Rules**:
1. Model must support Korean language
2. Embedding dimensions must match ChromaDB collection
3. Device must be available on system (check torch.cuda.is_available())
4. Batch size optimized for memory constraints

**Example**:
```python
embedding_config = EmbeddingConfiguration(
    model_name="paraphrase-multilingual-MiniLM-L12-v2",
    embedding_dim=384,
    device="cpu",
    batch_size=100,
    max_seq_length=512
)
```

---

### 3. Budget Tracking

**Purpose**: Monitors LLM API usage costs and enforces budget constraints

**Attributes**:

| Field | Type | Required | Default | Validation | Description |
|-------|------|----------|---------|------------|-------------|
| `subscription_cost` | Decimal | Yes | 100.00 | Positive number | Monthly subscription fee (USD) |
| `daily_query_count` | Integer | Yes | 0 | Non-negative | Number of queries today |
| `monthly_query_count` | Integer | Yes | 0 | Non-negative | Number of queries this month |
| `current_month` | Date | Yes | (current month) | Valid YYYY-MM format | Tracking period |
| `budget_limit` | Decimal | Yes | 100.00 | Positive number | Monthly budget ceiling (USD) |
| `alert_thresholds` | List[Decimal] | Yes | [80.0, 90.0, 100.0] | Ascending percentages | Alert trigger points (%) |
| `cost_per_query_estimate` | Decimal | Yes | 0.00 | Non-negative | Estimated cost per query (subscription model: $0) |
| `projected_monthly_cost` | Decimal | Computed | - | - | Projected end-of-month cost |
| `alert_status` | Enum(String) | Computed | "ok" | One of: "ok", "warning", "critical" | Current budget status |
| `last_alert_sent` | DateTime | No | None | Valid timestamp | Last alert notification time |

**Relationships**:
- Monitors: LLM Configuration usage
- Triggers: Alert notifications to administrators

**State Transitions**:
```
[OK] → [Warning (≥80%)] → [Critical (≥90%)] → [Over Budget (≥100%)]
  ↓           ↓                   ↓                      ↓
[Daily Reset] [Alert Sent]    [Alert Sent]        [Service Limit]
```

**Validation Rules**:
1. subscription_cost must match actual Claude Code plan ($100/month)
2. alert_thresholds must be ascending (80% < 90% < 100%)
3. current_month auto-resets on month boundary
4. Alert cooldown: minimum 1 hour between duplicate alerts

**Computed Properties**:
```python
projected_monthly_cost = subscription_cost  # Fixed for subscription model
budget_utilization = (subscription_cost / budget_limit) * 100  # Always 100%
alert_status = "ok"  # Subscription model eliminates overage risk
```

**Example**:
```python
budget_tracker = BudgetTracking(
    subscription_cost=Decimal("100.00"),
    daily_query_count=210,  # Updated in real-time
    monthly_query_count=6300,  # Updated daily
    current_month="2025-01",
    budget_limit=Decimal("100.00"),
    alert_thresholds=[Decimal("80.0"), Decimal("90.0"), Decimal("100.0")],
    cost_per_query_estimate=Decimal("0.00"),  # Subscription plan
)
```

---

### 4. Query Request

**Purpose**: Represents a user query submitted to the AI assistant

**Attributes**:

| Field | Type | Required | Default | Validation | Description |
|-------|------|----------|---------|------------|-------------|
| `query_id` | UUID | Yes | (auto-generated) | Valid UUID v4 | Unique query identifier |
| `user_id` | String | Yes | - | Non-empty | User identifier (Slack user ID) |
| `query_text` | String | Yes | - | 1 ≤ length ≤ 10000 | User's natural language query |
| `query_language` | String | Yes | "ko" | ISO 639-1 code | Query language (ko=Korean) |
| `query_type` | Enum(String) | Computed | - | One of: "text_to_sql", "knowledge", "assistant" | Query classification |
| `timestamp` | DateTime | Yes | (current UTC) | Valid ISO 8601 | Query submission time |
| `session_id` | UUID | No | None | Valid UUID v4 | Conversation session ID |
| `metadata` | JSON | No | {} | Valid JSON | Additional context (channel, thread, etc.) |

**Relationships**:
- Processed by: Router Chain (intent classification)
- Generates: Query Response (1:1)
- Tracks: Budget Tracking (increments query count)

**Validation Rules**:
1. query_text must not be empty or whitespace-only
2. query_language must be supported (ko, en)
3. User authentication required before processing
4. Rate limiting: max 10 queries per user per minute

**Example**:
```python
query_request = QueryRequest(
    query_id=uuid.uuid4(),
    user_id="U12345ABC",
    query_text="지난달 신규 가입자 수는?",
    query_language="ko",
    timestamp=datetime.utcnow(),
    session_id=uuid.uuid4(),
    metadata={"channel": "C123", "thread_ts": "1234567890.123456"}
)
```

---

### 5. Query Response

**Purpose**: Represents the AI assistant's response to a user query

**Attributes**:

| Field | Type | Required | Default | Validation | Description |
|-------|------|----------|---------|------------|-------------|
| `response_id` | UUID | Yes | (auto-generated) | Valid UUID v4 | Unique response identifier |
| `query_id` | UUID | Yes | - | Valid UUID v4 | Reference to original query |
| `response_text` | String | Yes | - | Non-empty | Generated response content |
| `response_type` | Enum(String) | Yes | - | One of: "sql_query", "document_answer", "assistant_message", "error" | Response classification |
| `sql_query` | String | No | None | Valid SQL syntax | Generated SQL (if type=sql_query) |
| `source_documents` | List[Dict] | No | [] | Valid document references | Retrieved documents (if type=document_answer) |
| `confidence_score` | Decimal | Yes | - | 0.0 ≤ score ≤ 1.0 | Response confidence |
| `execution_time` | Float | Yes | - | Positive number | Response generation time (seconds) |
| `token_usage` | Dict | Yes | - | Valid token counts | Input/output token usage |
| `timestamp` | DateTime | Yes | (current UTC) | Valid ISO 8601 | Response generation time |
| `error_message` | String | No | None | - | Error details (if type=error) |

**Relationships**:
- References: Query Request (1:1)
- Logs: Performance metrics

**Validation Rules**:
1. response_text must not be empty
2. sql_query must be valid PostgreSQL syntax (if present)
3. execution_time must not exceed SLA (60s for Text-to-SQL, 3s for Knowledge)
4. confidence_score used for quality monitoring

**Example (Text-to-SQL)**:
```python
response = QueryResponse(
    response_id=uuid.uuid4(),
    query_id=query_request.query_id,
    response_text="다음 SQL 쿼리를 실행하세요:\nSELECT COUNT(*) FROM users WHERE created_at >= '2024-12-01'",
    response_type="sql_query",
    sql_query="SELECT COUNT(*) FROM users WHERE created_at >= '2024-12-01' AND created_at < '2025-01-01'",
    confidence_score=Decimal("0.92"),
    execution_time=42.5,
    token_usage={"input": 6500, "output": 150},
    timestamp=datetime.utcnow()
)
```

**Example (Knowledge Discovery)**:
```python
response = QueryResponse(
    response_id=uuid.uuid4(),
    query_id=query_request.query_id,
    response_text="신규 가입자 수 관련 문서를 찾았습니다: ...",
    response_type="document_answer",
    source_documents=[
        {"doc_id": "doc123", "title": "회원가입 통계", "score": 0.95},
        {"doc_id": "doc456", "title": "월간 리포트", "score": 0.89}
    ],
    confidence_score=Decimal("0.90"),
    execution_time=2.1,
    token_usage={"input": 2000, "output": 500},
    timestamp=datetime.utcnow()
)
```

---

## Entity Relationship Diagram

```
┌─────────────────────┐
│  LLM Configuration  │
│  (Anthropic/Claude) │
└──────────┬──────────┘
           │ used by
           ▼
┌─────────────────────┐         ┌──────────────────┐
│   Query Request     │────────▶│ Query Response   │
│  (User input)       │ 1:1     │  (AI output)     │
└──────────┬──────────┘         └──────────────────┘
           │
           │ classified by
           ▼
┌─────────────────────┐
│   Router Chain      │
│ (Intent classifier) │
└─────────────────────┘
           │
           ├─────────▶ Text-to-SQL Chain
           ├─────────▶ Knowledge Chain ─────▶ Embedding Config ─────▶ ChromaDB
           └─────────▶ Assistant Chain

┌─────────────────────┐
│  Budget Tracking    │◀───── Monitors query volume
│  (Cost monitoring)  │
└─────────────────────┘
```

---

## Data Persistence

### SQLite (Conversation Memory)
- **Tables**: `conversations`, `messages`, `sessions`
- **Purpose**: Multi-turn dialogue context
- **Retention**: 30 days (configurable)

### PostgreSQL (Business Database)
- **Access**: Read-only (SELECT permissions only)
- **Schema**: 30-50 tables (user data, business metrics)
- **Purpose**: Text-to-SQL query execution

### ChromaDB (Vector Store)
- **Collection**: `documents`
- **Vectors**: 384 dimensions (paraphrase-multilingual-MiniLM-L12-v2)
- **Capacity**: 10,000 documents
- **Purpose**: Document retrieval for Knowledge Discovery

---

## Migration Impact on Data Model

| Entity | Change Type | Migration Action |
|--------|-------------|------------------|
| LLM Configuration | MODIFIED | Update `provider` from "openai" → "anthropic", add new fields (`model_name`, `timeout`, `max_retries`) |
| Embedding Configuration | UNCHANGED | No migration needed (model maintained) |
| Budget Tracking | NEW | Create new entity, initialize with $100 subscription |
| Query Request | UNCHANGED | No schema changes (LLM provider abstracted) |
| Query Response | MODIFIED | Update `token_usage` structure (Anthropic format) |

**Data Migration Steps**:
1. Add new LLM Configuration record for Claude
2. Initialize Budget Tracking with current month
3. Deprecate OpenAI LLM Configuration (mark as legacy)
4. Update application code to use new LLM Configuration
5. Test with sample queries before production cutover

**Zero Data Loss**: No existing data deleted. OpenAI configuration archived for rollback capability.

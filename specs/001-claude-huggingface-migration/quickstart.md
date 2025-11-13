# Quickstart Guide: Claude Code Migration

**Date**: 2025-01-13
**Version**: 2.0.0
**Audience**: Developers and system administrators
**Estimated Time**: 30 minutes (development setup)

---

## Prerequisites

### Required
- Python 3.10 or higher
- pip package manager
- Git access to repository
- Claude Code Professional subscription ($100/month)
- Anthropic API key (from https://console.anthropic.com/)
- Access to company PostgreSQL database (read-only credentials)

### Optional
- Docker (for containerized deployment)
- GPU with CUDA support (for faster embeddings, not required)

---

## Quick Start (Development Environment)

### Step 1: Clone and Setup (5 minutes)

```bash
# Clone repository
git clone <repository-url>
cd ai-tool

# Checkout migration branch
git checkout 001-claude-huggingface-migration

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**New dependency**: `anthropic` package (replacing `openai`)

---

### Step 2: Configure Environment Variables (5 minutes)

Create `.env` file in project root:

```bash
# Copy template
cp config/.env.example .env

# Edit .env with your values
nano .env  # Or use your preferred editor
```

**Required environment variables**:

```bash
# === LLM Configuration (NEW) ===
ANTHROPIC_API_KEY=sk-ant-YOUR-API-KEY-HERE

# === Legacy LLM (REMOVE AFTER MIGRATION) ===
# OPENAI_API_KEY=sk-YOUR-OLD-KEY  # Comment out or delete

# === Database Configuration (UNCHANGED) ===
POSTGRES_HOST=db.company.internal
POSTGRES_PORT=5432
POSTGRES_DB=company_analytics
POSTGRES_USER=readonly_user
POSTGRES_PASSWORD=YOUR_DB_PASSWORD

# === Vector Store (UNCHANGED) ===
CHROMA_PERSIST_DIRECTORY=./data/chroma
CHROMA_COLLECTION_NAME=documents

# === Application Settings ===
APP_ENV=development
LOG_LEVEL=INFO

# === Budget Monitoring (NEW) ===
BUDGET_LIMIT=100.00
ALERT_EMAIL=admin@company.com
```

**Security checklist**:
- ✅ `.env` file is in `.gitignore` (verify: `git check-ignore .env` returns `.env`)
- ✅ Never commit API keys to version control
- ✅ Use environment-specific `.env` files (dev, staging, prod)

---

### Step 3: Initialize Vector Store (10 minutes)

```bash
# Download embedding model (one-time setup)
python scripts/download_embedding_model.py

# Expected output:
# Downloading paraphrase-multilingual-MiniLM-L12-v2...
# Model downloaded: 471 MB
# Testing Korean language support... OK

# Initialize ChromaDB (creates ./data/chroma directory)
python scripts/init_vector_store.py

# Expected output:
# ChromaDB initialized
# Collection: documents (384 dimensions)
# Status: Ready for indexing
```

**Optional**: Index sample documents for testing

```bash
# Index test documents (Korean language examples)
python scripts/index_documents.py --source ./data/sample_docs/

# Expected output:
# Indexing 10 sample documents...
# Progress: [==========] 10/10 (2.1 seconds)
# Success: 10 documents indexed
# Vector search ready
```

---

### Step 4: Test LLM Connection (5 minutes)

```bash
# Test Claude API connectivity
python scripts/test_claude_connection.py

# Expected output:
# Testing Claude API connection...
# Provider: Anthropic
# Model: claude-3-5-sonnet-20241022
# Status: ✅ Connected
# Response time: 1.2 seconds
# Korean language test: ✅ Passed
#
# Sample query: "안녕하세요"
# Response: "안녕하세요! 무엇을 도와드릴까요?"
```

**Troubleshooting**:
- ❌ **Connection error**: Check `ANTHROPIC_API_KEY` in `.env`
- ❌ **Invalid model**: Verify model name is `claude-3-5-sonnet-20241022`
- ❌ **Timeout**: Check network connectivity to `api.anthropic.com`

---

### Step 5: Run Development Server (5 minutes)

```bash
# Start Streamlit UI
streamlit run src/ui/app.py

# Expected output:
# You can now view your Streamlit app in your browser.
# Local URL: http://localhost:8501
# Network URL: http://192.168.1.X:8501
```

**Access UI**: Open browser to `http://localhost:8501`

**Test queries** (Korean):
1. Text-to-SQL: "지난달 신규 가입자 수는?"
2. Knowledge Discovery: "회원가입 프로세스가 어떻게 되나요?"
3. Assistant: "안녕하세요, 도움이 필요해요"

**Expected behavior**:
- Router correctly classifies query type
- Text-to-SQL generates valid PostgreSQL query
- Knowledge Discovery retrieves relevant documents
- Response time within SLA (Text-to-SQL ≤60s, Knowledge ≤3s)

---

## Verification Checklist

After completing quick start, verify the following:

### ✅ Environment Setup
- [ ] Python 3.10+ installed (`python --version`)
- [ ] Virtual environment activated (prompt shows `(venv)`)
- [ ] All dependencies installed (`pip list | grep anthropic` shows package)
- [ ] `.env` file configured with `ANTHROPIC_API_KEY`
- [ ] `.env` not committed to Git (`git status` shows nothing staged)

### ✅ LLM Integration
- [ ] Claude API connection successful (test script passed)
- [ ] Korean language queries processed correctly
- [ ] Response time acceptable (<60s for Text-to-SQL)
- [ ] No OpenAI API references in active code

### ✅ Vector Store
- [ ] ChromaDB initialized (./data/chroma directory exists)
- [ ] Embedding model downloaded (471 MB in ~/.cache/huggingface/)
- [ ] Sample documents indexed (if running optional step)
- [ ] Vector search returns results (<0.5s response time)

### ✅ Budget Monitoring
- [ ] Budget tracking initialized (`BUDGET_LIMIT` in .env)
- [ ] Query count increments on each request
- [ ] Cost dashboard visible in Streamlit UI
- [ ] Alert thresholds configured (80%, 90%, 100%)

---

## Common Issues and Solutions

### Issue 1: `ModuleNotFoundError: No module named 'anthropic'`

**Solution**:
```bash
pip install anthropic langchain-anthropic
# Verify installation
python -c "import anthropic; print(anthropic.__version__)"
```

---

### Issue 2: `AuthenticationError: Invalid API key`

**Solution**:
1. Verify API key in `.env` starts with `sk-ant-`
2. Check for extra spaces or newlines in `.env` file
3. Confirm API key is active in Anthropic Console (https://console.anthropic.com/)
4. Restart application after updating `.env`

---

### Issue 3: `ConnectionError: Unable to reach Anthropic API`

**Solution**:
1. Check network connectivity: `ping api.anthropic.com`
2. Verify firewall allows HTTPS traffic (port 443)
3. Check proxy settings (if behind corporate firewall)
4. Retry with timeout increase in config

---

### Issue 4: Korean language queries not working

**Solution**:
1. Verify query encoding is UTF-8
2. Check few-shot examples include Korean prompts
3. Test with simple Korean query: "안녕하세요"
4. Review prompt template for Korean language support

---

### Issue 5: Slow response times (>60 seconds)

**Solution**:
1. Check Claude API status: https://status.anthropic.com/
2. Reduce `max_tokens` in LLM configuration (default: 4096)
3. Optimize prompt length (reduce schema context if too large)
4. Monitor network latency to Anthropic API

---

## Next Steps

After successful quick start:

1. **Run test suite**: `pytest tests/ -v` (verify 80%+ coverage)
2. **Review migration guide**: `docs/migration-guide.md` (detailed migration steps)
3. **Configure production**: Update `.env.production` with prod credentials
4. **Set up monitoring**: Configure budget alerts and performance tracking
5. **Deploy to staging**: Test with real data before production rollout

---

## Sample Code: Using Claude in LangChain

### Before (OpenAI)

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# OpenAI configuration
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    openai_api_key=os.environ["OPENAI_API_KEY"]
)

# Create chain
prompt = ChatPromptTemplate.from_template("Generate SQL for: {query}")
chain = prompt | llm
```

### After (Claude)

```python
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate

# Claude configuration
llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0,
    anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
    max_tokens=4096,
    timeout=60,
    max_retries=3
)

# Create chain (same interface!)
prompt = ChatPromptTemplate.from_template("Generate SQL for: {query}")
chain = prompt | llm
```

**Key changes**:
- Import: `ChatOpenAI` → `ChatAnthropic`
- API key: `openai_api_key` → `anthropic_api_key`
- Model: `gpt-4o` → `claude-3-5-sonnet-20241022`
- **Same LangChain interface**: Minimal code changes!

---

## Resources

- **Official Documentation**: https://docs.anthropic.com/
- **LangChain Integration**: https://python.langchain.com/docs/integrations/chat/anthropic
- **Claude Models**: https://docs.anthropic.com/claude/docs/models-overview
- **Anthropic Console**: https://console.anthropic.com/
- **Migration Guide**: `docs/migration-guide.md` (detailed step-by-step)
- **Cost Monitoring**: `docs/cost-monitoring.md` (budget tracking setup)

---

## Support

For issues or questions:

1. **Technical issues**: Check troubleshooting section above
2. **API errors**: Anthropic support (https://support.anthropic.com/)
3. **Project questions**: Contact AI team (Slack: #ai-assistant-dev)

---

**Quickstart Complete!** 🎉

You now have a working development environment with Claude Code integration. Proceed to `docs/migration-guide.md` for production deployment steps.

# Feature Specification: LLM Migration to Claude Code + Hugging Face Embeddings

**Feature Branch**: `001-claude-huggingface-migration`
**Created**: 2025-01-13
**Status**: Draft
**Input**: User description: "openai 사용안할거야. 메인은 claude code 사용할 예정이고 임베딩도 허깅스페이스에서 제공되는 하이 퀄리티를 찾아서 사용할거야."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Developer Integrates Claude Code API (Priority: P1)

As a developer, I need to replace the OpenAI GPT-4o integration with Claude Code API so that the system can operate without OpenAI dependencies while maintaining response quality and Korean language support.

**Why this priority**: This is the core migration requirement. Without LLM functionality, the entire AI assistant cannot operate. This delivers immediate value by enabling the system to function with Claude Code.

**Independent Test**: Can be fully tested by sending a natural language query (e.g., "지난달 신규 가입자 수는?") and verifying that Claude Code generates a valid SQL query response. Success is measured by receiving a properly formatted SQL query within the 30-60 second SLA.

**Acceptance Scenarios**:

1. **Given** the system is configured with Claude Code API credentials, **When** a user asks a Text-to-SQL question in Korean, **Then** the system generates a valid SQL query using Claude Code within 60 seconds
2. **Given** the Claude Code integration is active, **When** a user requests Knowledge Discovery (문서 검색), **Then** the system provides accurate answers with source citations within 3 seconds
3. **Given** the Claude Code API is called, **When** the response is received, **Then** the system logs the token usage and cost for budget monitoring
4. **Given** the system configuration file, **When** a developer reviews the LLM settings, **Then** no OpenAI API keys or references are present

---

### User Story 2 - System Uses High-Quality Embeddings (Priority: P2)

As a system administrator, I need the vector search to use high-quality Hugging Face embedding models so that document retrieval accuracy meets or exceeds the 90% target for Top-5 search results.

**Why this priority**: Embedding quality directly impacts search accuracy, which is critical for Knowledge Discovery. This builds upon P1 by enhancing the quality of one component, but the system can function with the existing embedding model while this is evaluated.

**Independent Test**: Can be tested by running a benchmark test set of 100 queries against the knowledge base and measuring Top-5 accuracy. Success is measured by achieving ≥90% accuracy with the new embedding model.

**Acceptance Scenarios**:

1. **Given** a high-quality Hugging Face embedding model is configured, **When** the system indexes 1000 documents, **Then** all documents are successfully embedded and stored in ChromaDB
2. **Given** the embedding model is active, **When** a user searches for a Korean language query, **Then** the system returns relevant results with ≥90% Top-5 accuracy
3. **Given** the embedding process runs, **When** monitoring the vector search latency, **Then** the search completes within 0.5 seconds (current SLA)
4. **Given** the new embedding model, **When** comparing with the previous model, **Then** the new model provides equal or better multilingual support for Korean text

---

### User Story 3 - Constitutional Budget Approval and Monitoring (Priority: P3)

As a project stakeholder, I need the constitutional budget updated and monitoring implemented so that the monthly operational costs remain sustainable and transparent.

**Why this priority**: Budget approval is necessary for legal/financial compliance, but the technical migration (P1, P2) can proceed in parallel. This ensures long-term sustainability but doesn't block immediate functionality testing.

**Independent Test**: Can be tested by running the system for one week and verifying that daily cost reports are generated and that the cumulative cost projection stays within the approved monthly budget. Success is measured by accurate cost tracking and alert functionality.

**Acceptance Scenarios**:

1. **Given** the constitution.md file, **When** a reviewer checks the version and LLM specification, **Then** the version is 2.0.0 (MAJOR bump) and the LLM is documented as Claude Code
2. **Given** the Claude Code API is in use, **When** the daily cost monitoring script runs, **Then** accurate API usage costs are calculated and logged
3. **Given** the monthly cost exceeds 80% of budget, **When** the monitoring system detects this, **Then** an alert is sent to the project administrator
4. **Given** the cost tracking dashboard, **When** a stakeholder reviews monthly expenses, **Then** they can see a breakdown of LLM API costs vs. budget allocation

---

### Edge Cases

- **What happens when** Claude Code API rate limits are exceeded during peak usage?
  - System should queue requests, show progress indicator to user, and respect rate limits with exponential backoff

- **What happens when** the embedding model file is corrupted or unavailable?
  - System should fail gracefully with clear error message, maintain existing embeddings, and prevent new document indexing until resolved

- **What happens when** monthly budget is exhausted mid-month?
  - System should send critical alert, optionally switch to read-only mode (no new queries), and notify administrators with cost projection

- **What happens when** a Korean query contains mixed English/Korean text?
  - Both the LLM and embedding model should handle multilingual input without degradation in quality

- **What happens when** migrating from existing embeddings to new Hugging Face embeddings for documents?
  - All existing documents must be re-indexed with new embeddings to maintain search consistency

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST remove all OpenAI GPT-4o API dependencies from codebase and configuration
- **FR-002**: System MUST integrate Claude API for all LLM operations using Anthropic API Python library (anthropic package) with direct HTTP API calls
- **FR-003**: System MUST continue using existing paraphrase-multilingual-MiniLM-L12-v2 embedding model (proven Korean support, 384 dimensions, fast inference) for vector search
- **FR-004**: System MUST update constitution.md with MAJOR version bump (1.0.0 → 2.0.0) reflecting the LLM change
- **FR-005**: System MUST implement budget monitoring that tracks daily Claude Code API costs and alerts when exceeding thresholds
- **FR-006**: System MUST maintain Korean language support for all user interactions (Text-to-SQL, Knowledge Discovery, Query Assistant)
- **FR-007**: System MUST preserve existing response time SLA (Text-to-SQL: 30-60 seconds, Knowledge Discovery: ≤3 seconds, Vector search: ≤0.5 seconds)
- **FR-008**: System MUST maintain or improve accuracy targets (SQL query generation: ≥85%, document search Top-5: ≥90%, intent classification: ≥95%)
- **FR-009**: System MUST update all documentation (README, plan.md, tasks.md) to reflect the Claude Code migration and remove OpenAI references
- **FR-010**: System MUST handle API authentication errors gracefully with user-friendly error messages
- **FR-011**: System MUST re-index all existing documents with new embeddings to ensure search consistency (if embedding model changes)

### Key Entities

- **LLM Configuration**: Represents the language model settings including API endpoint, authentication credentials, model name (e.g., Claude Sonnet, Opus), temperature, max tokens, and timeout settings. Relationship: Used by all AI chains (Text-to-SQL, Knowledge Discovery, Query Assistant, Multi-turn).

- **Embedding Configuration**: Represents the vector embedding model settings including model name/path, embedding dimension, tokenizer configuration, device (CPU/GPU), and batch size. Relationship: Used by RAG System for document indexing and query embedding.

- **Budget Tracking**: Represents cost monitoring data including daily API usage, cost per request, cumulative monthly cost, budget limit, alert thresholds (80%, 90%, 100%), and cost projection. Relationship: Monitors LLM Configuration usage and triggers alerts to administrators.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users receive responses to Korean language queries with equivalent or better quality compared to OpenAI GPT-4o baseline (measured by user satisfaction ratings ≥80% and feedback thumbs-up ratio ≥0.85)

- **SC-002**: Monthly operational costs for LLM API usage stay within approved budget of $100 per month (covering Claude Code subscription)

- **SC-003**: Korean language Text-to-SQL queries generate accurate SQL with ≥85% success rate (no user corrections needed)

- **SC-004**: System response times meet or exceed current SLA targets (Text-to-SQL: ≤60 seconds average, Knowledge Discovery: ≤3 seconds average, Vector search: ≤0.5 seconds average)

- **SC-005**: Migration is completed with zero data loss and less than 4 hours of service interruption (planned maintenance window)

- **SC-006**: Document search accuracy using Hugging Face embeddings achieves ≥90% Top-5 accuracy on benchmark test set (100 queries)

- **SC-007**: Constitutional amendment is ratified with version 2.0.0 and synchronized across all project documentation within 1 week of migration completion

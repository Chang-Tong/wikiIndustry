# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WikiProject is a knowledge graph + RAG system for Chinese news/policy documents. It extracts entities/relations via OneKE, stores them in Neo4j, and provides QA via LLM with hallucination guards.

**Stack:**
- Backend: FastAPI (Python 3.11+) in `services/backend/`
- Frontend: React 18 + TypeScript + AntV G6 in `services/frontend/`
- Database: Neo4j 5 (graph), SQLite (raw docs)
- NLP: OneKE service in `services/oneke-official/`
- Embeddings: Ollama (`qwen3-embedding:0.6b`)
- LLM: DeepSeek API (OpenAI-compatible)

## Common Commands

### Run the full system
```bash
# Production-like (Docker)
./start.sh

# Local dev with hot-reload
./dev.sh

# Stop everything
./stop.sh
```

### Backend (inside `services/backend/`)
```bash
# Create venv and install deps
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run with hot-reload (expects Neo4j + OneKE running via Docker)
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Tests
pytest

# Lint / typecheck
ruff check .
mypy .
```

### Frontend (inside `services/frontend/`)
```bash
npm install
npm run dev      # Vite dev server on :5173
npm run build    # tsc + vite build
npm run typecheck
npm run lint
```

### Quick health checks
```bash
curl http://localhost:8010/health          # OneKE
curl http://localhost:8000/healthz         # Backend
curl http://localhost:7474                 # Neo4j Browser
```

## Architecture

### Data Flow
1. **Upload** (`POST /api/v1/json/upload`) → `JSONNewsProcessor` parses news JSON
2. **Extract** → `OneKEClient` extracts entities/relations (no fallback allowed)
3. **Build** → `GraphBuilder` converts extraction into `GraphNode` + `GraphEdge`
4. **Store** → `Neo4jClient.upsert_graph()` writes to Neo4j; `SqliteStore` saves raw text
5. **Correlate** → `CorrelationMiningService` auto-builds `CORRELATED_WITH` edges (hybrid entity + vector similarity)
6. **Query** → `/qa/ask-graph` uses LLM-driven dynamic Cypher; `/qa/rag` uses retrieval + constrained prompting

### Key Modules
- `app/api/v1/routes_json.py` — JSON ingestion pipeline (OneKE → Neo4j)
- `app/api/v1/routes_graph.py` — Graph queries + theme filtering
- `app/api/v1/routes_correlation.py` — Similarity matrix + correlation edges
- `app/integrations/neo4j/client.py` — Async Neo4j driver, `QueryLog` support
- `app/integrations/neo4j/cypher.py` — All Cypher statements
- `app/integrations/oneke/client.py` — OneKE HTTP client
- `app/services/correlation_mining.py` — Hybrid similarity (0.6 entity + 0.4 vector)
- `app/services/rag_engine_v2.py` — LLM-driven dynamic schema discovery
- `app/services/graph_builder.py` — Converts extraction to graph primitives
- `frontend/src/App.tsx` — Single-file React app with G6 graph, upload, QA tabs
- `frontend/src/api/client.ts` — Axios client

### Graph Schema
- Nodes: `NewsItem`, `Organization`, `Person`, `Policy`, `ThemeTag`, `ProvinceTag`, `CityTag`, etc.
- Edges:
  - `REL` — Extracted by OneKE (white solid, directed)
  - `CORRELATED_WITH` — Computed similarity (red dashed `#FF375F`, bidirectional)

## Critical Rules

### 1. No Mock Data / No Simulation / No Fallback
All features must connect to real services. Fallbacks to demo/mock/simulated data are **strictly forbidden**:
- Neo4j must be real
- OneKE must be real (`REQUIRE_REAL_ONEKE=true`)
- LLM calls must be real
- Front-end must display real graph data

**Pipeline Integrity**: Every component in the data flow must perform real work:
- `JSONNewsProcessor` → real JSON parsing (no synthetic data injection)
- `OneKEClient.extract()` → real HTTP call to OneKE service (no LLM simulation, no demo extraction)
- `GraphBuilder` → real graph construction from extraction results
- `Neo4jClient.upsert_graph()` → real Neo4j write (no mock storage)
- `CorrelationMiningService` → real similarity computation (no fake edges)
- `RAGEngine` → real retrieval + real LLM inference (no canned responses)

**Prohibited Patterns**:
- Empty `base_url` passed to `OneKEClient` to trigger LLM fallback
- `_extract_llm()` or `_extract_demo()` methods in production paths
- Any endpoint that bypasses `REQUIRE_REAL_ONEKE` guard
- Mock Neo4j drivers or in-memory graph replacements
- Simulated embeddings (random vectors, zero vectors)
- Hardcoded QA responses

**Enforcement**: All ingestion paths must validate `REQUIRE_REAL_ONEKE=true` and `ONEKE_BASE_URL` points to a real service before calling `OneKEClient.extract()`. Any violation must raise `RuntimeError` immediately.

### 2. Mandatory Post-Change Review
**After every code change**, launch three subagents in parallel to review:
- Code reuse (duplicate code, existing utilities)
- Code quality (redundant state, parameter bloat, deep nesting)
- Efficiency (N+1 queries, repeated computation, leaks)

All reported issues must be fixed or explained with inline comments.

### 3. News Chunking Strategy
News documents are **never split**. Each JSON news item is one complete retrieval chunk:
- News is short and semantically whole
- Structured metadata (title, source, date, tags) must stay intact
- Implementation: `services/backend/app/services/rag_engine.py`

### 4. Ollama Embedding Is Required
`REQUIRE_OLLAMA_EMBEDDING=true` forces Ollama usage. If Ollama is unavailable, the system must error explicitly—no silent fallback to OpenAI or TF-IDF.

## Environment

Copy `.env.example` to `.env` and fill in:
```bash
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_API_KEY=sk-...
OPENAI_MODEL=deepseek-chat

ONEKE_BASE_URL=http://oneke:8000
REQUIRE_REAL_ONEKE=true

OLLAMA_BASE_URL=http://host.docker.internal:11434/v1
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:0.6b
REQUIRE_OLLAMA_EMBEDDING=true
```

## Useful Tests

```bash
# Upload test data
curl -X POST http://localhost:8000/api/v1/json/upload \
  -F "file=@test_data.json" \
  -F "schema_name=MOE_News" \
  -F "mode=incremental"

# Graph QA
curl -X POST http://localhost:8000/api/v1/qa/ask-graph \
  -H "Content-Type: application/json" \
  -d '{"question": "教育部发布了哪些政策？", "top_k": 10}'

# Get correlations
curl "http://localhost:8000/api/v1/correlations?min_score=0.3&limit=5"
```

## Git Workflow

### Push via gh CLI

This repo uses **gh CLI + HTTPS** for pushing (not SSH). When pushing, always use this flow:

```bash
# Ensure remote is HTTPS
git remote set-url origin https://github.com/Chang-Tong/wikiIndustry.git

# Ensure git uses gh credentials
git config --global credential.helper 'store'
gh auth setup-git

# Push
git push origin main
```

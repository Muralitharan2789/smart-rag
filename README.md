# Smart RAG — Table-Aware Hybrid Search RAG System

A retrieval-augmented generation system built from first principles — no LangChain,
no framework abstractions — to deeply understand how production RAG pipelines
actually work: parsing, table-aware chunking, hybrid search, reranking, and grounded
answer generation, with a real React chat UI on top.

Built as a hands-on learning project during a Data Engineer → Gen AI Engineer
transition, deliberately choosing to build and debug each component manually rather
than relying on a framework to hide the hard parts. Extended beyond the initial
build with evaluation tooling, latency/cost instrumentation, logging and
monitoring, and multi-hop agentic retrieval.

## Why "table-aware"?

Most naive RAG chunkers split documents into fixed-size text blocks without
understanding structure — which frequently cuts markdown/PDF tables mid-row,
corrupting the data before it's ever embedded. This project solves that with a
two-stage table detection pipeline:

1. **Ruled-table detection** via `pymupdf`'s `find_tables()` — catches tables with
   visible borders/lines in the PDF's structure.
2. **Position-based heuristic fallback** — for tables with no visible borders at
   all (common in reports and financial statements), groups words by visual
   line/column alignment to detect tables `find_tables()` misses entirely.

Both are fed into a block-based chunker that treats each detected table as an atomic
unit — a table is never split across two chunks, even if that means one chunk
exceeds the normal size target. Plain-text blocks are independently size-limited
too — a real bug found during testing: a long, uninterrupted stretch of text (e.g.
one speaker's long monologue in an earnings-call transcript) had no internal
structure to split on, and one such block was found ballooning to 12,788 characters
before this was caught and fixed.

## Architecture

```
Document (PDF/DOCX)
   |
   v
parser.py       - ruled + heuristic table detection -> markdown-formatted text
   |
   v
chunker.py      - block-based, table-aware chunking (tables never split mid-row;
                   oversized text blocks independently size-limited)
   |
   v
embedder.py     - sentence-transformers (all-MiniLM-L6-v2, CPU-only, 384-dim)
   |
   v
PostgreSQL + pgvector  - chunks + embeddings + full-text search index + query logs
   |
   v (at query time)
search.py       - vector search (cosine) + BM25 keyword search -> RRF fusion
   |
   v
search.py       - cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
   |
   v
agent.py        - (optional) multi-hop: LLM decides if more retrieval is needed
   |
   v
llm.py          - Gemini, grounded answer generation with source citations,
                   guardrailed against investment-advice-style questions
   |
   v
React frontend  - chat UI, file upload, document-picker, expandable source
                   citations with rendered tables
```

## Stack

- **Backend**: FastAPI (native Python, no ASGI containerization for the app itself)
- **Database**: PostgreSQL 16 + pgvector, running in Docker (Postgres only —
  Python runs natively on the host)
- **Parsing**: `pymupdf` (PDF) + `python-docx` (DOCX), with a custom heuristic
  table detector for borderless tables
- **Embeddings**: `sentence-transformers`, `all-MiniLM-L6-v2` (384-dim, CPU-only)
- **Search**: pgvector cosine similarity + Postgres full-text search (BM25-style),
  fused via Reciprocal Rank Fusion (RRF)
- **Reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM**: Google Gemini (`gemini-3.5-flash`, with fallback to `gemini-2.5-flash`
  on free-tier quota exhaustion) — originally scoped for GPT-4o, swapped due to
  OpenAI's API requiring paid billing with no free tier; the LLM call is isolated
  in `llm.py` specifically so the provider can be swapped without touching search,
  chunking, or the API contract
- **Frontend**: React (Vite), `react-markdown` + `remark-gfm` for rendering
  retrieved markdown tables as real HTML tables, `axios` for API calls
- **No LangChain** — every stage (parsing, chunking, embedding, search, fusion,
  reranking, generation) is implemented directly, to understand the mechanics
  rather than treat them as a black box

## Features

- Multi-format ingestion (PDF, DOCX) via `/upload`, with automatic parsing,
  table-aware chunking, embedding, and storage
- Hybrid search combining semantic (vector) and exact-keyword (BM25) retrieval
- Reciprocal Rank Fusion to merge both ranked result sets without needing to
  normalize incomparable scoring scales
- Cross-encoder reranking for precision on the final shortlist
- Grounded answer generation — the LLM is constrained to only use retrieved
  source chunks, and explicitly instructed to decline rather than hallucinate
  when the answer isn't in the sources
- A system-prompt guardrail against investment-advice-style questions: the
  system will factually summarize what a source says about financial
  performance or outlook, but explicitly declines to recommend buying, holding,
  or selling a security
- Multi-hop / agentic retrieval (`/query-agentic`) — kept as a separate endpoint
  from the standard single-pass `/query`. The LLM judges whether retrieved
  chunks are sufficient to answer; if not, it reformulates the search query and
  retrieves again (capped at a configurable number of hops). Verified on a real
  question where single-pass search failed outright ("no information found")
  but the reformulated second-hop query correctly located the answer
- Document-scoped search — a dropdown in the UI (and a `document_name` filter
  in the API) restricts search to one uploaded document, or all documents
  together
- Source citations with expandable "Exhibit" panels, rendering markdown tables
  as real HTML tables in the UI
- Basic in-memory query caching (dev-only; resets on server restart, and does
  not currently invalidate on new uploads — a known gap, not yet fixed)
- Per-stage latency instrumentation (`embedding_ms`, `hybrid_search_ms`,
  `rerank_ms`, `llm_generation_ms`) and LLM token usage tracking returned with
  every `/query` response. Empirically, steady-state local pipeline latency
  (embedding + search + rerank combined) runs under 100ms; the LLM API call
  itself is the dominant and most variable cost, occasionally ranging from
  ~8s to ~50s depending on provider-side load
- Logging and monitoring: every query is logged to both a rotating file
  (`app.log`) and a Postgres `query_logs` table, with a `/metrics` endpoint
  summarizing total queries, cache hit rate, error rate, and average latency
- A small evaluation harness (`run_eval.py`) combining a fast keyword sanity
  check with LLM-as-judge scoring for faithfulness (is the answer actually
  grounded in the retrieved sources?) and relevance (does it address the
  question?) — not exhaustive, but a real, working alternative to manually
  eyeballing outputs
- Synthetic scale benchmarking tooling (`generate_synthetic_data.py`,
  `benchmark_search.py`) used to empirically test vector search behavior at
  5,000 rows and validate when an `ivfflat` index starts to pay off

## Known Limitations

Documented honestly rather than glossed over:

- **Table detection isn't perfect.** Deliberately tab-simulated "tables" with no
  real structure (no borders, no consistent alignment) can slip past both
  detection passes. Verified against an adversarial 29-table test document; a
  small number of edge cases still aren't caught.
- **Table captions sometimes land in a separate chunk from the table itself**,
  since caption-to-table association isn't currently implemented — a known,
  minor retrieval-context gap.
- **The query cache is in-memory only** and does not invalidate when documents
  are uploaded or deleted — a stale cached answer can persist until the server
  restarts. A production deployment would use Redis (or similar) with proper
  invalidation, or a short TTL.
- **No approximate vector index at current data scale** (a few hundred chunks) —
  deliberately skipped, since `ivfflat`/`hnsw` indexes add overhead without
  benefit below roughly tens of thousands of rows. Benchmarked at 5,000
  synthetic rows: added a modest ~15% average speedup with comparable worst-case
  latency, confirming these indexes matter more at significantly larger scale.
- **Multi-hop retrieval is capped and simple** — a fixed max-hop limit, no
  memory of hop history beyond deduplicated chunks, and roughly 4x the token
  cost of a single-pass query. Not run by default; a deliberate, separate
  `/query-agentic` endpoint rather than the default query path.
- **No conversation memory** — each question is answered independently; the
  system does not know what was asked previously in the same session.
- **Free-tier LLM API quotas are restrictive** for heavy iterative testing
  (as low as 20 requests/day for some models on the free tier) — a real,
  personally-encountered constraint worth knowing about before assuming this
  scales to production traffic without a paid tier.

## Local Setup

**Backend:**
```powershell
docker compose up -d                          # starts Postgres + pgvector only
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd backend
uvicorn main:app --reload --port 8000
```

**Frontend** (separate terminal):
```powershell
cd frontend
npm install
npm run dev
```
Open http://localhost:5173

**Environment variables** (`.env` in project root, never committed):
```
DATABASE_URL=postgresql://raguser:ragpassword@localhost:5432/smartrag
GEMINI_API_KEY=your-key-here
```

**Running the evaluation harness:**
```powershell
cd backend
python run_eval.py
```

**Checking system metrics:**
```
GET http://127.0.0.1:8000/metrics
```

## Project Structure

```
smart-rag/
|-- backend/
|   |-- main.py                    - FastAPI app: /health, /upload, /query,
|   |                                 /query-agentic, /documents, /metrics
|   |-- database.py                - Postgres connection, chunk insert/count,
|   |                                 query logging
|   |-- parser.py                  - PDF/DOCX parsing, ruled + heuristic
|   |                                 table detection
|   |-- chunker.py                 - table-aware, block-based chunking, with
|   |                                 independent size-limiting for oversized
|   |                                 plain-text blocks
|   |-- embedder.py                - sentence-transformers embedding
|   |-- search.py                  - vector search, keyword search, RRF,
|   |                                 reranking, document-scoped filtering
|   |-- llm.py                     - Gemini grounded answer generation,
|   |                                 investment-advice guardrail, multi-hop
|   |                                 decision function
|   |-- agent.py                   - multi-hop retrieval orchestrator
|   |-- ingest.py                  - CLI pipeline (parse -> chunk -> embed -> store)
|   |-- eval_questions.py          - evaluation test set
|   |-- eval_judge.py              - LLM-as-judge faithfulness/relevance scoring
|   |-- run_eval.py                - evaluation harness runner
|   |-- generate_synthetic_data.py - synthetic data generator for scale testing
|   |-- benchmark_search.py        - vector search speed benchmarking
|   |-- app.log                    - application log file (gitignored)
|   `-- requirements.txt
|-- frontend/
|   |-- src/App.jsx                - chat UI, document picker, file upload,
|   |                                 source citations panel
|   `-- src/App.css                - design system (color, type, layout)
|-- uploads/                       - uploaded documents (gitignored)
`-- docker-compose.yml             - Postgres + pgvector container definition
```

## Status

Core system complete and demo-ready - table-aware hybrid search RAG pipeline
with a working React demo (select a document -> ask -> cited, grounded answer),
extended with evaluation, monitoring, and multi-hop retrieval capabilities.

A real chunking bug (oversized, unsplit text blocks on documents with long
uninterrupted passages) was found and fixed during evaluation against a real
financial earnings-call transcript - an example of the value of testing beyond
the original adversarial sample-tables document.
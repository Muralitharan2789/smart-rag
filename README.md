# Smart RAG — Table-Aware Hybrid Search RAG System

A retrieval-augmented generation system built from first principles — no LangChain,
no framework abstractions — to deeply understand how production RAG pipelines
actually work: parsing, table-aware chunking, hybrid search, reranking, and grounded
answer generation, with a real React chat UI on top.

Built as a hands-on learning project during a Data Engineer → Gen AI Engineer
transition, deliberately choosing to build and debug each component manually rather
than relying on a framework to hide the hard parts.

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
exceeds the normal size target.

## Architecture

```
Document (PDF/DOCX)
   │
   ▼
parser.py       — ruled + heuristic table detection → markdown-formatted text
   │
   ▼
chunker.py      — block-based, table-aware chunking (tables never split mid-row)
   │
   ▼
embedder.py     — sentence-transformers (all-MiniLM-L6-v2, CPU-only, 384-dim)
   │
   ▼
PostgreSQL + pgvector  — chunks + embeddings + full-text search index
   │
   ▼ (at query time)
search.py       — vector search (cosine) + BM25 keyword search → RRF fusion
   │
   ▼
search.py       — cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
   │
   ▼
llm.py          — Gemini, grounded answer generation with source citations
   │
   ▼
React frontend  — chat UI, file upload, expandable source citations (rendered tables)
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
- Source citations with expandable previews, rendering markdown tables as real
  HTML tables in the UI
- Document-scoped search — queries can be restricted to a single uploaded
  document via an optional `document_name` filter
- Basic in-memory query caching (dev-only; resets on server restart — a
  documented simplification, not a claim of production-grade caching)
- Per-stage latency instrumentation (`embedding_ms`, `hybrid_search_ms`,
  `rerank_ms`, `llm_generation_ms`) and LLM token usage tracking returned with
  every `/query` response
- A small evaluation harness (`run_eval.py`) combining a fast keyword sanity
  check with LLM-as-judge scoring for faithfulness (is the answer actually
  grounded in the retrieved sources?) and relevance (does it address the
  question?) — not exhaustive, but a real, working alternative to manually
  eyeballing outputs

## Known Limitations

Documented honestly rather than glossed over:

- **Table detection isn't perfect.** Deliberately tab-simulated "tables" with no
  real structure (no borders, no consistent alignment) can slip past both
  detection passes. Verified against an adversarial 29-table test document; a
  small number of edge cases still aren't caught.
- **Table captions sometimes land in a separate chunk from the table itself**,
  since caption-to-table association isn't currently implemented — a known,
  minor retrieval-context gap.
- **The query cache is in-memory only** — not persistent, not shared across
  multiple server instances. A production deployment would use Redis or similar.
- **No approximate vector index at current data scale** (a few hundred chunks) —
  deliberately skipped, since `ivfflat`/`hnsw` indexes add overhead without
  benefit below roughly tens of thousands of rows. Benchmarked at 5,000
  synthetic rows: added a modest ~15% average speedup with comparable worst-case
  latency, confirming these indexes matter more at significantly larger scale.
- **No multi-hop or agentic retrieval** — each query performs exactly one
  retrieval pass. Cannot currently follow a reference from one retrieved chunk
  to look up related information elsewhere in the document store.
- **Free-tier LLM API quotas are restrictive** for heavy iterative testing
  (as low as 20-1000 requests/day depending on model and tier) — a real
  constraint worth knowing about before assuming this scales to production
  traffic without a paid tier.

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

## Project Structure

```
smart-rag/
├── backend/
│   ├── main.py              — FastAPI app, /health, /upload, /query endpoints
│   ├── database.py          — Postgres connection, chunk insert/count
│   ├── parser.py            — PDF/DOCX parsing, ruled + heuristic table detection
│   ├── chunker.py           — table-aware, block-based chunking
│   ├── embedder.py          — sentence-transformers embedding
│   ├── search.py            — vector search, keyword search, RRF, reranking
│   ├── llm.py                — Gemini grounded answer generation
│   ├── ingest.py             — CLI pipeline (parse → chunk → embed → store)
│   ├── eval_questions.py     — evaluation test set
│   ├── eval_judge.py         — LLM-as-judge faithfulness/relevance scoring
│   ├── run_eval.py           — evaluation harness runner
│   └── requirements.txt
├── frontend/
│   └── src/App.jsx           — chat UI, file upload, source citations panel
├── uploads/                  — uploaded documents (gitignored)
└── docker-compose.yml        — Postgres + pgvector container definition
```

## Status

✅ Core system complete — table-aware hybrid search RAG pipeline with a working
React demo (upload → ask → cited, grounded answer).

🚧 Ongoing — extending beyond the initial build with evaluation tooling, latency/cost
instrumentation, document-scoped search, and scale benchmarking.

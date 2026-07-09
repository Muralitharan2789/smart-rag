# Smart RAG — Table-Aware Hybrid Search RAG System

A production-grade RAG system with table-aware chunking, hybrid search
(pgvector cosine + BM25), Reciprocal Rank Fusion, cross-encoder reranking,
and GPT-4o grounded answer generation.

## Stack
- FastAPI (native Python) + PostgreSQL 16 with pgvector (Docker)
- No LangChain — built from first principles
- sentence-transformers cross-encoder reranker
- React frontend with source citations

## Local Setup
1. `docker compose up -d`   (starts Postgres+pgvector only)
2. `python -m venv venv && .\venv\Scripts\Activate.ps1`
3. `pip install -r requirements.txt`
4. `cd backend && uvicorn main:app --reload`

## Status
🚧 Week 1 of 4 — environment + FastAPI skeleton + health check
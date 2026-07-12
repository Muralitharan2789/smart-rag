import shutil
import time
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database import check_db_connection, insert_chunk, count_chunks
from embedder import embed_text, embed_batch
from search import hybrid_search, rerank
from llm import generate_answer
from parser import parse_document
from chunker import chunk_document

app = FastAPI(title="Smart RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default dev server address
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"


class QueryRequest(BaseModel):
    question: str
    top_k: int = 10
    rerank_top_n: int = 5
    document_name: str | None = None  # None = search all documents


_query_cache: dict[str, dict] = {}


def _cache_key(request: QueryRequest) -> str:
    return f"{request.question.strip().lower()}|{request.document_name}|{request.top_k}|{request.rerank_top_n}"


@app.get("/health")
def health():
    db_ok = check_db_connection()
    return {
        "status": "ok" if db_ok else "degraded",
        "database": "connected" if db_ok else "unreachable",
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    UPLOAD_DIR.mkdir(exist_ok=True)
    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    text = parse_document(str(file_path))
    chunks = chunk_document(text, max_chunk_size=800, overlap=100)
    texts = [c.text for c in chunks]
    embeddings = embed_batch(texts)

    for chunk, embedding in zip(chunks, embeddings):
        insert_chunk(file.filename, chunk.text, chunk.chunk_type, embedding)

    return {
        "filename": file.filename,
        "chunks_stored": count_chunks(file.filename),
        "table_chunks": sum(1 for c in chunks if c.chunk_type == "table"),
        "text_chunks": sum(1 for c in chunks if c.chunk_type == "text"),
    }


@app.post("/query")
def query(request: QueryRequest):
    cache_key = _cache_key(request)
    if cache_key in _query_cache:
        cached = dict(_query_cache[cache_key])
        cached["cached"] = True
        cached["timings"] = {"total_ms": 0.1, "note": "served from cache, no pipeline re-run"}
        return cached

    timings = {}
    t0 = time.perf_counter()

    query_embedding = embed_text(request.question)
    timings["embedding_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    t1 = time.perf_counter()
    fused_results = hybrid_search(
        request.question, query_embedding, top_k=request.top_k, document_name=request.document_name
    )
    timings["hybrid_search_ms"] = round((time.perf_counter() - t1) * 1000, 1)

    t2 = time.perf_counter()
    reranked_results = rerank(request.question, fused_results, top_n=request.rerank_top_n)
    timings["rerank_ms"] = round((time.perf_counter() - t2) * 1000, 1)

    if not reranked_results:
        timings["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "answer": "No relevant information found in the indexed documents.",
            "sources": [],
            "timings": timings,
        }

    t3 = time.perf_counter()
    result = generate_answer(request.question, reranked_results)
    timings["llm_generation_ms"] = round((time.perf_counter() - t3) * 1000, 1)
    timings["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    result["timings"] = timings
    result["cached"] = False
    _query_cache[cache_key] = dict(result)
    return result
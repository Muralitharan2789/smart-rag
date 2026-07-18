import shutil
import time
from pathlib import Path
import logging

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database import check_db_connection, insert_chunk, count_chunks, log_query, get_connection
from embedder import embed_text, embed_batch
from search import hybrid_search, rerank
from llm import generate_answer
from parser import parse_document
from chunker import chunk_document
from agent import agentic_query

app = FastAPI(title="Smart RAG API")


logging.basicConfig(
    level=logging.WARNING,  # quiet by default for third-party libraries
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("smart_rag")
logger.setLevel(logging.INFO)  # but our own logger stays verbose

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
        logger.info(f"CACHE HIT | question={request.question[:60]!r}")
        log_query(request.question, request.document_name, cached=True, error=False,
                   num_sources=len(cached.get("sources", [])))
        return cached

    timings = {}
    t0 = time.perf_counter()

    try:
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
            logger.info(f"NO RESULTS | question={request.question[:60]!r}")
            log_query(request.question, request.document_name, cached=False, error=False,
                       timings=timings, num_sources=0)
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

        logger.info(
            f"SUCCESS | question={request.question[:60]!r} | total_ms={timings['total_ms']} | "
            f"sources={len(result.get('sources', []))}"
        )
        log_query(request.question, request.document_name, cached=False, error=False,
                   timings=timings, token_usage=result.get("token_usage"), num_sources=len(result.get("sources", [])))
        return result

    except Exception as e:
        logger.error(f"ERROR | question={request.question[:60]!r} | {e}")
        log_query(request.question, request.document_name, cached=False, error=True)
        raise

@app.post("/query-agentic")
def query_agentic(request: QueryRequest):
    logger.info(f"AGENTIC QUERY | question={request.question[:60]!r}")
    result = agentic_query(request.question, document_name=request.document_name, max_hops=2)
    log_query(request.question, request.document_name, cached=False, error=False,
               num_sources=len(result.get("sources", [])))
    return result

@app.get("/metrics")
def metrics():
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM query_logs;")
        total = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM query_logs WHERE cached = TRUE;")
        cached_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM query_logs WHERE error = TRUE;")
        error_count = cur.fetchone()[0]

        cur.execute("""
            SELECT AVG((timings->>'total_ms')::float)
            FROM query_logs
            WHERE cached = FALSE AND error = FALSE AND timings IS NOT NULL;
        """)
        avg_latency = cur.fetchone()[0]

        cur.execute("""
            SELECT question, cached, error, timings->>'total_ms' AS total_ms, created_at
            FROM query_logs
            ORDER BY created_at DESC
            LIMIT 10;
        """)
        recent = [
            {"question": r[0], "cached": r[1], "error": r[2], "total_ms": r[3], "created_at": str(r[4])}
            for r in cur.fetchall()
        ]

        return {
            "total_queries": total,
            "cache_hit_rate": round(cached_count / total, 3) if total else None,
            "error_rate": round(error_count / total, 3) if total else None,
            "avg_latency_ms": round(avg_latency, 1) if avg_latency else None,
            "recent_queries": recent,
        }
    finally:
        conn.close()
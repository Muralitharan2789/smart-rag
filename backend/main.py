from fastapi import FastAPI
from pydantic import BaseModel

from database import check_db_connection
from embedder import embed_text
from search import hybrid_search, rerank
from llm import generate_answer

app = FastAPI(title="Smart RAG API")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 10
    rerank_top_n: int = 5


@app.get("/health")
def health():
    db_ok = check_db_connection()
    return {
        "status": "ok" if db_ok else "degraded",
        "database": "connected" if db_ok else "unreachable",
    }


@app.post("/query")
def query(request: QueryRequest):
    query_embedding = embed_text(request.question)

    fused_results = hybrid_search(request.question, query_embedding, top_k=request.top_k)
    reranked_results = rerank(request.question, fused_results, top_n=request.rerank_top_n)

    if not reranked_results:
        return {
            "answer": "No relevant information found in the indexed documents.",
            "sources": [],
        }

    result = generate_answer(request.question, reranked_results)
    return result
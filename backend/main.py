from fastapi import FastAPI
from database import check_db_connection

app = FastAPI(title="Smart RAG API")

@app.get("/health")
def health():
    db_ok = check_db_connection()
    return {
        "status": "ok" if db_ok else "degraded",
        "database": "connected" if db_ok else "unreachable",
    }
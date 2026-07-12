from database import get_connection


def vector_search(query_embedding: list[float], top_k: int = 10) -> list[dict]:
    """
    Returns the top_k chunks most semantically similar to the query embedding,
    ranked by cosine similarity (1 - cosine distance). pgvector's <=> operator
    computes cosine distance directly, so lower is more similar.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, document_name, chunk_text, chunk_type,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """,
            (query_embedding, query_embedding, top_k),
        )
        rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "document_name": r[1],
                "chunk_text": r[2],
                "chunk_type": r[3],
                "similarity": r[4],
            }
            for r in rows
        ]
    finally:
        conn.close()

def keyword_search(query_text: str, top_k: int = 10) -> list[dict]:
    """
    Returns the top_k chunks best matching the query via BM25-style full-text
    ranking (ts_rank), using Postgres's built-in text search.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, document_name, chunk_text, chunk_type,
                   ts_rank(content_tsv, plainto_tsquery('english', %s)) AS rank
            FROM chunks
            WHERE content_tsv @@ plainto_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s;
            """,
            (query_text, query_text, top_k),
        )
        rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "document_name": r[1],
                "chunk_text": r[2],
                "chunk_type": r[3],
                "rank": r[4],
            }
            for r in rows
        ]
    finally:
        conn.close()

def reciprocal_rank_fusion(
    vector_results: list[dict],
    keyword_results: list[dict],
    k: int = 60,
    top_n: int = 10,
) -> list[dict]:
    """
    Merges two ranked result lists using RRF. Each chunk's fused score is the sum
    of 1/(k + rank) across every list it appears in (rank is 1-indexed position).
    k=60 is the standard constant from the original RRF paper — it dampens the
    impact of rank differences further down each list.
    """
    scores: dict[int, float] = {}
    chunk_data: dict[int, dict] = {}

    for rank, result in enumerate(vector_results, start=1):
        chunk_id = result["id"]
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)
        chunk_data[chunk_id] = result

    for rank, result in enumerate(keyword_results, start=1):
        chunk_id = result["id"]
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)
        chunk_data.setdefault(chunk_id, result)

    ranked_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

    fused = []
    for chunk_id in ranked_ids[:top_n]:
        entry = dict(chunk_data[chunk_id])
        entry["rrf_score"] = scores[chunk_id]
        fused.append(entry)

    return fused


def hybrid_search(query_text: str, query_embedding: list[float], top_k: int = 10) -> list[dict]:
    """Convenience wrapper: runs both searches and fuses them in one call."""
    vector_results = vector_search(query_embedding, top_k=top_k)
    keyword_results = keyword_search(query_text, top_k=top_k)
    return reciprocal_rank_fusion(vector_results, keyword_results, top_n=top_k)


from sentence_transformers import CrossEncoder

_reranker = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    return _reranker


def rerank(query: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
    """
    Reranks candidate chunks by true query-chunk relevance using a cross-encoder.
    Unlike vector similarity (which compares independently-computed embeddings),
    a cross-encoder processes the query and chunk text together, producing a more
    accurate but more computationally expensive relevance score — which is exactly
    why it's only run on the small top_n shortlist from RRF, not the whole dataset.
    """
    if not candidates:
        return []

    model = get_reranker()
    pairs = [(query, c["chunk_text"]) for c in candidates]
    scores = model.predict(pairs)

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda pair: pair[1], reverse=True)

    reranked = []
    for candidate, score in scored[:top_n]:
        entry = dict(candidate)
        entry["rerank_score"] = float(score)
        reranked.append(entry)

    return reranked
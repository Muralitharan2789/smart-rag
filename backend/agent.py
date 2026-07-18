from embedder import embed_text
from search import hybrid_search, rerank
from llm import decide_next_step, generate_answer


def agentic_query(question: str, document_name: str | None = None, max_hops: int = 2) -> dict:
    """
    Multi-hop retrieval: performs an initial search, then lets the LLM decide
    whether a follow-up search is needed before answering. Deduplicates chunks
    across hops by id, and caps at max_hops to bound cost.
    """
    gathered = {}  # keyed by chunk id, to dedupe across hops
    hop_log = []

    current_query = question
    hop = 0

    while hop < max_hops:
        query_embedding = embed_text(current_query)
        fused = hybrid_search(current_query, query_embedding, top_k=10, document_name=document_name)
        reranked = rerank(current_query, fused, top_n=5)

        for chunk in reranked:
            gathered[chunk["id"]] = chunk

        hop_log.append({"hop": hop + 1, "query": current_query, "chunks_found": len(reranked)})

        decision = decide_next_step(question, list(gathered.values()), hop, max_hops)
        hop += 1

        if decision["decision"] == "ANSWER":
            hop_log[-1]["decision"] = "ANSWER"
            hop_log[-1]["reasoning"] = decision.get("reasoning", "")
            break
        else:
            hop_log[-1]["decision"] = "SEARCH"
            hop_log[-1]["reasoning"] = decision.get("reasoning", "")
            current_query = decision["search_query"]

    final_chunks = list(gathered.values())
    if not final_chunks:
        return {
            "answer": "No relevant information found in the indexed documents.",
            "sources": [],
            "hops": hop_log,
        }

    result = generate_answer(question, final_chunks)
    result["hops"] = hop_log
    return result
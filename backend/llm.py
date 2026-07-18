import os
from google import genai
from dotenv import load_dotenv
import re
import json


load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def generate_answer(query: str, context_chunks: list[dict]) -> dict:
    """
    Generates a grounded answer using Gemini, given a query and the reranked
    context chunks. Returns both the answer text and the source list used,
    so the API response can show citations.
    """
    context_blocks = []
    for i, chunk in enumerate(context_chunks, start=1):
        context_blocks.append(
            f"[Source {i}: {chunk['document_name']}, chunk_type={chunk['chunk_type']}]\n"
            f"{chunk['chunk_text']}"
        )
    context_text = "\n\n".join(context_blocks)

    system_instruction = (
        "You are a precise assistant that answers questions using ONLY the provided "
        "source excerpts. If the answer isn't in the sources, say so clearly instead "
        "of guessing. When you use information from a source, reference it by its "
        "[Source N] number.\n\n"
        "You may factually summarize what a source says about financial performance, "
        "trends, or outlook. You must NOT provide investment advice or recommendations "
        "(e.g. whether to buy, hold, or sell a security). If asked for such a "
        "recommendation, summarize the relevant factual content from the sources, then "
        "clearly state that you can't provide investment advice and this isn't a "
        "substitute for professional financial guidance."
    )

    prompt = f"Sources:\n\n{context_text}\n\nQuestion: {query}"

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
        config={
            "system_instruction": system_instruction,
            "temperature": 0.2,
        },
    )

    answer_text = response.text

    usage = getattr(response, "usage_metadata", None)
    token_usage = {
        "prompt_tokens": getattr(usage, "prompt_token_count", None),
        "completion_tokens": getattr(usage, "candidates_token_count", None),
        "total_tokens": getattr(usage, "total_token_count", None),
    } if usage else None

    sources = [
        {
            "source_number": i,
            "document_name": chunk["document_name"],
            "chunk_type": chunk["chunk_type"],
            "chunk_text_preview": chunk["chunk_text"][:200],
        }
        for i, chunk in enumerate(context_chunks, start=1)
    ]

    return {"answer": answer_text, "sources": sources, "token_usage": token_usage}

def decide_next_step(question: str, gathered_chunks: list[dict], hop_number: int, max_hops: int) -> dict:
    """
    Asks the LLM whether the currently gathered chunks are sufficient to answer
    the question, or whether a follow-up, more specific search is needed.
    Returns {"decision": "ANSWER" | "SEARCH", "search_query": str | None, "reasoning": str}.
    """
    if hop_number >= max_hops:
        return {"decision": "ANSWER", "search_query": None, "reasoning": "Max hops reached, answering with what's available."}

    context_text = "\n\n".join(
        f"[Chunk {i+1} from {c['document_name']}]: {c['chunk_text'][:600]}"
        for i, c in enumerate(gathered_chunks)
    ) if gathered_chunks else "(nothing retrieved yet)"

    prompt = f"""You are deciding whether enough information has been retrieved to answer a question.

Question: {question}

Information gathered so far:
{context_text}

Can this question be FULLY and confidently answered using only the information above?

If YES, respond with: {{"decision": "ANSWER", "search_query": null, "reasoning": "<why this is enough>"}}

If NO, because a specific piece of related information is still missing (not because
the topic is entirely absent from the sources), respond with:
{{"decision": "SEARCH", "search_query": "<a short, specific search query for the missing piece>", "reasoning": "<what's missing and why>"}}

Respond with ONLY valid JSON, no other text."""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
        config={"temperature": 0},
    )

    raw = re.sub(r"^```json\s*|\s*```$", "", response.text.strip()).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"decision": "ANSWER", "search_query": None, "reasoning": "Could not parse decision, defaulting to answer."}
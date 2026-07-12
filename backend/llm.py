import os
from google import genai
from dotenv import load_dotenv

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
        "[Source N] number."
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
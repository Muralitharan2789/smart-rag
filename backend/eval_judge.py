import os
import json
import re
from google import genai
from dotenv import load_dotenv
import time

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

JUDGE_PROMPT = """You are an evaluator scoring a RAG (Retrieval-Augmented Generation) system's answer.

Question: {question}

Sources provided to the system:
{sources_text}

System's Answer: {answer}

Score this answer on two dimensions, each from 1 to 5:

FAITHFULNESS (1-5): Does the answer ONLY use information that is actually present in
the sources above? A score of 5 means every claim traces back to a source. A score of
1 means the answer contains information not found in any source (hallucination). If
the answer correctly says "I don't have this information" when the sources genuinely
don't cover the question, that is FAITHFUL and should score 5.

RELEVANCE (1-5): Does the answer actually address what was asked? A score of 5 means
it directly and completely answers the question. A score of 1 means it's off-topic or
doesn't engage with the question at all.

Respond with ONLY valid JSON in this exact format, no other text:
{{"faithfulness": <1-5>, "relevance": <1-5>, "reasoning": "<one sentence explaining your scores>"}}
"""


def judge_answer(question: str, answer: str, sources: list[dict]) -> dict:
    """
    Uses the LLM itself to score an answer's faithfulness (grounded in sources,
    no hallucination) and relevance (actually addresses the question).
    """
    sources_text = "\n\n".join(
        f"[Source {s['source_number']}]: {s['chunk_text_preview']}"
        for s in sources
    ) if sources else "(no sources were retrieved)"

    prompt = JUDGE_PROMPT.format(question=question, sources_text=sources_text, answer=answer)


    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt,
                config={"temperature": 0},
            )
            raw_text = response.text.strip()
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)  # 5s, then 10s
                print(f"  (Judge call failed, retrying in {wait_time}s: {e})")
                time.sleep(wait_time)
            else:
                raw_text = '{"faithfulness": null, "relevance": null, "reasoning": "Judge API unavailable after retries"}'

    # Models sometimes wrap JSON in ```json fences despite instructions — strip if present
    cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_text).strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        result = {"faithfulness": None, "relevance": None, "reasoning": f"Could not parse judge response: {raw_text[:100]}"}

    return result
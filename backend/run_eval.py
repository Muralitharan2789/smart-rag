import requests
from eval_questions import EVAL_SET
from eval_judge import judge_answer

API_URL = "http://localhost:8000/query"


def run_eval():
    passed, failed = 0, 0
    faithfulness_scores, relevance_scores = [], []

    for case in EVAL_SET:
        response = requests.post(API_URL, json={"question": case["question"]}).json()
        answer = response["answer"]
        sources = response.get("sources", [])

        # Old-style keyword check (kept as a quick sanity check)
        matched = any(kw.lower() in answer.lower() for kw in case["expected_answer_contains"])
        status = "PASS" if matched else "FAIL"
        if matched:
            passed += 1
        else:
            failed += 1

        # New: LLM-as-judge scoring
        judgment = judge_answer(case["question"], answer, sources)
        if judgment["faithfulness"] is not None:
            faithfulness_scores.append(judgment["faithfulness"])
        if judgment["relevance"] is not None:
            relevance_scores.append(judgment["relevance"])

        print(f"[{status}] {case['question']}")
        print(f"  Answer: {answer[:150]}")
        print(f"  Judge: faithfulness={judgment['faithfulness']}/5, relevance={judgment['relevance']}/5")
        print(f"  Reasoning: {judgment['reasoning']}")
        print()

    print(f"Keyword check: {passed}/{len(EVAL_SET)} passed")
    if faithfulness_scores:
        print(f"Average faithfulness: {sum(faithfulness_scores)/len(faithfulness_scores):.2f}/5")
    if relevance_scores:
        print(f"Average relevance: {sum(relevance_scores)/len(relevance_scores):.2f}/5")


if __name__ == "__main__":
    run_eval()
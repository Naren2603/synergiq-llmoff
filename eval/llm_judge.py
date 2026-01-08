from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from app.core.ollama_client import ollama_chat
from eval.run_meta import current_run_meta


OUT_FIELDNAMES = [
    "run_id",
    "timestamp",
    "git_commit",
    "pdf_name",
    "question_id",
    "method",
    "judge_model",
    "judge_correctness_1to5",
    "judge_groundedness_1to5",
    "judge_citation_relevance_1to5",
    "judge_explanation",
]


def _judge_one(*, question: str, answer: str, citations: str, retrieved: str) -> dict:
    system = (
        "You are an impartial evaluator for a Retrieval-Augmented Generation system. "
        "Score the answer on a 1-5 scale for correctness, groundedness, and citation relevance. "
        "Groundedness means the answer is supported by the retrieved evidence. "
        "Citation relevance means the provided citations align with the evidence and support the answer. "
        "If method is no_rag (no citations/evidence), set citation relevance to 0. "
        "Return ONLY valid JSON with keys: correctness, groundedness, citation_relevance, explanation."
    )

    prompt = (
        "Evaluate the following.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:\n{answer}\n\n"
        f"CITATIONS (may be empty):\n{citations}\n\n"
        f"RETRIEVED_EVIDENCE_PREVIEW (may be empty):\n{retrieved}\n\n"
        "Return JSON only."
    )

    raw = ollama_chat(prompt, system=system, temperature=0.0)
    try:
        data = json.loads(raw)
    except Exception:
        # Last-resort: wrap raw into an explanation
        data = {
            "correctness": 3,
            "groundedness": 3,
            "citation_relevance": 0 if not citations else 3,
            "explanation": f"Non-JSON judge output. Raw: {raw}",
        }

    def _clamp(v: object, lo: int, hi: int, default: int) -> int:
        try:
            iv = int(v)  # type: ignore[arg-type]
        except Exception:
            return default
        return max(lo, min(hi, iv))

    return {
        "judge_correctness_1to5": _clamp(data.get("correctness"), 1, 5, 3),
        "judge_groundedness_1to5": _clamp(data.get("groundedness"), 1, 5, 3),
        "judge_citation_relevance_1to5": (0 if not citations else _clamp(data.get("citation_relevance"), 1, 5, 3)),
        "judge_explanation": str(data.get("explanation") or "").strip(),
    }


def run(in_csv: Path, out_csv: Path) -> None:
    meta = current_run_meta()

    rows = list(csv.DictReader(in_csv.open("r", encoding="utf-8")))
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDNAMES)
        w.writeheader()

        for r in rows:
            question = (r.get("question") or "").strip()
            answer = (r.get("answer") or "").strip()
            citations = (r.get("citations") or "").strip()
            retrieved = (r.get("retrieved_chunks") or "").strip()

            scored = _judge_one(question=question, answer=answer, citations=citations, retrieved=retrieved)

            w.writerow(
                {
                    "run_id": meta.run_id,
                    "timestamp": meta.timestamp,
                    "git_commit": meta.git_commit or "",
                    "pdf_name": r.get("pdf_name") or "",
                    "question_id": r.get("question_id") or "",
                    "method": r.get("method") or "",
                    "judge_model": meta.ollama_model or "",
                    **scored,
                }
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM-judge scoring for qa_results.csv")
    ap.add_argument("--in", dest="in_csv", required=True, help="Input qa_results.csv")
    ap.add_argument("--out", dest="out_csv", required=True, help="Output llmjudge_scores.csv")
    args = ap.parse_args()

    run(Path(args.in_csv), Path(args.out_csv))
    print(f"Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()

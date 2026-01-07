from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from app.core.ollama_client import ollama_chat
from app.core.pdf_loader import extract_pdf_pages
from app.core.rag import answer_with_citations, build_or_load_index, load_faiss, retrieve
from app.core.storage import doc_dir, save_doc_meta, save_doc_pages
from app.core.summarizer import summarize_text
from app.core.summarize import chunk_text


@dataclass(frozen=True)
class QAItem:
    pdf: str
    doc_id: Optional[str]
    questions: list[dict[str, Any]]


def _stable_doc_id_for_pdf(pdf_path: Path) -> str:
    h = hashlib.sha256()
    with pdf_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:24]


def _load_qa_spec(path: Path) -> list[QAItem]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items")
    if not isinstance(items, list) or not items:
        raise SystemExit("QA spec must contain non-empty 'items' list")

    out: list[QAItem] = []
    for it in items:
        pdf = it.get("pdf")
        if not isinstance(pdf, str) or not pdf.strip():
            raise SystemExit("Each item must include a non-empty 'pdf' field")

        doc_id = it.get("doc_id")
        if doc_id is not None and (not isinstance(doc_id, str) or not doc_id.strip()):
            doc_id = None

        questions = it.get("questions")
        if not isinstance(questions, list) or not questions:
            raise SystemExit(f"Item {pdf} must include non-empty 'questions' list")

        out.append(QAItem(pdf=pdf, doc_id=doc_id, questions=questions))

    return out


def _ensure_index(doc_id: str, pages: list[str]) -> None:
    texts: list[str] = []
    metadatas: list[dict] = []
    for page_idx, page_text in enumerate(pages, start=1):
        if not page_text.strip():
            continue
        chunks = chunk_text(page_text, mode="detailed")
        for chunk_idx, chunk in enumerate(chunks):
            if chunk.strip():
                texts.append(chunk)
                metadatas.append({"page": page_idx, "chunk": chunk_idx, "doc_id": doc_id})

    if texts:
        build_or_load_index(doc_id, texts, metadatas)


def _answer_no_rag(question: str) -> str:
    system = (
        "You are a careful assistant. Answer the user's question. "
        "If you do not know, say you don't know. Keep the answer concise."
    )
    return ollama_chat(question, system=system, temperature=0.2)


def _answer_with_rag(doc_id: str, question: str, *, k: int) -> tuple[str, list[str], list[dict]]:
    db = load_faiss(doc_id)
    if db is None:
        raise RuntimeError("FAISS index not found for doc_id")

    evidence = retrieve(db, question, k=k)
    answer, sources = answer_with_citations(question, evidence)
    return answer, sources, evidence


QA_FIELDNAMES = [
    "doc_id",
    "pdf",
    "question_id",
    "question",
    "expected",
    "mode",
    "k",
    "answer",
    "sources",
    "evidence_preview",
    "t_answer_s",
]


def _evidence_preview(evidence: list[dict], *, max_chars: int = 800) -> str:
    parts: list[str] = []
    for ev in evidence:
        meta = ev.get("metadata") or {}
        page = meta.get("page", "?")
        txt = (ev.get("text") or "").strip().replace("\n", " ")
        if not txt:
            continue
        parts.append(f"p{page}: {txt}")
        if sum(len(p) for p in parts) > max_chars:
            break
    out = " | ".join(parts)
    return out[:max_chars]


def write_human_rating_templates(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    qa_ratings = out_dir / "qa_ratings_template.csv"
    with qa_ratings.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "doc_id",
                "pdf",
                "question_id",
                "mode",
                "correctness_1to5",
                "groundedness_1to5",
                "citation_relevance_1to5",
                "notes",
            ]
        )

    summary_ratings = out_dir / "summary_ratings_template.csv"
    with summary_ratings.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "pdf", "mode", "faithfulness_1to5", "coverage_1to5", "coherence_1to5", "notes"])

    rubric = out_dir / "HUMAN_RUBRIC.md"
    rubric.write_text(
        """# Human Rating Rubric (1–5)\n\n## Q/A\n- correctness: 1=wrong, 3=partly correct, 5=fully correct\n- groundedness: 1=hallucinated/unsupported, 3=mixed, 5=fully supported by document\n- citation_relevance (RAG only): 1=irrelevant pages, 3=some relevant, 5=directly supports answer\n\n## Summaries\n- faithfulness: 1=major hallucinations, 3=minor issues, 5=fully faithful\n- coverage: 1=misses most key points, 3=some key points, 5=covers main ideas well\n- coherence: 1=hard to read, 3=ok, 5=clear and well structured\n""",
        encoding="utf-8",
    )


def run_quality_eval(
    *,
    qa_spec_path: Path,
    pdf_dir: Path,
    out_dir: Path,
    k: int,
    do_summary: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    items = _load_qa_spec(qa_spec_path)

    qa_results_csv = out_dir / "qa_results.csv"
    qa_results_jsonl = out_dir / "qa_results.jsonl"

    with qa_results_csv.open("w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=QA_FIELDNAMES)
        writer.writeheader()

    qa_results_jsonl.write_text("", encoding="utf-8")

    for item in items:
        pdf_path = pdf_dir / item.pdf
        if not pdf_path.exists():
            raise SystemExit(f"Missing PDF: {pdf_path}")

        doc_id = item.doc_id or _stable_doc_id_for_pdf(pdf_path)

        # Extract pages (OCR empty pages)
        pages, num_pages = extract_pdf_pages(str(pdf_path), ocr_empty_pages=True)
        save_doc_pages(doc_id, pages)
        save_doc_meta(doc_id, {"doc_id": doc_id, "filename": pdf_path.name, "num_pages": num_pages})

        # Ensure FAISS index exists for RAG mode
        _ensure_index(doc_id, pages)

        # Generate summaries once per doc (optional)
        if do_summary:
            full_text = "\n\n".join(pages)
            detailed = summarize_text(full_text, mode="detailed")
            brief = summarize_text(full_text, mode="brief")
            (doc_dir(doc_id) / "summary_detailed.txt").write_text(detailed, encoding="utf-8")
            (doc_dir(doc_id) / "summary_brief.txt").write_text(brief, encoding="utf-8")

        for q in item.questions:
            qid = str(q.get("id") or "")
            question = str(q.get("question") or "").strip()
            expected = str(q.get("expected") or "").strip()

            if not question:
                continue

            # no-RAG baseline
            t0 = time.perf_counter()
            ans0 = _answer_no_rag(question)
            t_ans0 = time.perf_counter() - t0

            row0 = {
                "doc_id": doc_id,
                "pdf": pdf_path.name,
                "question_id": qid,
                "question": question,
                "expected": expected,
                "mode": "no_rag",
                "k": 0,
                "answer": ans0,
                "sources": "",
                "evidence_preview": "",
                "t_answer_s": round(t_ans0, 4),
            }

            # RAG mode
            t1 = time.perf_counter()
            ans1, sources1, evidence1 = _answer_with_rag(doc_id, question, k=k)
            t_ans1 = time.perf_counter() - t1

            row1 = {
                "doc_id": doc_id,
                "pdf": pdf_path.name,
                "question_id": qid,
                "question": question,
                "expected": expected,
                "mode": "rag",
                "k": k,
                "answer": ans1,
                "sources": ";".join(sources1),
                "evidence_preview": _evidence_preview(evidence1),
                "t_answer_s": round(t_ans1, 4),
            }

            # Append to CSV and JSONL
            for row in (row0, row1):
                with qa_results_csv.open("a", newline="", encoding="utf-8") as f_csv:
                    writer = csv.DictWriter(f_csv, fieldnames=QA_FIELDNAMES)
                    writer.writerow(row)

                with qa_results_jsonl.open("a", encoding="utf-8") as f_jsonl:
                    f_jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_human_rating_templates(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Quality evaluation: no-RAG vs RAG Q/A + (optional) summaries.")
    ap.add_argument("--qa", required=True, type=str, help="Path to QA spec JSON (see eval/qa_example.json)")
    ap.add_argument("--pdf-dir", required=True, type=str, help="Folder containing the PDFs referenced in QA spec")
    ap.add_argument("--out", default="eval/out_quality", type=str, help="Output directory")
    ap.add_argument("--k", default=5, type=int, help="Top-k retrieved chunks for RAG")
    ap.add_argument("--summary", action="store_true", help="Also generate brief+detailed summaries per PDF")
    args = ap.parse_args()

    run_quality_eval(
        qa_spec_path=Path(args.qa),
        pdf_dir=Path(args.pdf_dir),
        out_dir=Path(args.out),
        k=int(args.k),
        do_summary=bool(args.summary),
    )

    print(f"Wrote: {Path(args.out) / 'qa_results.csv'}")
    print(f"Wrote: {Path(args.out) / 'qa_results.jsonl'}")
    print(f"Rating templates: {Path(args.out)}")


if __name__ == "__main__":
    main()

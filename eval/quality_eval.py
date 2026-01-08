from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from app.core.ollama_client import ollama_chat
from app.core.pdf_loader import extract_pdf_pages
from app.core.rag import answer_with_citations, build_or_load_index, load_faiss, retrieve
from app.core.storage import doc_dir, save_doc_meta, save_doc_pages
from app.core.summarizer import summarize_text
from app.core.summarize import chunk_text
from eval.run_meta import current_run_meta


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


def _normalize_expected(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for v in value:
            s = str(v).strip()
            if s:
                parts.append(f"- {s}")
        return "\n".join(parts)
    return str(value).strip()


def _load_qa_spec(path: Path) -> list[QAItem]:
    data = json.loads(path.read_text(encoding="utf-8"))

    items = data.get("items")
    pdfs = data.get("pdfs")

    # Support two formats:
    # A) {"items": [{"pdf": "x.pdf", "doc_id": "...", "questions": [{"id","question","expected"}]}]}
    # B) {"pdfs":  [{"file": "x.pdf", "id": "...", "questions": [{"id","text","expected"}]}]}
    if items is None and pdfs is None:
        raise SystemExit("QA spec must contain either 'items' or 'pdfs' list")

    if items is None:
        items = []
        if not isinstance(pdfs, list) or not pdfs:
            raise SystemExit("QA spec 'pdfs' must be a non-empty list")
        for p in pdfs:
            items.append(
                {
                    "pdf": p.get("file"),
                    "doc_id": p.get("id"),
                    "questions": [
                        {
                            "id": q.get("id"),
                            "question": q.get("text"),
                            "expected": q.get("expected"),
                        }
                        for q in (p.get("questions") or [])
                    ],
                }
            )

    if not isinstance(items, list) or not items:
        raise SystemExit("QA spec must contain a non-empty list of PDFs")

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
    # run identity
    "run_id",
    "timestamp",
    "git_commit",

    # document
    "pdf_id",
    "pdf_name",
    "pages",

    # question
    "question_id",
    "question",
    "expected",

    # method
    "method",
    "top_k",

    # model/settings
    "model_name",
    "embed_model",
    "chunk_size",
    "chunk_overlap",

    # outputs
    "answer",
    "citations",
    "retrieved_chunks",
    "latency_s",
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
                "pdf_name",
                "question_id",
                "method",
                "answer",
                "citations",
                "correctness_1to5",
                "groundedness_1to5",
                "citation_relevance_1to5",
                "notes",
                "rater_id",
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


def _write_filled_rating_rows(
    *,
    out_dir: Path,
    qa_rows: list[dict[str, Any]],
    did_write_summaries: bool,
) -> None:
    """Write filled human-rating CSVs from computed outputs.

    This matches the paper workflow: raters should not need to copy/paste answers.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Q/A ratings sheet (2 rows per question: rag + no_rag)
    qa_ratings = out_dir / "qa_ratings_template.csv"
    with qa_ratings.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "pdf_name",
                "question_id",
                "method",
                "answer",
                "citations",
                "correctness_1to5",
                "groundedness_1to5",
                "citation_relevance_1to5",
                "notes",
                "rater_id",
            ]
        )
        for r in qa_rows:
            w.writerow(
                [
                    r.get("pdf_name", ""),
                    r.get("question_id", ""),
                    r.get("method", ""),
                    r.get("answer", ""),
                    r.get("citations", ""),
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )

    # Summary ratings sheet (per doc, per mode)
    summary_ratings = out_dir / "summary_ratings_template.csv"
    with summary_ratings.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "pdf", "mode", "summary", "faithfulness_1to5", "coverage_1to5", "coherence_1to5", "notes"])

        if did_write_summaries:
            # Summaries are saved to data/docs/<doc_id>/summary_*.txt; include them to rate.
            seen_docs: set[tuple[str, str]] = set()
            for r in qa_rows:
                doc_id = str(r.get("pdf_id") or "").strip()
                pdf = str(r.get("pdf_name") or "").strip()
                if not doc_id or not pdf:
                    continue
                if (doc_id, pdf) in seen_docs:
                    continue
                seen_docs.add((doc_id, pdf))

                for mode in ("brief", "detailed"):
                    p = doc_dir(doc_id) / f"summary_{mode}.txt"
                    summary_text = p.read_text(encoding="utf-8") if p.exists() else ""
                    w.writerow([doc_id, pdf, mode, summary_text, "", "", "", ""])


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

    meta = current_run_meta()
    # Use env defaults if not explicitly provided
    chunk_size = meta.chunk_size
    chunk_overlap = meta.chunk_overlap
    model_name = meta.ollama_model
    embed_model = meta.embed_model

    qa_results_csv = out_dir / "qa_results.csv"
    qa_results_jsonl = out_dir / "qa_results.jsonl"

    with qa_results_csv.open("w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=QA_FIELDNAMES)
        writer.writeheader()

    qa_results_jsonl.write_text("", encoding="utf-8")

    qa_rows: list[dict[str, Any]] = []

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
            expected = _normalize_expected(q.get("expected"))

            if not question:
                continue

            # no-RAG baseline
            t0 = time.perf_counter()
            ans0 = _answer_no_rag(question)
            t_ans0 = time.perf_counter() - t0

            row0 = {
                "run_id": meta.run_id,
                "timestamp": meta.timestamp,
                "git_commit": meta.git_commit or "",
                "pdf_id": doc_id,
                "pdf_name": pdf_path.name,
                "pages": num_pages,
                "question_id": qid,
                "question": question,
                "expected": expected,
                "method": "no_rag",
                "top_k": 0,
                "model_name": model_name or "",
                "embed_model": embed_model or "",
                "chunk_size": chunk_size or "",
                "chunk_overlap": chunk_overlap or "",
                "answer": ans0,
                "citations": "",
                "retrieved_chunks": "",
                "latency_s": round(t_ans0, 4),
            }

            # RAG mode
            t1 = time.perf_counter()
            ans1, sources1, evidence1 = _answer_with_rag(doc_id, question, k=k)
            t_ans1 = time.perf_counter() - t1

            row1 = {
                "run_id": meta.run_id,
                "timestamp": meta.timestamp,
                "git_commit": meta.git_commit or "",
                "pdf_id": doc_id,
                "pdf_name": pdf_path.name,
                "pages": num_pages,
                "question_id": qid,
                "question": question,
                "expected": expected,
                "method": "rag",
                "top_k": k,
                "model_name": model_name or "",
                "embed_model": embed_model or "",
                "chunk_size": chunk_size or "",
                "chunk_overlap": chunk_overlap or "",
                "answer": ans1,
                "citations": "|".join(sources1),
                "retrieved_chunks": _evidence_preview(evidence1),
                "latency_s": round(t_ans1, 4),
            }

            # Append to CSV and JSONL
            for row in (row0, row1):
                qa_rows.append(row)
                with qa_results_csv.open("a", newline="", encoding="utf-8") as f_csv:
                    writer = csv.DictWriter(f_csv, fieldnames=QA_FIELDNAMES)
                    writer.writerow(row)

                with qa_results_jsonl.open("a", encoding="utf-8") as f_jsonl:
                    f_jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Write rubric + filled rating templates
    write_human_rating_templates(out_dir)
    _write_filled_rating_rows(out_dir=out_dir, qa_rows=qa_rows, did_write_summaries=do_summary)

    # Latency summary (p50/p95) per method for paper reporting
    def _p(vals: list[float], q: float) -> float:
        if not vals:
            return 0.0
        vals = sorted(vals)
        idx = int(round((len(vals) - 1) * q))
        return float(vals[max(0, min(len(vals) - 1, idx))])

    by_method: dict[str, list[float]] = {}
    for r in qa_rows:
        m = str(r.get("method") or "")
        try:
            lat = float(r.get("latency_s") or 0.0)
        except Exception:
            lat = 0.0
        by_method.setdefault(m, []).append(lat)

    summary = {
        "run_id": meta.run_id,
        "timestamp": meta.timestamp,
        "methods": {},
    }
    for m, vals in by_method.items():
        if not vals:
            continue
        summary["methods"][m] = {
            "n": len(vals),
            "p50_s": round(_p(vals, 0.50), 4),
            "p95_s": round(_p(vals, 0.95), 4),
            "mean_s": round(float(statistics.mean(vals)), 4),
            "max_s": round(float(max(vals)), 4),
        }

    (out_dir / "latency_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


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

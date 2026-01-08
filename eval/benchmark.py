from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
import uuid
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

from app.core.pdf_loader import extract_pdf_pages
from app.core.rag import build_or_load_index
from app.core.storage import doc_dir, index_dir, save_doc_meta, save_doc_pages, save_status
from app.core.summarize import chunk_text
from app.core.summarizer import summarize_text
from app.core.tts import generate_audio


FIELDNAMES = [
    "doc_id",
    "file",
    "pages",
    "pdf_mb",
    "extracted_chars",
    "chunks",
    "avg_chunk_chars",
    "extract_s",
    "index_s",
    "summary_s",
    "tts_s",
    "total_s",
    "extract_s_per_page",
    "index_s_per_page",
    "summary_s_per_page",
    "tts_s_per_page",
    "index_s_per_chunk",
    "pages_per_sec",
    "index_mb",
    "doc_mb",
    "summary_chars",
    "compression_ratio",
    "error",
]


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def run_one(pdf_path: Path, *, do_summary: bool, do_tts: bool) -> dict:
    doc_id = str(uuid.uuid4())
    save_status(doc_id, {"state": "benchmark", "step": "start"})

    pdf_mb = round(pdf_path.stat().st_size / (1024 * 1024), 3)

    t0 = time.perf_counter()
    save_status(doc_id, {"state": "benchmark", "step": "extracting_pages"})
    pages, num_pages = extract_pdf_pages(str(pdf_path), ocr_empty_pages=True)
    t_extract = time.perf_counter() - t0

    extracted_text = "\n\n".join(pages)
    extracted_chars = len(extracted_text)

    save_doc_pages(doc_id, pages)
    save_doc_meta(doc_id, {"doc_id": doc_id, "filename": pdf_path.name, "num_pages": num_pages})

    t1 = time.perf_counter()
    save_status(doc_id, {"state": "benchmark", "step": "building_vectorstore"})
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

    num_chunks = len(texts)
    avg_chunk_chars = round(statistics.mean([len(t) for t in texts]), 2) if texts else 0.0

    if texts:
        build_or_load_index(doc_id, texts, metadatas)
    t_index = time.perf_counter() - t1

    summary_chars = 0
    compression_ratio = 0.0
    t_summary = 0.0
    t_tts = 0.0
    if do_summary:
        t2 = time.perf_counter()
        save_status(doc_id, {"state": "benchmark", "step": "summarizing"})
        summary = summarize_text(extracted_text, mode="detailed")
        t_summary = time.perf_counter() - t2
        summary_chars = len(summary)
        compression_ratio = round(_safe_div(float(summary_chars), float(extracted_chars)), 6)

        if do_tts:
            t3 = time.perf_counter()
            # Use detailed summary for TTS benchmarking.
            audio_path = str(doc_dir(doc_id) / "bench_audio.mp3")
            _ = generate_audio(summary, audio_path, use_edge=True)
            t_tts = time.perf_counter() - t3

    idx_size = _dir_size_bytes(index_dir(doc_id))
    doc_size = _dir_size_bytes(doc_dir(doc_id))

    total = t_extract + t_index + t_summary + t_tts
    save_status(doc_id, {"state": "benchmark", "step": "done"})

    pages_f = float(num_pages or 0)
    total = t_extract + t_index + t_summary + t_tts
    pages_per_sec = round(_safe_div(pages_f, total), 4)
    return {
        "doc_id": doc_id,
        "file": pdf_path.name,
        "pages": num_pages,
        "pdf_mb": pdf_mb,
        "extracted_chars": extracted_chars,
        "chunks": num_chunks,
        "avg_chunk_chars": avg_chunk_chars,
        "extract_s": round(t_extract, 4),
        "index_s": round(t_index, 4),
        "summary_s": round(t_summary, 4),
        "tts_s": round(t_tts, 4),
        "total_s": round(total, 4),
        "extract_s_per_page": round(_safe_div(t_extract, pages_f), 6),
        "index_s_per_page": round(_safe_div(t_index, pages_f), 6),
        "summary_s_per_page": round(_safe_div(t_summary, pages_f), 6),
        "tts_s_per_page": round(_safe_div(t_tts, pages_f), 6),
        "index_s_per_chunk": round(_safe_div(t_index, float(num_chunks)), 6),
        "pages_per_sec": pages_per_sec,
        "index_mb": round(idx_size / (1024 * 1024), 3),
        "doc_mb": round(doc_size / (1024 * 1024), 3),
        "summary_chars": summary_chars,
        "compression_ratio": compression_ratio,
        "error": "",
    }


def save_plots(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    pages = [r["pages"] for r in rows]
    total_s = [r["total_s"] for r in rows]
    extract_s = [r["extract_s"] for r in rows]
    index_s = [r["index_s"] for r in rows]
    summary_s = [r["summary_s"] for r in rows]
    tts_s = [r.get("tts_s", 0.0) for r in rows]
    pages_per_sec = [r.get("pages_per_sec", 0.0) for r in rows]
    chunks = [r.get("chunks", 0) for r in rows]
    index_s_per_page = [r.get("index_s_per_page", 0.0) for r in rows]

    # Line: total time vs pages
    plt.figure(figsize=(7, 4))
    plt.plot(pages, total_s, marker="o")
    plt.xlabel("Pages")
    plt.ylabel("Total time (s)")
    plt.title("Pipeline time vs PDF length")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "time_vs_pages.png", dpi=200)
    plt.close()

    # Stacked bar: stage breakdown
    plt.figure(figsize=(8, 4.5))
    x = list(range(len(rows)))
    plt.bar(x, extract_s, label="Extract/OCR")
    plt.bar(x, index_s, bottom=extract_s, label="Index")
    bottoms = [a + b for a, b in zip(extract_s, index_s)]
    plt.bar(x, summary_s, bottom=bottoms, label="Summarize")
    bottoms2 = [a + b for a, b in zip(bottoms, summary_s)]
    if any(float(v) > 0 for v in tts_s):
        plt.bar(x, tts_s, bottom=bottoms2, label="TTS")
    plt.xticks(x, [str(p) for p in pages], rotation=0)
    plt.xlabel("Pages")
    plt.ylabel("Time (s)")
    plt.title("Stage time breakdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "stage_breakdown.png", dpi=200)
    plt.close()

    # Line: index size vs pages
    index_mb = [r["index_mb"] for r in rows]
    plt.figure(figsize=(7, 4))
    plt.plot(pages, index_mb, marker="o")
    plt.xlabel("Pages")
    plt.ylabel("Index size (MB)")
    plt.title("FAISS index size vs PDF length")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "index_size_vs_pages.png", dpi=200)
    plt.close()

    # Line: indexing time per page vs pages
    plt.figure(figsize=(7, 4))
    plt.plot(pages, index_s_per_page, marker="o")
    plt.xlabel("Pages")
    plt.ylabel("Index time per page (s/page)")
    plt.title("Indexing cost per page")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "index_time_per_page.png", dpi=200)
    plt.close()

    # Line: throughput pages/sec vs pages
    plt.figure(figsize=(7, 4))
    plt.plot(pages, pages_per_sec, marker="o")
    plt.xlabel("Pages")
    plt.ylabel("Throughput (pages/s)")
    plt.title("Pipeline throughput")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "throughput_pages_per_sec.png", dpi=200)
    plt.close()

    # Scatter: chunks vs pages
    plt.figure(figsize=(7, 4))
    plt.scatter(pages, chunks)
    plt.xlabel("Pages")
    plt.ylabel("Chunks indexed")
    plt.title("Chunks vs PDF length")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "chunks_vs_pages.png", dpi=200)
    plt.close()


def save_summary_stats(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def series(key: str) -> list[float]:
        vals: list[float] = []
        for r in rows:
            v = r.get(key)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        return vals

    def stat(vals: list[float]) -> dict:
        if not vals:
            return {"n": 0}
        return {
            "n": len(vals),
            "min": round(min(vals), 6),
            "p50": round(statistics.median(vals), 6),
            "mean": round(statistics.mean(vals), 6),
            "max": round(max(vals), 6),
        }

    stats = {
        "pages": stat(series("pages")),
        "pdf_mb": stat(series("pdf_mb")),
        "extract_s": stat(series("extract_s")),
        "index_s": stat(series("index_s")),
        "summary_s": stat(series("summary_s")),
        "total_s": stat(series("total_s")),
        "index_mb": stat(series("index_mb")),
        "chunks": stat(series("chunks")),
        "pages_per_sec": stat(series("pages_per_sec")),
        "index_s_per_page": stat(series("index_s_per_page")),
        "compression_ratio": stat(series("compression_ratio")),
    }

    (out_dir / "summary_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark synergiq-llmoff pipeline and generate CSV + plots.")
    ap.add_argument("pdf_dir", type=str, help="Folder containing PDFs")
    ap.add_argument("--out", type=str, default="eval/out", help="Output folder")
    ap.add_argument("--summary", action="store_true", help="Include summarization timing (requires Ollama model)")
    ap.add_argument("--tts", action="store_true", help="Include TTS timing (runs only if --summary is also set)")
    ap.add_argument("--sleep", type=float, default=0.5, help="Sleep seconds between PDFs to reduce Ollama load")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted([p for p in pdf_dir.rglob("*.pdf") if p.is_file()])
    if not pdfs:
        raise SystemExit(f"No PDFs found under: {pdf_dir}")

    csv_path = out_dir / "results.csv"
    rows: list[dict] = []

    # Write header upfront so partial runs still produce usable outputs.
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

    for p in pdfs:
        print(f"Benchmarking: {p}")
        try:
            row = run_one(p, do_summary=bool(args.summary), do_tts=bool(args.tts) and bool(args.summary))
        except Exception as e:
            row = {k: 0 for k in FIELDNAMES}
            row["doc_id"] = ""
            row["file"] = p.name
            row["pages"] = 0
            row["pdf_mb"] = round(p.stat().st_size / (1024 * 1024), 3)
            row["error"] = str(e)
            print(f"FAILED: {p.name}: {e}")

        rows.append(row)

        # Append row immediately
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writerow({k: row.get(k, 0) for k in FIELDNAMES})

        if args.sleep and args.sleep > 0:
            time.sleep(args.sleep)

    ok_rows = [r for r in rows if not r.get("error")]
    # Sort by pages for nicer plots
    ok_rows.sort(key=lambda r: r.get("pages", 0))

    plots_dir = out_dir / "plots"
    if ok_rows:
        save_plots(ok_rows, plots_dir)
        save_summary_stats(ok_rows, out_dir)
    else:
        print("No successful rows; skipping plots/stats.")

    print(f"Wrote: {csv_path}")
    if ok_rows:
        print(f"Plots: {plots_dir}")
        print(f"Stats: {out_dir / 'summary_stats.json'}")


if __name__ == "__main__":
    main()

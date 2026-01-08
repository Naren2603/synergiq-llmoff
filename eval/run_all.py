from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from eval.run_meta import current_run_meta, ensure_run_id, run_meta_dict


def _run(cmd: list[str]) -> None:
    print("\n[run_all] $ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Single-command end-to-end evaluation for the paper.")
    ap.add_argument("--qa", required=True, help="Path to QA JSON spec")
    ap.add_argument("--pdf-dir", required=True, help="Folder containing PDFs referenced in QA spec")
    ap.add_argument("--out", default="eval/out_paper", help="Output folder")
    ap.add_argument("--k", type=int, default=5, help="Top-k retrieval")
    ap.add_argument("--seed", type=int, default=1337, help="Seed for run identity")
    ap.add_argument("--summary", action="store_true", help="Generate summaries")
    ap.add_argument("--benchmark", action="store_true", help="Run benchmark (performance)")
    ap.add_argument("--benchmark-tts", action="store_true", help="Include TTS timing in benchmark")
    args = ap.parse_args()

    os.environ["SYNERGIQ_SEED"] = str(args.seed)
    ensure_run_id()
    meta = current_run_meta()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write run metadata once per run
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta_dict(), indent=2), encoding="utf-8")

    # 1) QA generation (no_rag vs rag) + rating templates
    qa_out = out_dir
    _run(
        [
            sys.executable,
            "-m",
            "eval.quality_eval",
            "--qa",
            args.qa,
            "--pdf-dir",
            args.pdf_dir,
            "--out",
            str(qa_out),
            "--k",
            str(args.k),
        ]
        + (["--summary"] if args.summary else [])
    )

    # 2) LLM-judge
    in_csv = qa_out / "qa_results.csv"
    judge_out = qa_out / "llmjudge_scores.csv"
    if in_csv.exists():
        _run([sys.executable, "-m", "eval.llm_judge", "--in", str(in_csv), "--out", str(judge_out)])
    else:
        print(f"[run_all] WARNING: Missing {in_csv}; skipping LLM-judge")

    # 3) Benchmark (optional)
    if args.benchmark:
        bench_out = out_dir / "benchmark"
        bench_out.mkdir(parents=True, exist_ok=True)
        _run(
            [sys.executable, "-m", "eval.benchmark", args.pdf_dir, "--out", str(bench_out), "--summary"]
            + (["--tts"] if args.benchmark_tts else [])
        )

    print(f"\n[run_all] DONE. run_id={meta.run_id}")
    print(f"[run_all] Outputs: {out_dir}")


if __name__ == "__main__":
    main()

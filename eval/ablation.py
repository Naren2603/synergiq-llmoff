from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


def _p(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    vals = sorted(vals)
    idx = int(round((len(vals) - 1) * q))
    return float(vals[max(0, min(len(vals) - 1, idx))])


def _run(cmd: list[str], env: dict[str, str]) -> None:
    print("\n[ablation] $ " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run ablation sweep and write ablation_results.csv")
    ap.add_argument("--qa", required=True, help="Path to QA JSON spec")
    ap.add_argument("--pdf-dir", required=True, help="Folder containing PDFs")
    ap.add_argument("--out", default="eval/out_ablation", help="Output folder")
    ap.add_argument("--topk", default="3,5,10", help="Comma-separated top-k values")
    ap.add_argument("--chunk-size", default="600,900,1200", help="Comma-separated chunk sizes")
    ap.add_argument("--chunk-overlap", default="0,150,200", help="Comma-separated overlaps")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    topks = [int(x.strip()) for x in str(args.topk).split(",") if x.strip()]
    chunk_sizes = [int(x.strip()) for x in str(args.chunk_size).split(",") if x.strip()]
    overlaps = [int(x.strip()) for x in str(args.chunk_overlap).split(",") if x.strip()]

    results_path = out_dir / "ablation_results.csv"
    fieldnames = [
        "chunk_size",
        "chunk_overlap",
        "top_k",
        "p95_latency_s",
        "judge_correctness_avg",
        "judge_groundedness_avg",
        "judge_citation_relevance_avg",
        "run_dir",
        "notes",
    ]

    with results_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for cs in chunk_sizes:
            for co in overlaps:
                for k in topks:
                    run_subdir = out_dir / f"cs{cs}_co{co}_k{k}"
                    run_subdir.mkdir(parents=True, exist_ok=True)

                    env = os.environ.copy()
                    env["CHUNK_SIZE"] = str(cs)
                    env["CHUNK_OVERLAP"] = str(co)
                    env["TOP_K"] = str(k)
                    env["SYNERGIQ_SEED"] = str(args.seed)

                    _run(
                        [
                            sys.executable,
                            "-m",
                            "eval.run_all",
                            "--qa",
                            args.qa,
                            "--pdf-dir",
                            args.pdf_dir,
                            "--out",
                            str(run_subdir),
                            "--k",
                            str(k),
                            "--seed",
                            str(args.seed),
                            "--summary",
                        ],
                        env,
                    )

                    qa_rows = _read_csv(run_subdir / "qa_results.csv")
                    latencies = [float(r.get("latency_s") or 0.0) for r in qa_rows if (r.get("method") == "rag")]
                    p95 = round(_p(latencies, 0.95), 4)

                    judge_rows = _read_csv(run_subdir / "llmjudge_scores.csv")

                    def avg(key: str) -> float:
                        vals: list[float] = []
                        for r in judge_rows:
                            try:
                                vals.append(float(r.get(key) or 0.0))
                            except Exception:
                                pass
                        return round(float(statistics.mean(vals)), 4) if vals else 0.0

                    w.writerow(
                        {
                            "chunk_size": cs,
                            "chunk_overlap": co,
                            "top_k": k,
                            "p95_latency_s": p95,
                            "judge_correctness_avg": avg("judge_correctness_1to5"),
                            "judge_groundedness_avg": avg("judge_groundedness_1to5"),
                            "judge_citation_relevance_avg": avg("judge_citation_relevance_1to5"),
                            "run_dir": str(run_subdir),
                            "notes": "",
                        }
                    )

    print(f"Wrote: {results_path}")


if __name__ == "__main__":
    main()

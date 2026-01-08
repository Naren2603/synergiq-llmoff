from __future__ import annotations

import os
import platform
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Optional


def _git_commit() -> Optional[str]:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return r.stdout.strip() or None
    except Exception:
        return None


@dataclass(frozen=True)
class RunMeta:
    run_id: str
    timestamp: str
    git_commit: Optional[str]
    python: str
    os: str

    ollama_model: Optional[str]
    embed_model: Optional[str]
    chunk_size: Optional[str]
    chunk_overlap: Optional[str]
    top_k: Optional[str]

    seed: Optional[str]


def ensure_run_id() -> str:
    run_id = os.getenv("SYNERGIQ_RUN_ID")
    if run_id and run_id.strip():
        return run_id.strip()

    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    os.environ["SYNERGIQ_RUN_ID"] = run_id
    return run_id


def current_run_meta() -> RunMeta:
    run_id = ensure_run_id()
    return RunMeta(
        run_id=run_id,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        git_commit=_git_commit(),
        python=platform.python_version(),
        os=f"{platform.system()} {platform.release()}",
        ollama_model=os.getenv("OLLAMA_MODEL"),
        embed_model=os.getenv("EMBED_MODEL"),
        chunk_size=os.getenv("CHUNK_SIZE"),
        chunk_overlap=os.getenv("CHUNK_OVERLAP"),
        top_k=os.getenv("TOP_K"),
        seed=os.getenv("SYNERGIQ_SEED"),
    )


def run_meta_dict(extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    d = asdict(current_run_meta())
    if extra:
        d.update(extra)
    return d

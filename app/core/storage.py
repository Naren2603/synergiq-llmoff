from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DOCS_DIR = DATA_DIR / "docs"
INDICES_DIR = DATA_DIR / "indices"


def ensure_dirs() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    INDICES_DIR.mkdir(parents=True, exist_ok=True)


def doc_dir(doc_id: str) -> Path:
    ensure_dirs()
    d = DOCS_DIR / doc_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_doc_meta(doc_id: str, meta: Dict[str, Any]) -> Path:
    p = doc_dir(doc_id) / "meta.json"
    write_json(p, meta)
    return p


def load_doc_meta(doc_id: str) -> Optional[Dict[str, Any]]:
    p = DOCS_DIR / doc_id / "meta.json"
    return read_json(p)


def save_doc_pages(doc_id: str, pages: list[str]) -> Path:
    p = doc_dir(doc_id) / "pages.json"
    write_json(p, {"pages": pages})
    return p


def load_doc_pages(doc_id: str) -> Optional[list[str]]:
    p = DOCS_DIR / doc_id / "pages.json"
    data = read_json(p)
    if not data:
        return None
    return data.get("pages")


def status_path(doc_id: str) -> Path:
    return doc_dir(doc_id) / "status.json"


def save_status(doc_id: str, status: Dict[str, Any]) -> Path:
    p = status_path(doc_id)
    write_json(p, status)
    return p


def load_status(doc_id: str) -> Optional[Dict[str, Any]]:
    p = DOCS_DIR / doc_id / "status.json"
    return read_json(p)


def index_dir(doc_id: str) -> Path:
    ensure_dirs()
    d = INDICES_DIR / doc_id
    d.mkdir(parents=True, exist_ok=True)
    return d

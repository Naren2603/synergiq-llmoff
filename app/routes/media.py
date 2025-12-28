from __future__ import annotations

import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, Query

from app.core.storage import (
    save_doc_meta,
    save_doc_pages,
    save_status,
    load_status,
    load_doc_meta,
)
from app.core.pdf_loader import extract_pdf_pages
from app.core.summarize import chunk_text

router = APIRouter(prefix="/media", tags=["media"])


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF supported")

    doc_id = str(uuid.uuid4())
    save_status(doc_id, {"state": "processing"})

    pdf_bytes = await file.read()
    # Persist original PDF
    from pathlib import Path
    from app.core.storage import doc_dir

    pdf_path = doc_dir(doc_id) / "source.pdf"
    pdf_path.write_bytes(pdf_bytes)

    pages, num_pages = extract_pdf_pages(str(pdf_path), ocr_empty_pages=True)
    save_doc_pages(doc_id, pages)

    meta = {
        "doc_id": doc_id,
        "filename": file.filename,
        "num_pages": num_pages,
    }
    save_doc_meta(doc_id, meta)

    save_status(doc_id, {"state": "ready", "num_pages": num_pages})
    return meta


@router.get("/status/{doc_id}")
def status(doc_id: str):
    st = load_status(doc_id)
    if not st:
        raise HTTPException(status_code=404, detail="Unknown doc_id")
    return st


@router.get("/summary/{doc_id}")
def summary(doc_id: str, mode: str = Query("detailed", pattern="^(brief|detailed)$")):
    meta = load_doc_meta(doc_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Unknown doc_id")

    # Placeholder summarization pipeline; keep compatible with existing callers.
    # Fix bug: doc[["summary"]] -> doc["summary"]
    from app.core.storage import load_doc_pages, save_status

    pages = load_doc_pages(doc_id)
    if not pages:
        raise HTTPException(status_code=404, detail="No pages stored for doc")

    text = "\n\n".join(pages)
    chunks = chunk_text(text, mode=mode)  # type: ignore[arg-type]

    save_status(doc_id, {"state": "summarizing", "chunks": len(chunks)})

    # If your project already has an LLM summarizer, wire it here.
    # For now, return a simple extractive summary for production safety.
    if mode == "brief":
        summary_text = "\n".join([c[:200] for c in chunks[:5]]).strip()
    else:
        summary_text = "\n".join([c[:400] for c in chunks[:10]]).strip()

    save_status(doc_id, {"state": "ready", "summary_mode": mode})

    return {
        "doc_id": doc_id,
        "mode": mode,
        "summary": summary_text,
        "num_pages": meta.get("num_pages"),
    }

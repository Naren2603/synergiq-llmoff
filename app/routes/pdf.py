import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.state import DOCS
from app.core.pdf_utils import file_id_from_bytes, extract_pages, clean_text, is_low_information
from app.core.rag import build_vectorstore

router = APIRouter()


@router.post("/upload_pdf", include_in_schema=False)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    data = await file.read()
    doc_id = file_id_from_bytes(data)

    if doc_id in DOCS:
        return {"doc_id": doc_id, "status": "already_loaded"}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    pages = extract_pages(tmp_path)
    for p in pages:
        p["text"] = clean_text(p.get("text") or "")

    full_text = "\n".join(p["text"] for p in pages if p["text"]).strip()
    if is_low_information(full_text):
        raise HTTPException(
            status_code=400,
            detail="Low-information PDF text detected (likely scanned/protected). Use OCR/searchable PDF.",
        )

    db, _docs = build_vectorstore(pages, doc_id=doc_id)

    DOCS[doc_id] = {
        "pages": pages,
        "text": full_text,
        "db": db,
        "summary": None,
        "audio_path": None,
        "video_path": None,
    }

    return {"doc_id": doc_id, "pages": len(pages)}

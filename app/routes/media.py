from __future__ import annotations

import os
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse

from app.core.storage import (
    save_doc_meta,
    save_doc_pages,
    save_status,
    load_status,
    load_doc_meta,
    load_doc_pages,
    doc_dir,
)
from app.core.pdf_loader import extract_pdf_pages
from app.core.summarize import chunk_text
from app.core.summarizer import summarize_text
from app.core.tts import generate_audio
from app.core.video import generate_video
from app.core.rag import build_or_load_index, load_faiss, retrieve, answer_with_citations
from pydantic import BaseModel

router = APIRouter(prefix="/media", tags=["media"])


class ChatRequest(BaseModel):
    doc_id: str
    question: str


def _process_document(doc_id: str, filename: str) -> None:
    """Run the full pipeline for an uploaded PDF.

    This runs in a background task so /media/upload can return immediately.
    """
    try:
        pdf_path = doc_dir(doc_id) / "source.pdf"
        if not pdf_path.exists():
            raise RuntimeError("source.pdf not found for doc")

        # Extract pages with OCR for empty pages
        save_status(doc_id, {"state": "processing", "step": "extracting_pages"})
        pages, num_pages = extract_pdf_pages(str(pdf_path), ocr_empty_pages=True)
        save_doc_pages(doc_id, pages)

        meta = {
            "doc_id": doc_id,
            "filename": filename,
            "num_pages": num_pages,
        }
        save_doc_meta(doc_id, meta)

        # Build vector store for RAG (chunk-level)
        save_status(doc_id, {"state": "processing", "step": "building_vectorstore"})
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

        # Generate summary
        save_status(doc_id, {"state": "summarizing", "step": "generating_summary"})
        full_text = "\n\n".join(pages)
        summary = summarize_text(full_text, mode="detailed")

        summary_path = doc_dir(doc_id) / "summary.txt"
        summary_path.write_text(summary, encoding="utf-8")

        # Generate audio from summary
        save_status(doc_id, {"state": "tts", "step": "generating_audio"})
        audio_path = str(doc_dir(doc_id) / "audio.mp3")
        audio_result = generate_audio(summary, audio_path, use_edge=True)

        # Generate video from summary (with audio if available)
        save_status(doc_id, {"state": "video", "step": "generating_video"})
        video_path = str(doc_dir(doc_id) / "video.mp4")
        video_result = generate_video(summary, video_path, audio_path=audio_result)

        # Update final status
        save_status(
            doc_id,
            {
                "state": "ready",
                "doc_id": doc_id,
                "num_pages": num_pages,
                "has_summary": True,
                "has_audio": audio_result is not None,
                "has_video": video_result is not None,
            },
        )

    except Exception as e:
        save_status(doc_id, {"state": "error", "error": str(e)})


@router.post("/upload")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload PDF and trigger full processing pipeline.

    Returns immediately with a doc_id; processing continues in the background.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF supported")

    doc_id = str(uuid.uuid4())
    save_status(doc_id, {"state": "processing", "step": "upload"})

    pdf_bytes = await file.read()

    # Persist original PDF (fast)
    pdf_path = doc_dir(doc_id) / "source.pdf"
    pdf_path.write_bytes(pdf_bytes)

    # Kick off background processing
    background_tasks.add_task(_process_document, doc_id, file.filename)

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "status": "processing",
    }


@router.get("/status/{doc_id}")
def status(doc_id: str):
    """Get processing status for a document."""
    st = load_status(doc_id)
    if not st:
        raise HTTPException(status_code=404, detail="Unknown doc_id")
    return st


@router.get("/summary/{doc_id}")
def summary(doc_id: str, mode: str = Query("detailed", pattern="^(brief|detailed)$")):
    """Get summary for a document."""
    meta = load_doc_meta(doc_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Unknown doc_id")
    
    # Check if summary already exists
    summary_path = doc_dir(doc_id) / "summary.txt"
    if summary_path.exists():
        summary_text = summary_path.read_text(encoding="utf-8")
    else:
        # Generate summary on demand
        pages = load_doc_pages(doc_id)
        if not pages:
            raise HTTPException(status_code=404, detail="No pages stored for doc")
        
        save_status(doc_id, {"state": "summarizing", "mode": mode})
        
        text = "\n\n".join(pages)
        summary_text = summarize_text(text, mode=mode)  # type: ignore[arg-type]
        
        # Save for future use
        summary_path.write_text(summary_text, encoding="utf-8")
        
        save_status(doc_id, {"state": "ready", "summary_mode": mode})
    
    return {
        "doc_id": doc_id,
        "mode": mode,
        "summary": summary_text,
        "num_pages": meta.get("num_pages"),
    }


@router.get("/audio/{doc_id}")
def get_audio(doc_id: str):
    """Download or stream audio file for a document."""
    meta = load_doc_meta(doc_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Unknown doc_id")
    
    audio_mp3 = doc_dir(doc_id) / "audio.mp3"
    audio_wav = doc_dir(doc_id) / "audio.wav"

    if audio_mp3.exists():
        return FileResponse(
            path=str(audio_mp3),
            media_type="audio/mpeg",
            filename=f"{doc_id}_audio.mp3",
        )

    if audio_wav.exists():
        return FileResponse(
            path=str(audio_wav),
            media_type="audio/wav",
            filename=f"{doc_id}_audio.wav",
        )

    raise HTTPException(status_code=404, detail="Audio not generated yet. Processing may still be in progress.")


@router.get("/video/{doc_id}")
def get_video(doc_id: str):
    """Download or stream video file for a document."""
    meta = load_doc_meta(doc_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Unknown doc_id")
    
    video_path = doc_dir(doc_id) / "video.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not generated yet. Processing may still be in progress.")
    
    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=f"{doc_id}_video.mp4"
    )


@router.post("/chat")
def chat(req: ChatRequest):
    """Chat with a document using RAG."""
    meta = load_doc_meta(req.doc_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Unknown doc_id. Upload a PDF first.")
    
    # Load vector store
    db = load_faiss(req.doc_id)
    if not db:
        raise HTTPException(status_code=404, detail="Vector store not found. Document may not be fully processed.")
    
    # Retrieve evidence and generate answer
    evidence = retrieve(db, req.question)
    answer, sources = answer_with_citations(req.question, evidence)
    
    return {"answer": answer, "sources": sources}

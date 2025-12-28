from fastapi import FastAPI
from app.routes import pdf, chat, media

app = FastAPI(title="Offline-PDF-RAG (Ollama)")

app.include_router(pdf.router, prefix="", tags=["pdf"])
app.include_router(chat.router, prefix="", tags=["chat"])
app.include_router(media.router, prefix="", tags=["media"])


@app.get("/health")
def health():
    return {"ok": True, "service": "offline-pdf-rag"}

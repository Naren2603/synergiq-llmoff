from pydantic import BaseModel
from fastapi import APIRouter, HTTPException

from app.state import DOCS
from app.core.rag import retrieve, answer_with_citations

router = APIRouter()


class ChatRequest(BaseModel):
    doc_id: str
    question: str


@router.post("/chat")
def chat(req: ChatRequest):
    doc = DOCS.get(req.doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Unknown doc_id. Upload a PDF first.")

    evidence = retrieve(doc["db"], req.question)
    answer, sources = answer_with_citations(req.question, evidence)
    return {"answer": answer, "sources": sources}

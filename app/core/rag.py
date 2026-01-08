from __future__ import annotations

import os
import time
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

from app.core.storage import index_dir
from app.core.config import OLLAMA_BASE_URL, EMBED_MODEL


def citations_from_evidence(evidence: List[dict], *, max_citations: int | None = None) -> List[str]:
    """Build citation strings from retrieved evidence.

    Citation rule (paper + code consistency):
    - citations come from the top-k retrieved chunks actually sent to the LLM
    - citations are unique (page, chunk) in first-occurrence order
    - format: p{page}:c{chunk} when chunk is available, otherwise p{page}
    """
    seen: set[tuple[str, str]] = set()
    out: List[str] = []

    for ev in evidence:
        meta = ev.get("metadata") or {}
        page = meta.get("page")
        chunk = meta.get("chunk")

        page_s = "?" if page is None else str(page)
        chunk_s = "" if chunk is None else str(chunk)

        key = (page_s, chunk_s)
        if key in seen:
            continue
        seen.add(key)

        if chunk_s:
            out.append(f"p{page_s}:c{chunk_s}")
        else:
            out.append(f"p{page_s}")

        if max_citations is not None and len(out) >= max_citations:
            break

    return out


def _embeddings() -> OllamaEmbeddings:
    # Keep chat and embedding models configurable separately.
    # If EMBED_MODEL isn't set, config falls back to OLLAMA_MODEL.
    return OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=EMBED_MODEL)


def persist_faiss(doc_id: str, vs: FAISS) -> str:
    path = str(index_dir(doc_id))
    vs.save_local(path)
    return path


def load_faiss(doc_id: str) -> Optional[FAISS]:
    path = str(index_dir(doc_id))
    if not os.path.exists(path):
        return None
    try:
        return FAISS.load_local(path, _embeddings(), allow_dangerous_deserialization=True)
    except Exception:
        return None


def build_or_load_index(doc_id: str, texts: List[str], metadatas: Optional[List[dict]] = None) -> FAISS:
    existing = load_faiss(doc_id)
    if existing is not None:
        return existing

    last_err: Exception | None = None
    # Ollama can occasionally drop connections under load; retry with backoff.
    for attempt in range(1, 4):
        try:
            vs = FAISS.from_texts(texts=texts, embedding=_embeddings(), metadatas=metadatas)
            persist_faiss(doc_id, vs)
            return vs
        except Exception as e:
            last_err = e
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"Failed to build FAISS index after retries: {last_err}")


def build_vectorstore(pages: List[dict], doc_id: str) -> tuple[FAISS, List[dict]]:
    """Build vectorstore from pages. Each page is a dict with 'page' and 'text' keys."""
    texts = []
    metadatas = []
    for p in pages:
        page_num = p.get("page", 0)
        text = p.get("text", "").strip()
        if text:
            texts.append(text)
            metadatas.append({"page": page_num, "doc_id": doc_id})
    
    if not texts:
        # Empty vectorstore - create with dummy document
        texts = ["No content available"]
        metadatas = [{"page": 0, "doc_id": doc_id}]
    
    db = build_or_load_index(doc_id, texts, metadatas)
    return db, metadatas


def retrieve(db: FAISS, query: str, k: int = 5) -> List[dict]:
    """Retrieve top-k relevant documents from vectorstore."""
    try:
        docs = db.similarity_search(query, k=k)
        evidence = []
        for doc in docs:
            evidence.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
            })
        return evidence
    except Exception:
        return []


def answer_with_citations(question: str, evidence: List[dict]) -> tuple[str, List[str]]:
    """Answer question using evidence with citations."""
    from app.core.ollama_client import ollama_chat
    
    if not evidence:
        return "No relevant information found in the document.", []
    
    # Build context from evidence
    context_parts = []
    sources = citations_from_evidence(evidence)
    for i, ev in enumerate(evidence):
        meta = ev.get("metadata") or {}
        page = meta.get("page", "?")
        chunk = meta.get("chunk")
        label = f"Page {page}" if chunk is None else f"Page {page}, Chunk {chunk}"
        text = ev.get("text", "")
        context_parts.append(f"[Source {i+1}, {label}]:\n{text}")
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""Based on the following context from a document, answer the question.
If the answer is not in the context, say so.

Context:
{context}

Question: {question}

Answer:"""
    
    try:
        answer = ollama_chat(prompt, system="You are a helpful assistant that answers questions based on provided context. Always ground your answers in the context provided.")
        return answer, sources
    except Exception as e:
        return f"Error generating answer: {str(e)}", sources

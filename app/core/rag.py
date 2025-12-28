from __future__ import annotations

import os
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

from app.core.storage import index_dir


def _embeddings() -> OllamaEmbeddings:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.1")
    return OllamaEmbeddings(base_url=base_url, model=model)


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
    vs = FAISS.from_texts(texts=texts, embedding=_embeddings(), metadatas=metadatas)
    persist_faiss(doc_id, vs)
    return vs

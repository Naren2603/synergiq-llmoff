from typing import List, Dict, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.core.config import EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
from app.core.ollama_client import ollama_chat


def build_vectorstore(pages: List[Dict], doc_id: str) -> Tuple[FAISS, List[Document]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    docs: List[Document] = []
    chunk_id = 0
    for p in pages:
        page_no = p["page"]
        text = (p.get("text") or "").strip()
        if not text:
            continue

        chunks = splitter.split_text(text)
        for ch in chunks:
            chunk_id += 1
            docs.append(
                Document(
                    page_content=ch,
                    metadata={"doc_id": doc_id, "page": page_no, "chunk_id": chunk_id},
                )
            )

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = FAISS.from_documents(docs, embeddings)
    return db, docs


def retrieve(db: FAISS, question: str, k: int = TOP_K) -> List[Document]:
    return db.similarity_search(question, k=k)


def answer_with_citations(question: str, evidence: List[Document]):
    sources = []
    labeled = []
    for d in evidence:
        p = d.metadata.get("page", "?")
        c = d.metadata.get("chunk_id", "?")
        label = f"[p{p}:c{c}]"
        labeled.append(f"{label} {d.page_content}")
        sources.append({"page": p, "chunk_id": c, "text": d.page_content})

    context_block = "\n\n".join(labeled)

    system = (
        "You are a grounded QA assistant for educational PDFs.\n"
        "RULES:\n"
        "1) Use ONLY the provided EVIDENCE.\n"
        "2) If the answer is not in the evidence, output exactly: Not found in the document.\n"
        "3) Cite major claims with the exact tags like [p12:c3].\n"
        "4) Be concise and factual.\n"
    )

    prompt = f"""QUESTION:\n{question}\n\nEVIDENCE:\n{context_block}\n\nANSWER (with citations):"""

    ans = ollama_chat(prompt, system=system, temperature=0.2)
    if not ans:
        ans = "Not found in the document."
    return ans, sources

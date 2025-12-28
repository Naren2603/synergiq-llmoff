import hashlib
from typing import List, Dict
from pypdf import PdfReader


def file_id_from_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:24]


def extract_pages(pdf_path: str) -> List[Dict]:
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({"page": i, "text": text})
    return pages


def clean_text(text: str) -> str:
    t = (text or "").replace("\x00", " ").strip()
    t = " ".join(t.split())
    return t


def is_low_information(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 200:
        return True
    letters = sum(c.isalpha() for c in t)
    if letters < 50:
        return True
    return False

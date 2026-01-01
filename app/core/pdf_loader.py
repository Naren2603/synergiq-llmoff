from __future__ import annotations

from typing import List, Tuple

from pypdf import PdfReader

from app.core.ocr import safe_ocr_single_page


def extract_pdf_pages(pdf_path: str, ocr_empty_pages: bool = True) -> Tuple[List[str], int]:
    """Return (pages_text, num_pages). OCR pages that have no extractable text."""
    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    pages: List[str] = []

    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        if ocr_empty_pages and len(text.strip()) == 0:
            ocr_text = safe_ocr_single_page(pdf_path, i + 1)
            if ocr_text:
                text = ocr_text

        pages.append(text)

    return pages, num_pages


def load_pdf_text(pdf_path: str) -> str:
    pages, _ = extract_pdf_pages(pdf_path, ocr_empty_pages=True)
    return "\n\n".join(pages)

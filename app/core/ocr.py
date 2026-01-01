from __future__ import annotations

import os
from typing import List, Optional

from pdf2image import convert_from_path
import pytesseract


def ocr_page_images(pdf_path: str, first_page: int, last_page: int, dpi: int = 200) -> List[str]:
    """OCR a range of pages (1-indexed) from a PDF."""
    images = convert_from_path(
        pdf_path,
        dpi=dpi,
        first_page=first_page,
        last_page=last_page,
    )
    out: List[str] = []
    for img in images:
        out.append(pytesseract.image_to_string(img))
    return out


def ocr_single_page(pdf_path: str, page_number: int, dpi: int = 200) -> str:
    text = "".join(ocr_page_images(pdf_path, page_number, page_number, dpi=dpi))
    return text


def has_tesseract() -> bool:
    # If TESSERACT_CMD provided, pytesseract uses it.
    cmd = os.getenv("TESSERACT_CMD")
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    try:
        _ = pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def safe_ocr_single_page(pdf_path: str, page_number: int, dpi: int = 200) -> Optional[str]:
    if not has_tesseract():
        return None
    try:
        return ocr_single_page(pdf_path, page_number, dpi=dpi)
    except Exception:
        return None

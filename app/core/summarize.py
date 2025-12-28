from __future__ import annotations

import os
from typing import List, Literal

from langchain_text_splitters import RecursiveCharacterTextSplitter


SummaryMode = Literal["brief", "detailed"]


def _splitter_for_mode(mode: SummaryMode) -> RecursiveCharacterTextSplitter:
    # Smaller chunks for long docs; can be tuned via env.
    chunk_size = int(os.getenv("CHUNK_SIZE", "1200"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
    if mode == "brief":
        chunk_size = max(600, chunk_size // 2)
        chunk_overlap = max(100, chunk_overlap // 2)
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def chunk_text(text: str, mode: SummaryMode = "detailed") -> List[str]:
    return _splitter_for_mode(mode).split_text(text)


def map_target_chars(mode: SummaryMode) -> int:
    base = int(os.getenv("SUMMARY_MAP_CHARS", "6000"))
    if mode == "brief":
        return max(2500, base // 2)
    return base

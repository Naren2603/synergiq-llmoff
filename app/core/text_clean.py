from __future__ import annotations

import re


_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```")
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")


def strip_markdown(text: str) -> str:
    """Best-effort markdown cleanup for LLM outputs.

    Keeps the readable text but removes common markdown tokens.
    """
    t = text or ""

    # Remove fenced code blocks entirely
    t = _CODE_FENCE_RE.sub(" ", t)

    # Images -> drop
    t = _IMAGE_RE.sub(" ", t)

    # Links: [text](url) -> text
    t = _LINK_RE.sub(r"\1", t)

    # Inline code -> content
    t = _INLINE_CODE_RE.sub(r"\1", t)

    # Headings / blockquotes markers at line start
    t = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", t)
    t = re.sub(r"(?m)^\s{0,3}>\s?", "", t)

    # Bold/italic markers
    t = t.replace("**", "")
    t = t.replace("__", "")
    t = t.replace("*", "")
    t = t.replace("_", "")

    # Horizontal rules
    t = re.sub(r"(?m)^\s*([-*_]\s*){3,}$", " ", t)

    return t


def normalize_whitespace(text: str) -> str:
    t = (text or "").replace("\x00", " ")
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    # Trim trailing spaces per line
    t = re.sub(r"(?m)[ \t]+$", "", t)
    # Collapse 3+ newlines
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def to_display_text(text: str) -> str:
    """Clean text for on-screen display (keeps paragraphs)."""
    t = strip_markdown(text)

    # Remove common bullet prefixes but keep line breaks
    t = re.sub(r"(?m)^\s*[-•]\s+", "", t)
    t = re.sub(r"(?m)^\s*\d+\)\s+", "", t)
    t = re.sub(r"(?m)^\s*\d+\.\s+", "", t)

    return normalize_whitespace(t)


def to_speech_text(text: str) -> str:
    """Clean text for TTS (removes markdown and makes flow more natural)."""
    t = to_display_text(text)

    # Make bullets flow as sentences
    t = re.sub(r"(?m)^\s*([A-Za-z][^:\n]{0,60}):\s*$", r"\1.", t)

    # Convert remaining newlines to sentence breaks
    t = re.sub(r"\n\n+", ". ", t)
    t = re.sub(r"\n+", ". ", t)

    # Collapse repeated punctuation/spaces
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\.\s*\.+", ". ", t)

    return t.strip()

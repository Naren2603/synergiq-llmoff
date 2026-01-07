from __future__ import annotations

import os

from app.core.ollama_client import ollama_chat
from app.core.text_clean import to_speech_text


def rewrite_for_speech(text: str, *, style: str = "casual") -> str:
    """Rewrite text into a spoken-script style for TTS.

    This improves perceived "human-ness" by making the narration flow naturally.
    Runs fully offline using local Ollama.
    """
    cleaned = to_speech_text(text)
    if not cleaned:
        return ""

    style = (style or "").strip().lower()
    if style not in {"casual", "neutral"}:
        style = "casual"

    max_chars = int(os.getenv("SPOKEN_MAX_CHARS", "7000"))
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rsplit(" ", 1)[0]

    if style == "neutral":
        system = (
            "Rewrite the text as a natural spoken narration in plain English. "
            "No markdown, no headings, no bullets. Use full sentences and smooth flow."
        )
    else:
        system = (
            "Rewrite the text as a natural spoken narration in casual English. "
            "Use contractions (don't, it's, we're), simple words, and a friendly tone. "
            "Avoid slang that is offensive or hard to understand. "
            "No markdown, no headings, no bullets."
        )

    prompt = (
        "Rewrite the following content into a single spoken script suitable for voice-over. "
        "Keep it coherent and flowing. Do not add extra facts.\n\n"
        f"CONTENT:\n{cleaned}\n\nSPOKEN SCRIPT:"
    )

    try:
        out = ollama_chat(prompt, system=system, temperature=0.4)
        return to_speech_text(out)
    except Exception:
        # If Ollama is unavailable, fall back to cleaned text.
        return cleaned

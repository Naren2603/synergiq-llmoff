from app.core.ollama_client import ollama_chat
from app.core.config import SUMMARY_MAP_CHARS


def summarize_map_reduce(full_text: str) -> str:
    text = (full_text or "").strip()
    if not text:
        return ""

    system = "You are a precise academic summarizer. Do not add facts not present in the text."

    parts = []
    for i in range(0, len(text), SUMMARY_MAP_CHARS):
        chunk = text[i : i + SUMMARY_MAP_CHARS]
        prompt = f"""Summarize this chunk into 8-12 bullet points.\nKeep it factual and concise.\n\nCHUNK:\n{chunk}\n"""
        parts.append(ollama_chat(prompt, system=system, temperature=0.2))

    combined = "\n".join(parts)

    reduce_prompt = f"""You will receive chunk summaries of a long PDF.\n\nTASK:\n1) Produce a final structured summary with:\n   - Title (guess from content)\n   - 10 key bullet points\n   - 5 important definitions/formulas (if any)\n   - 5 likely exam questions (with short answers)\n\nCHUNK SUMMARIES:\n{combined}\n"""

    return ollama_chat(reduce_prompt, system=system, temperature=0.2)

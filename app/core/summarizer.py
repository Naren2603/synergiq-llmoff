"""LLM-based summarization using Ollama with map-reduce for long documents."""
from __future__ import annotations

from typing import List
from app.core.ollama_client import ollama_chat
from app.core.summarize import chunk_text, SummaryMode


def summarize_chunk(chunk: str, mode: SummaryMode = "detailed") -> str:
    """Summarize a single chunk of text using LLM."""
    if mode == "brief":
        system = (
            "You create brief, concise summaries in plain text. "
            "Do NOT use markdown (no headings, bullets, **bold**, ###, etc.)."
        )
        prompt = (
            "Provide a brief summary (3-5 sentences) of the following text. "
            "Write in full sentences as a single paragraph.\n\n"
            f"{chunk}"
        )
    else:
        system = (
            "You create detailed summaries in plain text. "
            "Do NOT use markdown (no headings, bullets, **bold**, ###, etc.)."
        )
        prompt = (
            "Provide a detailed summary of the following text, capturing key points and important details. "
            "Write in plain text with flowing paragraphs. Avoid lists/bullets unless strictly necessary.\n\n"
            f"{chunk}"
        )
    
    try:
        return ollama_chat(prompt, system=system, temperature=0.3)
    except Exception as e:
        print(f"Error summarizing chunk: {e}")
        # Fallback to simple extraction
        lines = chunk.split('\n')
        return ' '.join(lines[:5])


def summarize_text(text: str, mode: SummaryMode = "detailed") -> str:
    """
    Summarize text using map-reduce approach for long documents.
    
    Args:
        text: Full text to summarize
        mode: Summary mode - "brief" or "detailed"
    
    Returns:
        Summary text
    """
    if not text or not text.strip():
        return "No content available to summarize."
    
    # Split into chunks
    chunks = chunk_text(text, mode=mode)
    
    if not chunks:
        return "No content available to summarize."
    
    # If text is short enough, summarize directly
    if len(chunks) == 1:
        return summarize_chunk(chunks[0], mode)
    
    # Map phase: summarize each chunk
    print(f"Summarizing {len(chunks)} chunks...")
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i+1}/{len(chunks)}...")
        summary = summarize_chunk(chunk, mode)
        if summary:
            chunk_summaries.append(summary)
    
    if not chunk_summaries:
        return "Failed to generate summary."
    
    # If we have many chunk summaries, do a reduce phase.
    # Use hierarchical reduction in small batches to avoid huge prompts/timeouts.
    if len(chunk_summaries) > 5:
        system = (
            "You synthesize multiple summaries into a coherent whole in plain text. "
            "Do NOT use markdown (no headings, bullets, **bold**, ###, etc.)."
        )

        def _reduce_batch(summaries: List[str]) -> str:
            combined = "\n\n".join(summaries)
            if mode == "brief":
                reduce_prompt = (
                    "The following are summaries of different sections of a document. "
                    "Combine them into one brief, coherent summary (3-5 sentences):\n\n"
                    f"{combined}"
                )
            else:
                reduce_prompt = (
                    "The following are summaries of different sections of a document. "
                    "Combine them into one detailed, coherent summary that captures all key points:\n\n"
                    f"{combined}"
                )
            return ollama_chat(reduce_prompt, system=system, temperature=0.3)

        summaries = list(chunk_summaries)
        try:
            while len(summaries) > 5:
                print(f"Reducing intermediate summaries... ({len(summaries)} parts)")
                next_level: List[str] = []
                for i in range(0, len(summaries), 5):
                    batch = summaries[i : i + 5]
                    try:
                        next_level.append(_reduce_batch(batch))
                    except Exception as e:
                        print(f"Error reducing batch: {e}")
                        next_level.append("\n\n".join(batch))
                summaries = next_level

            print(f"Final reduction... ({len(summaries)} parts)")
            return _reduce_batch(summaries)
        except Exception as e:
            print(f"Error in reduce phase: {e}")
            return "\n\n".join(chunk_summaries[:10])
    
    # If few chunks, just combine them
    return "\n\n".join(chunk_summaries)

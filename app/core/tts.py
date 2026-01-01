"""Text-to-speech module using edge-tts with gtts fallback."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Optional

from app.core.text_clean import to_speech_text


def generate_audio_gtts(text: str, output_path: str, lang: str = "en") -> bool:
    """Generate audio using gTTS (Google Text-to-Speech) - fallback option."""
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(output_path)
        return True
    except Exception as e:
        print(f"gTTS failed: {e}")
        return False


async def generate_audio_edge_async(text: str, output_path: str, voice: str = "en-US-AriaNeural") -> bool:
    """Generate audio using edge-tts (Microsoft Edge TTS) - preferred option."""
    try:
        import edge_tts
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)
        return True
    except Exception as e:
        print(f"edge-tts failed: {e}")
        return False


def generate_audio_edge(text: str, output_path: str, voice: str = "en-US-AriaNeural") -> bool:
    """Synchronous wrapper for edge-tts."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(generate_audio_edge_async(text, output_path, voice))
        loop.close()
        return result
    except Exception as e:
        print(f"edge-tts sync wrapper failed: {e}")
        return False


def generate_audio(text: str, output_path: str, use_edge: bool = True) -> Optional[str]:
    """
    Generate audio from text. Returns path to generated audio file or None on failure.
    
    Args:
        text: Text to convert to speech
        output_path: Path where audio file should be saved
        use_edge: Whether to try edge-tts first (True) or use gtts directly (False)
    
    Returns:
        Path to generated audio file or None if generation failed
    """
    if not text or not text.strip():
        print("Empty text provided for TTS")
        return None

    # Clean up markdown-like artifacts and improve flow for speech
    text = to_speech_text(text)
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    success = False
    
    if use_edge:
        # Try edge-tts first (better quality, free, offline-capable)
        print("Attempting TTS with edge-tts...")
        success = generate_audio_edge(text, output_path)
    
    if not success:
        # Fallback to gTTS
        print("Falling back to gTTS...")
        success = generate_audio_gtts(text, output_path)
    
    if success and os.path.exists(output_path):
        print(f"Audio generated successfully: {output_path}")
        return output_path
    
    print("Failed to generate audio with all available methods")
    return None

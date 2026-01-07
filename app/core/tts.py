"""Text-to-speech module using edge-tts with gtts fallback."""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from app.core.text_clean import to_speech_text
from app.core.speech_style import rewrite_for_speech


def _ffmpeg_exists() -> bool:
    return shutil.which("ffmpeg") is not None


def _wav_to_mp3(wav_path: str, mp3_path: str) -> bool:
    if not _ffmpeg_exists():
        return False
    try:
        # -y overwrite; -vn no video; reasonable quality
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-vn", "-codec:a", "libmp3lame", "-q:a", "4", mp3_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return os.path.exists(mp3_path)
    except Exception:
        return False


def generate_audio_piper(text: str, output_path: str) -> Optional[str]:
    """Generate audio using Piper (offline neural TTS).

    Requires:
    - Piper installed (either `piper` on PATH or PIPER_CMD set)
    - A Piper voice model (.onnx) path set via PIPER_MODEL
    """
    piper_cmd = os.getenv("PIPER_CMD", "piper")
    model_path = os.getenv("PIPER_MODEL")
    if not model_path or not os.path.exists(model_path):
        print("Piper requested but PIPER_MODEL is missing or not found.")
        return None

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    wants_mp3 = output_path.lower().endswith(".mp3")
    wav_path = output_path
    if wants_mp3:
        wav_path = str(Path(output_path).with_suffix(".wav"))

    try:
        proc = subprocess.run(
            [piper_cmd, "--model", model_path, "--output_file", wav_path],
            input=text.encode("utf-8"),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"Piper failed: {e}")
        return None

    if not os.path.exists(wav_path):
        return None

    if wants_mp3:
        if _wav_to_mp3(wav_path, output_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass
            return output_path
        # No ffmpeg: keep wav
        return wav_path

    return wav_path


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
    speech = to_speech_text(text)

    # Optional: rewrite into a more natural "spoken script" using local Ollama
    spoken_style = os.getenv("SPOKEN_STYLE", "neutral").strip().lower()
    if spoken_style in {"casual", "neutral"}:
        speech = rewrite_for_speech(speech, style=spoken_style)
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Preferred engine selection
    engine = os.getenv("TTS_ENGINE", "edge").strip().lower()

    if engine == "piper":
        print("Attempting TTS with Piper (offline)...")
        out = generate_audio_piper(speech, output_path)
        if out is not None and os.path.exists(out):
            print(f"Audio generated successfully: {out}")
            return out
        print("Piper failed or not configured; falling back...")

    # Existing online engines
    success = False
    if use_edge and engine in {"edge", "piper", "auto"}:
        print("Attempting TTS with edge-tts...")
        success = generate_audio_edge(speech, output_path)

    if not success:
        print("Falling back to gTTS...")
        success = generate_audio_gtts(speech, output_path)

    if success and os.path.exists(output_path):
        print(f"Audio generated successfully: {output_path}")
        return output_path

    print("Failed to generate audio with all available methods")
    return None

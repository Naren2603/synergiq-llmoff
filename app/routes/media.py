import os
import tempfile
import asyncio

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from gtts import gTTS

import edge_tts

from app.state import DOCS
from app.core.summarize import summarize_map_reduce

router = APIRouter()


@router.get("/summary/{doc_id}")
def get_summary(doc_id: str):
    doc = DOCS.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Unknown doc_id. Upload a PDF first.")
    if not doc.get("summary"):
        doc["summary"] = summarize_map_reduce(doc["text"])
    return {"doc_id": doc_id, "summary": doc["summary"]}


async def _edge_tts_to_file(text: str, out_path: str, voice: str = "en-IN-PrabhatNeural"):
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(out_path)


@router.get("/audio/{doc_id}")
def get_audio(doc_id: str):
    doc = DOCS.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Unknown doc_id. Upload a PDF first.")

    if not doc.get("summary"):
        doc["summary"] = summarize_map_reduce(doc["text"])

    if not doc.get("audio_path") or not os.path.exists(doc["audio_path"]):
        text = doc["summary"][:9000]
        out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        out.close()
        try:
            asyncio.run(_edge_tts_to_file(text, out.name))
        except Exception:
            tts = gTTS(text=text, lang="en")
            tts.save(out.name)
        doc["audio_path"] = out.name

    return FileResponse(doc["audio_path"], media_type="audio/mpeg", filename=f"{doc_id}.mp3")


@router.get("/video/{doc_id}")
def get_video(doc_id: str):
    doc = DOCS.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Unknown doc_id. Upload a PDF first.")

    if not doc.get("summary"):
        doc["summary"] = summarize_map_reduce(doc["text"])

    if not doc.get("video_path") or not os.path.exists(doc["video_path"]):
        text = doc["summary"]
        out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        out.close()

        width, height = 1280, 720
        fps = 30
        duration = 12
        frames = fps * duration

        writer = cv2.VideoWriter(out.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        try:
            font = ImageFont.truetype("arial.ttf", 36)
        except Exception:
            font = ImageFont.load_default()

        words = text.split()
        lines = [" ".join(words[i:i+10]) for i in range(0, min(len(words), 200), 10)]
        if not lines:
            lines = ["(No summary text)"]

        for f in range(frames):
            img = Image.new("RGB", (width, height), color=(10, 10, 18))
            draw = ImageDraw.Draw(img)

            offset = int((f / frames) * max(1, len(lines) * 50))
            y = 80 - offset

            draw.text((60, 20), "PDF Summary Video", fill=(255, 255, 255), font=font)
            for line in lines:
                draw.text((60, y), line, fill=(220, 220, 220), font=font)
                y += 50

            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            writer.write(frame)

        writer.release()
        doc["video_path"] = out.name

    return FileResponse(doc["video_path"], media_type="video/mp4", filename=f"{doc_id}.mp4")

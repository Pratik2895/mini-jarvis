"""app/main.py — FastAPI backend.

Endpoints:
  GET  /health          liveness check
  POST /query           text → JARVIS response
  POST /voice           audio file → STT → JARVIS → text response
  GET  /docs            auto Swagger UI

Run locally: uvicorn app.main:app --reload
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.brain import JarvisBrain

app = FastAPI(title="Mini JARVIS", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One brain instance per process (stateful conversation history)
_brain = JarvisBrain()


# ── Models ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    text: str
    reset: bool = False  # set True to clear conversation history first

class QueryResponse(BaseModel):
    response: str


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "JARVIS online", "model": "gemini-2.0-flash"}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Text → JARVIS response."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")
    if req.reset:
        _brain.reset()
    response = _brain.chat(req.text)
    return QueryResponse(response=response)


@app.post("/query/reset")
async def reset_conversation():
    """Clear conversation history."""
    _brain.reset()
    return {"status": "conversation reset"}


@app.post("/voice", response_model=QueryResponse)
async def voice(file: Annotated[UploadFile, File(description="WAV/MP3 audio clip")]):
    """Audio → STT → JARVIS → text response."""
    # Lazy import so non-voice deployments don't need faster-whisper
    try:
        from client.speech_to_text import transcribe_file
    except ImportError:
        raise HTTPException(status_code=501, detail="faster-whisper not installed")

    # Save upload to temp file
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    transcript = transcribe_file(tmp_path)
    if not transcript.strip():
        raise HTTPException(status_code=422, detail="Could not transcribe audio")

    response = _brain.chat(transcript)
    return QueryResponse(response=response)

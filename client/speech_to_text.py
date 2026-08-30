"""client/speech_to_text.py — Local STT using faster-whisper (no API key needed).

faster-whisper runs on CPU; first run downloads the model (~150 MB for "base").
Model sizes: tiny | base | small | medium | large-v3
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Lazy import to allow the module to load even before faster-whisper is installed
_model = None


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        size = os.getenv("WHISPER_MODEL_SIZE", "base")
        print(f"[STT] Loading Whisper model '{size}' on CPU...")
        _model = WhisperModel(size, device="cpu", compute_type="int8")
        print("[STT] Model loaded.")
    return _model


def transcribe_file(audio_path: str) -> str:
    """Transcribe an audio file and return the text."""
    model = _get_model()
    segments, info = model.transcribe(audio_path, beam_size=5)
    text = " ".join(seg.text.strip() for seg in segments)
    print(f"[STT] Transcribed ({info.language}): {text!r}")
    return text


def transcribe_bytes(audio_bytes: bytes, suffix: str = ".wav") -> str:
    """Transcribe raw audio bytes (writes to a temp file)."""
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        return transcribe_file(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

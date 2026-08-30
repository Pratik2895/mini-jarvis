"""client/jarvis_client.py — Local voice loop.

Pipeline:
  Microphone → sounddevice → WAV buffer
  → faster-whisper (STT) → transcript
  → FastAPI /query endpoint → JARVIS response text
  → pyttsx3 (TTS) → Speaker

Usage:
  # Make sure FastAPI is running in another terminal:
  #   uvicorn app.main:app --reload
  python client/jarvis_client.py

Press Ctrl+C to stop.
"""
from __future__ import annotations

import io
import os
import sys
import time
import wave
from pathlib import Path

# Ensure project root is on sys.path so `client.*` imports resolve
# when running: python client/jarvis_client.py
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import requests
import sounddevice as sd

from client.speech_to_text import transcribe_bytes
from client.text_to_speech import speak

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = os.getenv("JARVIS_API_BASE", "http://localhost:8000")
SAMPLE_RATE = 16_000      # Hz — Whisper works best at 16 kHz
RECORD_SECONDS = 5        # seconds to record per turn
CHANNELS = 1
SILENCE_THRESHOLD = 0.01  # RMS below this = silence (skip sending)


def record_audio(seconds: int = RECORD_SECONDS) -> np.ndarray:
    """Record from mic and return float32 numpy array."""
    print(f"\n🎤  Listening for {seconds}s... (speak now)")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
    )
    sd.wait()
    return audio.flatten()


def audio_to_wav_bytes(audio: np.ndarray) -> bytes:
    """Convert float32 numpy array to WAV bytes."""
    pcm = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def is_silent(audio: np.ndarray) -> bool:
    return float(np.sqrt(np.mean(audio ** 2))) < SILENCE_THRESHOLD


def query_api(text: str) -> str:
    """Send transcript to FastAPI backend, return JARVIS response."""
    try:
        resp = requests.post(
            f"{API_BASE}/query",
            json={"text": text},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["response"]
    except requests.exceptions.ConnectionError:
        return "Cannot connect to JARVIS backend. Is the server running?"
    except Exception as e:
        return f"Error: {e}"


def run_voice_loop():
    print("=" * 55)
    print("  JARVIS — Voice Mode  (Ctrl+C to exit)")
    print("=" * 55)
    speak("JARVIS online. How can I help you?")

    while True:
        try:
            audio = record_audio()
        except KeyboardInterrupt:
            break

        if is_silent(audio):
            print("  (silence detected, skipping)")
            continue

        wav_bytes = audio_to_wav_bytes(audio)
        print("  Transcribing...")
        transcript = transcribe_bytes(wav_bytes, suffix=".wav")

        if not transcript.strip():
            print("  (no speech detected)")
            continue

        print(f"You: {transcript}")
        response = query_api(transcript)
        print(f"JARVIS: {response}")
        speak(response)

    print("\nGoodbye!")
    speak("Goodbye!")


if __name__ == "__main__":
    run_voice_loop()

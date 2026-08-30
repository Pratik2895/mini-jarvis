"""client/text_to_speech.py — Offline TTS using pyttsx3."""
from __future__ import annotations

import os
import pyttsx3

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = pyttsx3.init()
        rate = int(os.getenv("TTS_RATE", "175"))
        volume = float(os.getenv("TTS_VOLUME", "1.0"))
        _engine.setProperty("rate", rate)
        _engine.setProperty("volume", volume)
        # Pick a clear voice if available
        voices = _engine.getProperty("voices")
        if voices:
            # Prefer first English voice
            eng = [v for v in voices if "en" in v.id.lower()]
            if eng:
                _engine.setProperty("voice", eng[0].id)
    return _engine


def speak(text: str) -> None:
    """Speak text aloud and block until done."""
    engine = _get_engine()
    engine.say(text)
    engine.runAndWait()


def speak_async(text: str) -> None:
    """Non-blocking speak — fires in the background."""
    import threading
    threading.Thread(target=speak, args=(text,), daemon=True).start()

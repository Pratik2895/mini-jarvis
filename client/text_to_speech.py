"""client/text_to_speech.py — Offline TTS using pyttsx3."""
from __future__ import annotations

import os
import pyttsx3

_engine = None


def _create_fresh_engine():
    try:
        import pythoncom
        pythoncom.CoInitialize()
    except Exception:
        pass
    engine = pyttsx3.init()
    rate = int(os.getenv("TTS_RATE", "175"))
    volume = float(os.getenv("TTS_VOLUME", "1.0"))
    engine.setProperty("rate", rate)
    engine.setProperty("volume", volume)
    voices = engine.getProperty("voices")
    if voices:
        eng = [v for v in voices if "en" in v.id.lower()]
        if eng:
            engine.setProperty("voice", eng[0].id)
    return engine


def speak(text: str) -> None:
    """Speak text aloud and block until done."""
    engine = _create_fresh_engine()
    engine.say(text)
    engine.runAndWait()
    engine.stop()


def speak_async(text: str) -> None:
    """Non-blocking speak — fires in the background."""
    import threading
    threading.Thread(target=speak, args=(text,), daemon=True).start()


def text_to_wav(text: str, output_path: str) -> str:
    """Synthesize text into a WAV file for browser/audio playback."""
    engine = _create_fresh_engine()
    engine.save_to_file(text, output_path)
    engine.runAndWait()
    engine.stop()
    return output_path



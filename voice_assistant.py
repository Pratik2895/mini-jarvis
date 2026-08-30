"""voice_assistant.py — Full Two-Way Voice Conversational Assistant (JARVIS).

Features:
  - Natural 2-way voice loop: JARVIS speaks -> listens to you -> responds -> continues loop
  - Dynamic Voice Activity Detection (VAD): auto-detects when you start & finish speaking
  - Tool execution (time, weather, reminders, calculation)
  - Retains conversational context across turns
  - Direct local pipeline (no external server required)

Usage:
    python voice_assistant.py
"""
from __future__ import annotations

import collections
import io
import math
import os
import sys
import time
import wave
from pathlib import Path

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import sounddevice as sd

from app.brain import JarvisBrain
from client.speech_to_text import transcribe_bytes
from client.text_to_speech import speak

# Audio Settings
SAMPLE_RATE = 16_000        # Whisper optimal rate
BLOCK_SIZE = 1024           # Audio chunk size (~64ms)
SILENCE_DURATION = 1.4      # Seconds of silence after speech to consider turn finished
ENERGY_THRESHOLD = 0.015    # RMS threshold for voice detection
MAX_RECORD_SECONDS = 30     # Max length for single utterance


def calculate_rms(chunk: np.ndarray) -> float:
    """Calculate Root Mean Square audio energy."""
    return float(np.sqrt(np.mean(chunk**2))) if len(chunk) > 0 else 0.0


def record_voice_with_vad() -> np.ndarray | None:
    """Listens continuously to mic and records audio from speech start to speech end."""
    print("\n🎤  [JARVIS is listening...] (Speak into your mic)")

    ring_buffer = collections.deque(maxlen=int(SAMPLE_RATE / BLOCK_SIZE * 0.5))  # 0.5s pre-buffer
    audio_frames = []
    has_spoken = False
    silence_start_time = None
    start_record_time = time.time()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=BLOCK_SIZE) as stream:
        while True:
            chunk, overflowed = stream.read(BLOCK_SIZE)
            chunk = chunk.flatten()
            rms = calculate_rms(chunk)

            # Check if speaking
            if rms > ENERGY_THRESHOLD:
                if not has_spoken:
                    print("  🎙️ Speech detected... recording")
                    has_spoken = True
                    # Include pre-buffer so beginning of word isn't clipped
                    audio_frames.extend(ring_buffer)
                audio_frames.append(chunk)
                silence_start_time = None
            else:
                if has_spoken:
                    audio_frames.append(chunk)
                    if silence_start_time is None:
                        silence_start_time = time.time()
                    elif (time.time() - silence_start_time) >= SILENCE_DURATION:
                        # User stopped speaking
                        break
                else:
                    ring_buffer.append(chunk)

            # Max duration safeguard
            if has_spoken and (time.time() - start_record_time) > MAX_RECORD_SECONDS:
                break

            # Timeout after 45s of complete silence
            if not has_spoken and (time.time() - start_record_time) > 45:
                print("  💤 (Listening timed out after silence)")
                return None

    if audio_frames:
        return np.concatenate(audio_frames)
    return None


def audio_to_wav_bytes(audio: np.ndarray) -> bytes:
    """Convert float32 numpy array to WAV bytes."""
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def run_conversational_voice_loop():
    print("=" * 65)
    print("  ⚡ J.A.R.V.I.S. TWO-WAY VOICE CONVERSATION MODE ⚡")
    print("  • Talk naturally into your microphone")
    print("  • JARVIS will listen, reply through speakers, and listen back")
    print("  • Say 'Goodbye' or press Ctrl+C to exit")
    print("=" * 65)

    brain = JarvisBrain()
    initial_greeting = "JARVIS online and ready, sir. How can I assist you today?"
    print(f"\nJARVIS: {initial_greeting}")
    speak(initial_greeting)

    while True:
        try:
            # 1. Listen with VAD
            audio_data = record_voice_with_vad()
            if audio_data is None:
                continue

            # 2. Transcribe voice
            print("  ⏳ Transcribing your speech...")
            wav_bytes = audio_to_wav_bytes(audio_data)
            user_text = transcribe_bytes(wav_bytes, suffix=".wav").strip()

            if not user_text:
                print("  ⚠️ (Could not catch that, listening again...)")
                continue

            print(f"\n👤 You: {user_text}")

            # Check exit phrases
            if user_text.lower() in ("goodbye", "exit", "quit", "goodbye jarvis", "bye jarvis", "bye"):
                farewell = "Goodbye, sir. Have a wonderful day!"
                print(f"JARVIS: {farewell}")
                speak(farewell)
                break

            # 3. Brain & Tool Processing (Gemini)
            print("  🧠 JARVIS is thinking & processing tools...")
            response = brain.chat(user_text)

            # 4. Speak response aloud
            print(f"🤖 JARVIS: {response}")
            speak(response)

            # Small pause before listening back
            time.sleep(0.3)

        except KeyboardInterrupt:
            print("\n\nJARVIS voice loop terminated by user.")
            speak("Powering down.")
            break
        except Exception as e:
            print(f"\n[Error in Voice Loop]: {e}")
            time.sleep(1)


if __name__ == "__main__":
    run_conversational_voice_loop()
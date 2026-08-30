"""dashboard.py — JARVIS All-in-One Command Center Dashboard.

One-Stop Solution:
  - 🎙️ Hands-Free Live Voice Conversation Mode (Speak -> Listen -> Respond Loop)
  - 💬 Live Real-Time Chatbot & Message Stream
  - 🛠️ Live Tool Execution Feed & Visual Status HUD
  - ⚙️ Direct Tool Diagnostics Playground
  - 📐 Databricks Architecture Specs

Run with:
    python dashboard.py
"""
from __future__ import annotations

import collections
import datetime
import io
import os
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import gradio as gr
import numpy as np
import sounddevice as sd

from app.brain import JarvisBrain
from app.config import settings
from app.tools import calculate, get_current_time, get_weather, set_reminder, TOOL_SPECS
from client.speech_to_text import transcribe_bytes, transcribe_file
from client.text_to_speech import speak, text_to_wav

# Global Brain Instance & Shared State
brain = JarvisBrain()
conversation_history: list[dict] = []
is_voice_mode_active = False
voice_status_text = "Idle (Standby)"
voice_thread: threading.Thread | None = None
state_lock = threading.Lock()

# Audio VAD Settings
SAMPLE_RATE = 16_000
BLOCK_SIZE = 1024
SILENCE_DURATION = 1.3
ENERGY_THRESHOLD = 0.015


def calculate_rms(chunk: np.ndarray) -> float:
    return float(np.sqrt(np.mean(chunk**2))) if len(chunk) > 0 else 0.0


def record_voice_with_vad() -> np.ndarray | None:
    global voice_status_text
    with state_lock:
        voice_status_text = "🎤 LISTENING... (Speak now)"

    ring_buffer = collections.deque(maxlen=int(SAMPLE_RATE / BLOCK_SIZE * 0.4))
    audio_frames = []
    has_spoken = False
    silence_start_time = None
    start_time = time.time()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=BLOCK_SIZE) as stream:
        while is_voice_mode_active:
            chunk, _ = stream.read(BLOCK_SIZE)
            chunk = chunk.flatten()
            rms = calculate_rms(chunk)

            if rms > ENERGY_THRESHOLD:
                if not has_spoken:
                    with state_lock:
                        voice_status_text = "🎙️ RECORDING SPEECH..."
                    has_spoken = True
                    audio_frames.extend(ring_buffer)
                audio_frames.append(chunk)
                silence_start_time = None
            else:
                if has_spoken:
                    audio_frames.append(chunk)
                    if silence_start_time is None:
                        silence_start_time = time.time()
                    elif (time.time() - silence_start_time) >= SILENCE_DURATION:
                        break
                else:
                    ring_buffer.append(chunk)

            if has_spoken and (time.time() - start_time) > 25:
                break

            if not has_spoken and (time.time() - start_time) > 60:
                return None

    if audio_frames:
        return np.concatenate(audio_frames)
    return None


def audio_to_wav_bytes(audio: np.ndarray) -> bytes:
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def continuous_voice_worker():
    global is_voice_mode_active, voice_status_text, conversation_history

    greeting = "JARVIS voice mode activated. I am listening, sir."
    with state_lock:
        conversation_history.append({"role": "assistant", "content": greeting})
        voice_status_text = "🔊 SPEAKING..."
    speak(greeting)

    while is_voice_mode_active:
        try:
            audio_data = record_voice_with_vad()
            if not is_voice_mode_active:
                break

            if audio_data is None:
                continue

            with state_lock:
                voice_status_text = "⏳ TRANSCRIBING SPEECH..."
            wav_bytes = audio_to_wav_bytes(audio_data)
            user_text = transcribe_bytes(wav_bytes, suffix=".wav").strip()

            if not user_text:
                continue

            with state_lock:
                conversation_history.append({"role": "user", "content": user_text})
                voice_status_text = "🧠 PROCESSING & EXECUTING TOOLS..."

            # Check stop phrase
            if user_text.lower() in ("stop voice mode", "stop listening", "goodbye", "exit", "quit"):
                farewell = "Stopping live voice mode. Standing by, sir."
                with state_lock:
                    conversation_history.append({"role": "assistant", "content": farewell})
                    voice_status_text = "🔊 SPEAKING..."
                speak(farewell)
                is_voice_mode_active = False
                break

            # Query Gemini Brain
            response = brain.chat(user_text)

            with state_lock:
                conversation_history.append({"role": "assistant", "content": response})
                voice_status_text = "🔊 SPEAKING RESPONSE..."

            speak(response)
            time.sleep(0.3)

        except Exception as e:
            print(f"[Error in Voice Thread]: {e}")
            time.sleep(1)

    with state_lock:
        voice_status_text = "Idle (Standby)"


def start_voice_mode():
    global is_voice_mode_active, voice_thread
    if not is_voice_mode_active:
        is_voice_mode_active = True
        voice_thread = threading.Thread(target=continuous_voice_worker, daemon=True)
        voice_thread.start()
    return "🟢 Voice Mode Running", format_tool_logs()


def stop_voice_mode():
    global is_voice_mode_active
    is_voice_mode_active = False
    return "⚪ Voice Mode Stopped", format_tool_logs()


def process_text_input(user_text, history, enable_voice):
    global conversation_history
    if not user_text or not user_text.strip():
        return conversation_history, None, "", format_tool_logs()

    prompt = user_text.strip()
    with state_lock:
        conversation_history.append({"role": "user", "content": prompt})

    response = brain.chat(prompt)

    with state_lock:
        conversation_history.append({"role": "assistant", "content": response})

    audio_path = None
    if enable_voice and response:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                audio_path = text_to_wav(response, tmp.name)
        except Exception as e:
            print(f"[TTS Error]: {e}")

    return conversation_history, audio_path, "", format_tool_logs()


def format_tool_logs():
    if not brain.tool_log:
        return "_No tool calls recorded in this session yet._"

    rows = [
        "| Time | Tool | Arguments | Result |",
        "| :--- | :--- | :--- | :--- |"
    ]
    for entry in reversed(brain.tool_log):
        args_str = ", ".join(f"`{k}={v}`" for k, v in entry["args"].items()) if entry["args"] else "None"
        res_str = str(entry["result"]).replace("\n", " ")
        rows.append(f"| `{entry['timestamp']}` | **{entry['tool']}** | {args_str} | {res_str} |")

    return "\n".join(rows)


def refresh_live_ui():
    """Timer callback that continuously pushes live conversation & tool state to the browser."""
    status_html = f"""
    <div style="padding: 10px 14px; border-radius: 8px; background: {'rgba(0, 240, 255, 0.15)' if is_voice_mode_active else 'rgba(255, 255, 255, 0.05)'}; border: 1px solid {'#00f0ff' if is_voice_mode_active else '#334155'};">
        <span style="font-size: 0.85rem; color: {'#00f0ff' if is_voice_mode_active else '#94a3b8'}; font-weight: bold; letter-spacing: 1px;">
            {'🔴 LIVE CONVERSATION ACTIVE' if is_voice_mode_active else '⚪ STANDBY MODE'}
        </span>
        <div style="font-size: 1.05rem; font-weight: 700; color: #ffffff; margin-top: 4px;">
            {voice_status_text}
        </div>
    </div>
    """
    return conversation_history, status_html, format_tool_logs()


def reset_dashboard():
    global conversation_history, is_voice_mode_active
    is_voice_mode_active = False
    brain.reset()
    conversation_history = []
    return [], None, "", "_Session memory & tool history cleared._"


CUSTOM_CSS = """
:root {
    --primary: #00f0ff;
    --bg-dark: #070d18;
}

body, .gradio-container {
    background: radial-gradient(circle at top right, #0d1e38 0%, #050b14 100%) !important;
    color: #e2f1f8 !important;
    font-family: 'Segoe UI', 'Roboto', monospace, sans-serif !important;
}

.jarvis-header {
    text-align: center;
    padding: 1.2rem;
    background: linear-gradient(180deg, rgba(0, 240, 255, 0.08) 0%, rgba(0, 0, 0, 0) 100%);
    border-bottom: 1px solid rgba(0, 240, 255, 0.25);
    margin-bottom: 1.2rem;
    border-radius: 12px;
}

.jarvis-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: 3px;
    color: #00f0ff;
    text-shadow: 0 0 20px rgba(0, 240, 255, 0.6);
    margin: 0;
}

.status-card {
    background: rgba(11, 21, 38, 0.85);
    border: 1px solid #14355a;
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.8rem;
}
"""

with gr.Blocks() as demo:
    # Top Banner
    gr.HTML("""
        <div class="jarvis-header">
            <h1 class="jarvis-title">⚡ J.A.R.V.I.S. ONE-STOP COMMAND CENTER ⚡</h1>
            <div style="font-size: 0.95rem; color: #8da4be; letter-spacing: 1.5px; margin-top: 0.4rem;">
                CONTINUOUS 2-WAY VOICE CONVERSATION • GEMINI 3.6 FLASH • LIVE TOOL HUD
            </div>
        </div>
    """)

    # Live UI Sync Timer (refreshes chat stream and tools every 0.8s)
    ui_timer = gr.Timer(value=0.8)

    with gr.Tabs():
        # TAB 1: Voice & Chat HUD
        with gr.TabItem("🎙️ Voice & Chat HUD"):
            with gr.Row():
                # Left Column: Chat Stream & Voice Controls
                with gr.Column(scale=8):
                    # Prominent Hands-Free Voice Control Bar
                    with gr.Row():
                        start_voice_btn = gr.Button("🔴 START HANDS-FREE VOICE ASSISTANT", variant="primary", scale=2)
                        stop_voice_btn = gr.Button("⏹️ STOP VOICE ASSISTANT", variant="stop", scale=1)
                        clear_btn = gr.Button("Reset Chat 🔄", variant="secondary", scale=1)

                    live_status_hud = gr.HTML("""
                        <div style="padding: 10px 14px; border-radius: 8px; background: rgba(255, 255, 255, 0.05); border: 1px solid #334155;">
                            <span style="font-size: 0.85rem; color: #94a3b8; font-weight: bold; letter-spacing: 1px;">⚪ STANDBY MODE</span>
                            <div style="font-size: 1.05rem; font-weight: 700; color: #ffffff; margin-top: 4px;">Idle (Standby) - Click Start Hands-Free Voice to begin conversation</div>
                        </div>
                    """)

                    chatbot = gr.Chatbot(
                        label="Live Interactive Stream (Auto-updates with Speech)",
                        height=380,
                    )

                    with gr.Row():
                        text_input = gr.Textbox(
                            show_label=False,
                            placeholder="Type a message manually or talk aloud when voice mode is on...",
                            scale=8,
                            lines=1,
                        )
                        send_btn = gr.Button("Send 🚀", variant="primary", scale=1)

                    with gr.Row():
                        voice_toggle = gr.Checkbox(label="🔊 Browser Voice Playback for typed messages", value=False)
                        audio_player = gr.Audio(label="Audio Output Player", type="filepath", interactive=False)

                    # Quick Shortcuts
                    gr.Markdown("#### ⚡ Quick Command Shortcuts:")
                    with gr.Row():
                        q1 = gr.Button("🕒 Time in New York")
                        q2 = gr.Button("🌦️ Weather in Tokyo")
                        q3 = gr.Button("🧮 Calculate (144 * 12) / 4")
                        q4 = gr.Button("⏰ Set 5-min Reminder")

                # Right Column: Telemetry & Live Tool Feed
                with gr.Column(scale=4):
                    gr.HTML(f"""
                        <div class="status-card">
                            <h4 style="margin:0 0 8px 0; color:#00f0ff;">📡 SYSTEM TELEMETRY</h4>
                            <p style="margin:4px 0;"><strong>Status:</strong> <span style="color:#00f0ff; font-weight:bold;">ONLINE / ACTIVE</span></p>
                            <p style="margin:4px 0;"><strong>Model:</strong> <code>{settings.gemini_model}</code></p>
                            <p style="margin:4px 0;"><strong>Voice VAD:</strong> <code>Energy RMS (1.3s Pause)</code></p>
                            <p style="margin:4px 0;"><strong>STT Engine:</strong> <code>faster-whisper (CPU)</code></p>
                            <p style="margin:4px 0;"><strong>TTS Engine:</strong> <code>pyttsx3 (Native Audio)</code></p>
                            <p style="margin:4px 0;"><strong>Tools Registered:</strong> <code>{len(TOOL_SPECS)} Active</code></p>
                        </div>
                    """)

                    gr.Markdown("### 🛠️ Live Tool Execution Feed")
                    tool_logs_display = gr.Markdown(value=format_tool_logs())
                    refresh_logs_btn = gr.Button("Refresh Feed 🔄", size="sm")

        # TAB 2: Direct Tool Diagnostics
        with gr.TabItem("⚙️ Tool Diagnostics & Playground"):
            gr.Markdown("### 🧪 Direct Diagnostic Panel for Registered JARVIS Tools")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🕒 Timezone Diagnostic")
                    tz_in = gr.Textbox(value="America/New_York", label="Timezone Name")
                    t_btn = gr.Button("Query Time", size="sm")
                    t_out = gr.Textbox(label="Output", interactive=False)
                    t_btn.click(get_current_time, inputs=[tz_in], outputs=[t_out])

                with gr.Column():
                    gr.Markdown("#### 🌦️ Weather Diagnostic")
                    city_in = gr.Textbox(value="Brampton", label="City")
                    w_btn = gr.Button("Query Weather", size="sm")
                    w_out = gr.Textbox(label="Output", interactive=False)
                    w_btn.click(get_weather, inputs=[city_in], outputs=[w_out])

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🧮 Calculator Diagnostic")
                    c_in = gr.Textbox(value="(144 * 12) / 4", label="Expression")
                    c_btn = gr.Button("Calculate", size="sm")
                    c_out = gr.Textbox(label="Output", interactive=False)
                    c_btn.click(calculate, inputs=[c_in], outputs=[c_out])

                with gr.Column():
                    gr.Markdown("#### ⏰ Reminder Diagnostic")
                    r_msg = gr.Textbox(value="Review Databricks Deployment", label="Message")
                    r_min = gr.Number(value=5, label="Minutes")
                    r_btn = gr.Button("Set Reminder", size="sm")
                    r_out = gr.Textbox(label="Output", interactive=False)
                    r_btn.click(lambda m, mins: set_reminder(m, int(mins)), inputs=[r_msg, r_min], outputs=[r_out])

        # TAB 3: Specs
        with gr.TabItem("📐 Architecture & Databricks Specs"):
            gr.Markdown("""
            ### 🏗️ Mini JARVIS Architecture Overview

            ```
            ┌─────────────────────────────────────────────────────────────┐
            │ ONE-STOP DASHBOARD (dashboard.py)                           │
            │ • Background Conversational Thread                          │
            │ • Dynamic Microphone VAD (Speech-to-Text via faster-whisper)│
            │ • Gemini 3.6 Flash Brain + Function Calling (app/brain.py)  │
            │ • Unity Catalog Tools Dispatch (app/tools.py)               │
            │ • Native Speaker Output (TTS via pyttsx3)                   │
            │ • Real-time Live UI Sync & Tool Logs Display                │
            └──────────────────────────────┬──────────────────────────────┘
                                           │ CI/CD Deploy
            ┌──────────────────────────────▼──────────────────────────────┐
            │ DATABRICKS ASSET BUNDLE (DAB)                               │
            │ • Unity Catalog Schema: agentic_catalog.mini_jarvis         │
            │ • Serverless Setup Job: resources/jarvis_job.yml            │
            │ • Gemini External Model Endpoint                            │
            │ • MLflow PyFunc Registered Agent Model                      │
            │ • Databricks App (Gradio Web UI): src/app/app.py            │
            └─────────────────────────────────────────────────────────────┘
            ```

            - **GitHub Repository**: [Pratik2895/mini-jarvis](https://github.com/Pratik2895/mini-jarvis)
            - **Databricks Workspace**: [dbc-316f5fb6-3c9c.cloud.databricks.com](https://dbc-316f5fb6-3c9c.cloud.databricks.com/)
            - **Secret Scope**: `mini-jarvis` (GEMINI_API_KEY)
            """)

    # Event Bindings
    # Start / Stop Voice Mode
    start_voice_btn.click(start_voice_mode, inputs=[], outputs=[])
    stop_voice_btn.click(stop_voice_mode, inputs=[], outputs=[])

    # Text Input Submit
    text_input.submit(
        process_text_input,
        inputs=[text_input, chatbot, voice_toggle],
        outputs=[chatbot, audio_player, text_input, tool_logs_display]
    )
    send_btn.click(
        process_text_input,
        inputs=[text_input, chatbot, voice_toggle],
        outputs=[chatbot, audio_player, text_input, tool_logs_display]
    )

    # Periodic UI Refresh (syncs live speech and tool updates with the dashboard)
    ui_timer.tick(
        refresh_live_ui,
        inputs=[],
        outputs=[chatbot, live_status_hud, tool_logs_display]
    )

    # Reset
    clear_btn.click(
        reset_dashboard,
        inputs=[],
        outputs=[chatbot, audio_player, text_input, tool_logs_display]
    )
    refresh_logs_btn.click(
        format_tool_logs,
        inputs=[],
        outputs=[tool_logs_display]
    )

    # Quick Shortcuts
    q1.click(lambda: "What time is it in New York right now?", None, text_input)
    q2.click(lambda: "What is the weather in Tokyo?", None, text_input)
    q3.click(lambda: "What is (144 * 12) / 4?", None, text_input)
    q4.click(lambda: "Set a reminder to review code in 5 minutes", None, text_input)


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  🚀 Launching JARVIS All-in-One Command Center Dashboard...")
    print("  🔗 Access the dashboard at: http://127.0.0.1:7860")
    print("=" * 65 + "\n")
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, css=CUSTOM_CSS, theme=gr.themes.Default())
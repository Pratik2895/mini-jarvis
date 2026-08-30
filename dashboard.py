"""dashboard.py — High-Tech JARVIS Web Control Center & Dashboard.

Run with:
    python dashboard.py
or:
    python -m dashboard
"""
from __future__ import annotations

import datetime
import os
import sys
import tempfile
from pathlib import Path

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import gradio as gr
from app.brain import JarvisBrain
from app.config import settings
from app.tools import calculate, get_current_time, get_weather, set_reminder, TOOL_SPECS
from client.speech_to_text import transcribe_file
from client.text_to_speech import text_to_wav

# Global Brain Instance
brain = JarvisBrain()

CUSTOM_CSS = """
/* Futuristic JARVIS HUD Theme */
:root {
    --primary-hue: 185;
    --primary: #00f0ff;
    --primary-dark: #008b99;
    --bg-dark: #070d18;
    --card-bg: #0b1526;
    --border-color: #12304d;
}

body, .gradio-container {
    background: radial-gradient(circle at top right, #0d1e38 0%, #050b14 100%) !important;
    color: #e2f1f8 !important;
    font-family: 'Segoe UI', 'Roboto', monospace, sans-serif !important;
}

.jarvis-header {
    text-align: center;
    padding: 1.5rem;
    background: linear-gradient(180deg, rgba(0, 240, 255, 0.08) 0%, rgba(0, 0, 0, 0) 100%);
    border-bottom: 1px solid rgba(0, 240, 255, 0.25);
    margin-bottom: 1.5rem;
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

.jarvis-subtitle {
    font-size: 0.95rem;
    color: #8da4be;
    letter-spacing: 1.5px;
    margin-top: 0.4rem;
}

.status-card {
    background: rgba(11, 21, 38, 0.85);
    border: 1px solid #14355a;
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
}

.status-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: bold;
    background: rgba(0, 240, 255, 0.15);
    color: #00f0ff;
    border: 1px solid #00f0ff;
}

.hud-panel {
    background: rgba(11, 21, 38, 0.75) !important;
    border: 1px solid #14355a !important;
    border-radius: 10px !important;
}
"""


def process_user_interaction(user_text, audio_path, history, enable_voice):
    """Handles both text input and recorded audio input."""
    if history is None:
        history = []

    prompt = ""
    if audio_path:
        try:
            prompt = transcribe_file(audio_path)
        except Exception as e:
            prompt = f"[Audio Transcription Failed: {e}]"
    elif user_text and user_text.strip():
        prompt = user_text.strip()

    if not prompt:
        return history, None, "", None, format_tool_logs()

    # Query the brain
    response = brain.chat(prompt)
    history.append((prompt, response))

    # Generate speech audio if enabled
    audio_output = None
    if enable_voice and response:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                wav_path = text_to_wav(response, tmp.name)
                audio_output = wav_path
        except Exception as e:
            print(f"[TTS Error in Dashboard]: {e}")

    updated_logs = format_tool_logs()
    return history, audio_output, "", None, updated_logs


def format_tool_logs():
    """Format tool execution history into a neat Markdown table."""
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


def reset_dashboard():
    """Resets memory and logs."""
    brain.reset()
    return [], None, "", None, "_Session memory & tool history cleared._"


def test_time_tool(tz):
    return get_current_time(tz)


def test_weather_tool(city):
    return get_weather(city)


def test_calc_tool(expr):
    return calculate(expr)


def test_reminder_tool(msg, mins):
    return set_reminder(msg, int(mins))


with gr.Blocks() as demo:
    # Header Banner
    gr.HTML("""
        <div class="jarvis-header">
            <h1 class="jarvis-title">⚡ J.A.R.V.I.S. COMMAND CENTER ⚡</h1>
            <div class="jarvis-subtitle">VOICE & MULTI-AGENT INTELLIGENCE SYSTEM • GEMINI 3.6 FLASH • DATABRICKS READY</div>
        </div>
    """)

    with gr.Tabs():
        # TAB 1: Main Assistant
        with gr.TabItem("🎙️ Voice & Chat HUD"):
            with gr.Row():
                with gr.Column(scale=8):
                    chatbot = gr.Chatbot(
                        label="Live Interaction Stream",
                        height=420,
                    )

                    with gr.Row():
                        text_input = gr.Textbox(
                            show_label=False,
                            placeholder="Type a message or use the microphone below... (Press Enter to send)",
                            scale=8,
                            lines=1,
                        )
                        send_btn = gr.Button("Send 🚀", variant="primary", scale=1)
                        clear_btn = gr.Button("Reset 🔄", variant="secondary", scale=1)

                    with gr.Row():
                        audio_input = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            label="🎤 Microphone Audio Input (Speaks into faster-whisper STT)",
                            scale=5,
                        )
                        with gr.Column(scale=5):
                            voice_toggle = gr.Checkbox(label="🔊 Synthesize Voice Response (pyttsx3 TTS)", value=True)
                            audio_player = gr.Audio(label="JARVIS Voice Output", type="filepath", interactive=False)

                    # Quick Prompt Suggestions
                    gr.Markdown("#### ⚡ Quick Command Shortcuts:")
                    with gr.Row():
                        q1 = gr.Button("🕒 Time in New York")
                        q2 = gr.Button("🌦️ Weather in Tokyo")
                        q3 = gr.Button("🧮 Calculate (144 * 12) / 4")
                        q4 = gr.Button("⏰ Set 5-min Reminder")

                # Right Sidebar Telemetry
                with gr.Column(scale=4):
                    gr.HTML(f"""
                        <div class="status-card">
                            <h4 style="margin:0 0 8px 0; color:#00f0ff;">📡 SYSTEM TELEMETRY</h4>
                            <p style="margin:4px 0;"><strong>Status:</strong> <span class="status-badge">ONLINE / ACTIVE</span></p>
                            <p style="margin:4px 0;"><strong>Model:</strong> <code>{settings.gemini_model}</code></p>
                            <p style="margin:4px 0;"><strong>STT Engine:</strong> <code>faster-whisper (CPU)</code></p>
                            <p style="margin:4px 0;"><strong>TTS Engine:</strong> <code>pyttsx3 (offline)</code></p>
                            <p style="margin:4px 0;"><strong>Databricks Host:</strong> <code>dbc-316f5fb6...</code></p>
                            <p style="margin:4px 0;"><strong>Tools Registered:</strong> <code>{len(TOOL_SPECS)} Active</code></p>
                        </div>
                    """)

                    gr.Markdown("### 🛠️ Live Tool Execution Feed")
                    tool_logs_display = gr.Markdown(value=format_tool_logs())
                    refresh_logs_btn = gr.Button("Refresh Feed 🔄", size="sm")

        # TAB 2: Tool Playground
        with gr.TabItem("⚙️ Tool Diagnostics & Playground"):
            gr.Markdown("### 🧪 Direct Diagnostic Panel for Registered JARVIS Tools")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🕒 Timezone Diagnostic")
                    tz_input = gr.Textbox(value="America/New_York", label="Timezone Name")
                    time_btn = gr.Button("Query Time", size="sm")
                    time_out = gr.Textbox(label="Output", interactive=False)
                    time_btn.click(test_time_tool, inputs=[tz_input], outputs=[time_out])

                with gr.Column():
                    gr.Markdown("#### 🌦️ Weather Diagnostic")
                    city_input = gr.Textbox(value="London", label="City")
                    weather_btn = gr.Button("Query Weather", size="sm")
                    weather_out = gr.Textbox(label="Output", interactive=False)
                    weather_btn.click(test_weather_tool, inputs=[city_input], outputs=[weather_out])

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🧮 Calculator Diagnostic")
                    calc_input = gr.Textbox(value="2 ** 16", label="Arithmetic Expression")
                    calc_btn = gr.Button("Calculate", size="sm")
                    calc_out = gr.Textbox(label="Output", interactive=False)
                    calc_btn.click(test_calc_tool, inputs=[calc_input], outputs=[calc_out])

                with gr.Column():
                    gr.Markdown("#### ⏰ Reminder Diagnostic")
                    rem_msg = gr.Textbox(value="Review Databricks Deployment", label="Message")
                    rem_min = gr.Number(value=5, label="Minutes")
                    rem_btn = gr.Button("Set Reminder", size="sm")
                    rem_out = gr.Textbox(label="Output", interactive=False)
                    rem_btn.click(test_reminder_tool, inputs=[rem_msg, rem_min], outputs=[rem_out])

        # TAB 3: System Architecture & Deployment Docs
        with gr.TabItem("📐 Architecture & Databricks Specs"):
            gr.Markdown("""
            ### 🏗️ Mini JARVIS Architecture Overview

            ```
            ┌─────────────────────────────────────────────────────────────┐
            │ LOCAL CLIENT (dashboard.py / jarvis_client.py)              │
            │ Microphone → faster-whisper STT                             │
            │      ↓                                                      │
            │ Gemini 3.6 Flash Brain (app/brain.py)                       │
            │      ↓ (Tool Calling Protocol)                              │
            │ Local & UC Tool Dispatch (app/tools.py)                     │
            │      ↓                                                      │
            │ pyttsx3 TTS → Speaker                                       │
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

    # Wire up Event Handlers
    # Text input submit
    text_input.submit(
        process_user_interaction,
        inputs=[text_input, audio_input, chatbot, voice_toggle],
        outputs=[chatbot, audio_player, text_input, audio_input, tool_logs_display]
    )
    # Send button click
    send_btn.click(
        process_user_interaction,
        inputs=[text_input, audio_input, chatbot, voice_toggle],
        outputs=[chatbot, audio_player, text_input, audio_input, tool_logs_display]
    )
    # Audio change (when user records or uploads audio)
    audio_input.stop_recording(
        process_user_interaction,
        inputs=[text_input, audio_input, chatbot, voice_toggle],
        outputs=[chatbot, audio_player, text_input, audio_input, tool_logs_display]
    )
    # Clear / Reset
    clear_btn.click(
        reset_dashboard,
        inputs=[],
        outputs=[chatbot, audio_player, text_input, audio_input, tool_logs_display]
    )
    # Refresh logs
    refresh_logs_btn.click(
        format_tool_logs,
        inputs=[],
        outputs=[tool_logs_display]
    )

    # Quick prompt buttons
    q1.click(lambda: "What time is it in New York right now?", None, text_input)
    q2.click(lambda: "What is the weather in Tokyo?", None, text_input)
    q3.click(lambda: "What is (144 * 12) / 4?", None, text_input)
    q4.click(lambda: "Set a reminder to review code in 5 minutes", None, text_input)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🚀 Launching JARVIS Interactive Control Center Dashboard...")
    print("  🔗 Access the dashboard in your browser at: http://127.0.0.1:7860")
    print("=" * 60 + "\n")
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, css=CUSTOM_CSS, theme=gr.themes.Default())
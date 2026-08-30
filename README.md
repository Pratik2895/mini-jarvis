# Mini JARVIS — Voice AI Assistant
# LLM: Gemini 2.0 Flash  |  STT: faster-whisper  |  TTS: pyttsx3
# Local test → Databricks DAB deploy

## Quick Start (Local)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Copy and fill in your env
copy .env.example .env

# Stage 1: Interactive Control Center Dashboard (Web UI + Voice)
python dashboard.py

# Stage 2: CLI text chatbot
python -m app.brain

# Stage 3: Headless local voice loop
python client/jarvis_client.py
```

## Deploy to Databricks
```bash
databricks bundle deploy --target dev
databricks bundle run jarvis_setup_job --target dev
```


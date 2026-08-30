"""app/config.py � Central settings loaded from .env or environment variables."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # Databricks (used by local client to call deployed endpoint directly)
    databricks_host: str = "https://dbc-316f5fb6-3c9c.cloud.databricks.com"
    databricks_token: str = ""
    databricks_agent_endpoint: str = "mini-jarvis-agent-endpoint"

    # FastAPI
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # STT
    whisper_model_size: str = "base"  # tiny | base | small | medium

    # TTS
    tts_rate: int = 175  # words per minute
    tts_volume: float = 1.0


settings = Settings()

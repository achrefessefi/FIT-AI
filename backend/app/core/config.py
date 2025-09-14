# backend/app/core/config.py
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv, find_dotenv

# Load nearest .env (repo root)
load_dotenv(find_dotenv(), override=False)

class Settings(BaseSettings):
    # Server
    APP_NAME: str = "Fitness API"
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000

    # CORS (comma-separated)
    CORS_ORIGINS: str = "http://localhost:5173"

    # Server-side secrets
    GROQ_API_KEY: str | None = None

    # Accept extra env vars without error (e.g., VITE_API_BASE, workout tuner keys, etc.)
    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        extra="ignore",            # <-- this line makes your current .env safe
        case_sensitive=False,
    )

settings = Settings()

def get_cors_origins() -> List[str]:
    return [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]

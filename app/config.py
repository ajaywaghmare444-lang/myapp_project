import os
# pyrefly: ignore [missing-import]
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # App Settings
    PROJECT_NAME: str = "FastAPI LLM Agent Backend"
    API_PREFIX: str = "/api/v1"
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    CORS_ORIGINS: list[str] = ["*"]

    # LLM Settings
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_API_BASE: Optional[str] = None
    MODEL_NAME: str = "gpt-5-mini"

    # Atlassian Settings
    ATLASSIAN_BASE_URL: Optional[str] = None
    ATLASSIAN_EMAIL: Optional[str] = None
    ATLASSIAN_API_TOKEN: Optional[str] = None

    # Use dotenv to load environment variables for local development
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

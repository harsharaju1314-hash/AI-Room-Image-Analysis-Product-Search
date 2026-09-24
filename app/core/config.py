import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "AI Room Image Analysis & Product Search"
    APP_ENV: str = "development"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Security & Input Validation
    MAX_UPLOAD_SIZE_MB: int = 5
    ALLOWED_EXTENSIONS: List[str] = ["jpg", "jpeg", "png", "webp"]

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/room_search_db"
    )
    USE_DB_MOCK_FALLBACK: bool = os.getenv("USE_DB_MOCK_FALLBACK", "True").lower() in ("true", "1", "yes")

    # Model & Feature Extraction
    MODEL_NAME: str = "resnet18"
    EMBEDDING_DIM: int = 512
    TOP_K_DEFAULT: int = 3


settings = Settings()

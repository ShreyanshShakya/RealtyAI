"""Central application configuration."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings loaded from environment variables and an optional .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Scraping
    REQUEST_TIMEOUT: int = 30
    REQUEST_DELAY: float = 1.5
    REQUEST_JITTER: float = 0.5
    MAX_RETRIES: int = 3
    USER_AGENT: str = "RealtyAI/1.0 (+https://github.com/ShreyanshShakya/RealtyAI)"
    PLAYWRIGHT_HEADLESS: bool = True
    PLAYWRIGHT_TIMEOUT: int = 30_000

    # Dataset and persistence paths
    RAW_DATA_DIR: str = "dataset/raw"
    PROCESSED_DATA_DIR: str = "dataset/processed"
    REJECTED_DATA_DIR: str = "dataset/rejected"
    EXCLUDED_DATA_DIR: str = "dataset/excluded"
    CHROMA_DB_PATH: str = "dataset/chroma_db"
    CHROMA_COLLECTION: str = "realtyai_projects"

    # Source websites
    DARGLOBAL_BASE_URL: str = "https://darglobal.co.uk"
    DARGLOBAL_PROJECTS_PATH: str = "/projects/"
    WASALT_BASE_URL: str = "https://wasalt.sa"
    WASALT_LISTINGS_PATH: str = "/en/properties"

    # Retrieval and embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE: int = 32
    RETRIEVAL_TOP_K: int = 5
    RETRIEVAL_MIN_RESULTS: int = 1

    # OpenRouter
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    OPENROUTER_FALLBACK_MODEL: str = "openrouter/free"
    OPENROUTER_SITE_URL: str = ""
    OPENROUTER_APP_NAME: str = "RealtyAI"
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 1200
    LLM_TIMEOUT: int = 60


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()

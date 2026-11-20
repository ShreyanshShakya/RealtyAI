from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # Scraping
    REQUEST_TIMEOUT: int = 30
    REQUEST_DELAY: float = 1.5
    REQUEST_JITTER: float = 0.5
    MAX_RETRIES: int = 3
    USER_AGENT: str = "RealtyAI/1.0 (+https://github.com/yourusername/realty-ai)"
    PLAYWRIGHT_HEADLESS: bool = True
    PLAYWRIGHT_TIMEOUT: int = 30000
    
    # Paths
    RAW_DATA_DIR: str = "dataset/raw"
    PROCESSED_DATA_DIR: str = "dataset/processed"
    REJECTED_DATA_DIR: str = "dataset/rejected"
    EXCLUDED_DATA_DIR: str = "dataset/excluded"
    CHROMA_DB_PATH: str = "dataset/chroma_db"
    
    # Scraping URLs
    DARGLOBAL_BASE_URL: str = "https://darglobal.co.uk"
    DARGLOBAL_PROJECTS_PATH: str = "/projects/"
    WASALT_BASE_URL: str = "https://wasalt.sa"
    WASALT_LISTINGS_PATH: str = "/en/properties"
    
    # RAG Configuration
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHROMA_DB_PATH: str = "dataset/chroma_db"
    CHROMA_COLLECTION: str = "realtyai_projects"
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
    
    # Scraping URLs
    DARGLOBAL_BASE_URL: str = "https://darglobal.co.uk"
    DARGLOBAL_PROJECTS_PATH: str = "/projects/"
    WASALT_BASE_URL: str = "https://wasalt.sa"
    WASALT_LISTINGS_PATH: str = "/en/properties"
    
    class Config:
        env_file = ".env"
        extra = "ignore"

@lru_cache
def get_settings() -> Settings:
    return Settings()
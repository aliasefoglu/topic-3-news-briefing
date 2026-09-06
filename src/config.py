from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    ## LLM Settings
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-6"

    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None

    ## Embedding Settings
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"

    ## Application Settings
    log_level: str = "INFO"
    database_url: str = "postgresql+asyncpg://postgres:dev@localhost:5432/newsbrief"
    digests_dir: str = "./digests"

    ## Fetch and deduplication settings
    dedup_near_duplicate_threshold: float = 0.70
    fetch_timeout_seconds: int = 15
    max_parallel_fetches: int = 8

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
settings = Settings()
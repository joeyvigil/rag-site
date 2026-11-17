from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration comes from environment variables (or a .env file)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Infrastructure
    database_url: str = "postgresql+asyncpg://rag:rag@localhost:5432/rag"
    redis_url: str = "redis://localhost:6379"
    storage_dir: str = "./data/uploads"

    # Auth
    jwt_secret: str = "dev-secret-change-me-to-a-long-random-string"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24

    gemini_api_key: str = ""

    # LLM (any OpenAI-compatible chat completions endpoint)
    llm_provider: Literal["openai_compat", "fake"] = "openai_compat"
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_api_key: str = ""
    llm_model: str = "gemini-3.5-flash-lite"
    # Comma-separated; tried in order if the primary model errors or times out.
    llm_fallback_models: str = "gemini-3.1-flash-lite,gemini-3.6-flash"
    llm_timeout_seconds: float = 30
    llm_temperature: float = 0.2

    # Embeddings
    embedding_provider: Literal["gemini", "fake"] = "gemini"
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768
    embedding_batch_size: int = 100

    # Chunking / retrieval
    chunk_size: int = 1200  # characters (~300 tokens)
    chunk_overlap: int = 200
    retrieval_top_k: int = 6
    max_history_messages: int = 6

    # Uploads
    max_upload_mb: int = 25

    @property
    def llm_fallback_model_list(self) -> list[str]:
        return [m.strip() for m in self.llm_fallback_models.split(",") if m.strip()]

    @property
    def effective_llm_api_key(self) -> str:
        return self.llm_api_key or self.gemini_api_key


settings = Settings()

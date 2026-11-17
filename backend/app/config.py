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

    # Chunking
    chunk_size: int = 1200  # characters (~300 tokens)
    chunk_overlap: int = 200
    # Repeat a CSV's header row at the top of every chunk of that CSV, so each
    # chunk is self-describing (otherwise only the first chunk has column names).
    csv_header_in_chunks: bool = True

    # Retrieval
    retrieval_top_k: int = 6  # chunks given to the LLM
    # "vector" (embeddings only), "keyword" (Postgres full-text only), or
    # "hybrid" (both, merged with Reciprocal Rank Fusion)
    # Default chosen by evaluation (Phase 3): hybrid raised correctness from
    # 0.917 to 0.976 for +10% cost; an LLM reranker added only noise-level
    # gains for 3.5x the cost, so it's off by default.
    retrieval_mode: Literal["vector", "keyword", "hybrid"] = "hybrid"
    hybrid_candidates: int = 30  # results taken from EACH retriever before fusion
    rrf_k: int = 60  # RRF constant: score = sum(1 / (rrf_k + rank))
    # "none", "cross_encoder" (local ONNX model, CPU), or "llm" (listwise via the chat model)
    reranker: Literal["none", "cross_encoder", "llm"] = "none"
    rerank_candidates: int = 20  # how many first-stage results the reranker reorders
    cross_encoder_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    model_cache_dir: str = "./data/models"
    max_history_messages: int = 6

    # OCR for scanned PDFs and images: "gemini" (vision LLM) or "tesseract" (local)
    ocr_provider: Literal["gemini", "tesseract"] = "gemini"
    ocr_model: str = "gemini-3.5-flash-lite"
    ocr_dpi: int = 150

    # Uploads
    max_upload_mb: int = 25

    @property
    def llm_fallback_model_list(self) -> list[str]:
        return [m.strip() for m in self.llm_fallback_models.split(",") if m.strip()]

    @property
    def effective_llm_api_key(self) -> str:
        return self.llm_api_key or self.gemini_api_key


settings = Settings()

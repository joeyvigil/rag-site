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

    # Phase 4 -------------------------------------------------------------
    # Rewrite follow-up questions ("what's its budget?") into standalone search
    # queries using the conversation. Only runs when there is history.
    query_rewrite: bool = True
    # Context added to each chunk before embedding/indexing:
    # "none", "title" (filename/page label, free), or "llm" (an LLM-written
    # sentence situating the chunk in its document; one call per chunk at ingest)
    chunk_context: Literal["none", "title", "llm"] = "none"
    # How questions are answered:
    # "rag" (retrieve top-k, then answer), "agent" (the LLM calls search/read
    # tools in a loop), or "long_context" (no retrieval: all documents in the prompt)
    answer_mode: Literal["rag", "agent", "long_context"] = "rag"
    agent_max_steps: int = 6
    agent_read_max_chars: int = 40_000  # read_document tool truncates long docs
    long_context_max_chars: int = 400_000  # ~100k tokens; refuse beyond this

    # OCR for scanned PDFs and images: "gemini" (vision LLM) or "tesseract" (local)
    ocr_provider: Literal["gemini", "tesseract"] = "gemini"
    ocr_model: str = "gemini-3.5-flash-lite"
    ocr_dpi: int = 150

    # Uploads
    max_upload_mb: int = 25

    # Phase 5: scaling & load testing ------------------------------------
    # DB connection pool, PER PROCESS. Total connections = processes x
    # (pool_size + max_overflow) and must stay under Postgres max_connections (100).
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout: float = 30  # seconds to wait for a free connection before erroring
    # Fake providers pretend to be a real API (so load tests measure realistic
    # I/O waits, not an instant echo): time to first token, then a steady token rate.
    fake_llm_ttft_ms: int = 0
    fake_llm_tokens_per_s: float = 0  # 0 = as fast as possible (~10 ms/token)
    fake_embed_latency_ms: int = 0
    # Per-user chat requests per minute (0 = unlimited). Protects the shared
    # LLM quota from one user (or one runaway script).
    rate_limit_chat_per_minute: int = 20
    # Exact-match answer cache TTL for questions without history (0 = off).
    answer_cache_ttl_seconds: int = 3600

    @property
    def llm_fallback_model_list(self) -> list[str]:
        return [m.strip() for m in self.llm_fallback_models.split(",") if m.strip()]

    @property
    def effective_llm_api_key(self) -> str:
        return self.llm_api_key or self.gemini_api_key


settings = Settings()

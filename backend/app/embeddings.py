"""Embeddings with a content-addressed cache.

Providers:
- "gemini": Google's embedding API (free tier). Called directly over HTTP so
  every parameter is visible. Uses *asymmetric* task types: documents are
  embedded with RETRIEVAL_DOCUMENT and questions with RETRIEVAL_QUERY.
- "fake": deterministic keyword-hash vectors. No network, no key. Retrieval
  then behaves like crude keyword matching, which is enough for tests and
  load testing.
"""

import hashlib
import logging
import math
import re
from typing import Literal

import httpx
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.config import settings
from app.models import EmbeddingCache

log = logging.getLogger(__name__)

TaskType = Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]

_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents"
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

_http: httpx.AsyncClient | None = None


class EmbeddingError(RuntimeError):
    pass


def _client() -> httpx.AsyncClient:
    global _http
    if _http is None:
        _http = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))
    return _http


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


async def embed_documents(session: AsyncSession, texts: list[str]) -> list[list[float]]:
    return await _embed_cached(session, texts, "RETRIEVAL_DOCUMENT")


async def embed_query(session: AsyncSession, text: str) -> list[float]:
    return (await _embed_cached(session, [text], "RETRIEVAL_QUERY"))[0]


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------


def _cache_key(text: str, task: TaskType) -> str:
    ident = f"{settings.embedding_provider}|{settings.embedding_model}|{settings.embedding_dim}|{task}|{text}"
    return hashlib.sha256(ident.encode()).hexdigest()


async def _embed_cached(session: AsyncSession, texts: list[str], task: TaskType) -> list[list[float]]:
    keys = [_cache_key(t, task) for t in texts]
    found: dict[str, list[float]] = {}

    unique_keys = list(dict.fromkeys(keys))
    for i in range(0, len(unique_keys), 1000):
        rows = await session.execute(
            select(EmbeddingCache.key, EmbeddingCache.embedding).where(
                EmbeddingCache.key.in_(unique_keys[i : i + 1000])
            )
        )
        for key, vec in rows:
            found[key] = _to_list(vec)

    missing = {k: t for k, t in zip(keys, texts, strict=True) if k not in found}
    if missing:
        log.info("embedding %d texts (%d cache hits)", len(missing), len(unique_keys) - len(missing))
        vectors = await _provider_embed(list(missing.values()), task)
        new_rows = [{"key": k, "embedding": v} for k, v in zip(missing, vectors, strict=True)]
        for row in new_rows:
            found[row["key"]] = row["embedding"]
        for i in range(0, len(new_rows), 500):
            await session.execute(
                pg_insert(EmbeddingCache).values(new_rows[i : i + 500]).on_conflict_do_nothing()
            )

    return [found[k] for k in keys]


def _to_list(vec) -> list[float]:
    return vec.tolist() if hasattr(vec, "tolist") else list(vec)


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        out = [0.0] * len(vec)
        out[0] = 1.0  # avoid a zero vector (cosine distance undefined)
        return out
    return [x / norm for x in vec]


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------


async def _provider_embed(texts: list[str], task: TaskType) -> list[list[float]]:
    if settings.embedding_provider == "fake":
        return [_fake_embed(t) for t in texts]

    if not settings.gemini_api_key:
        raise EmbeddingError(
            "GEMINI_API_KEY is not set. Add it to .env (or set EMBEDDING_PROVIDER=fake)."
        )
    out: list[list[float]] = []
    bs = settings.embedding_batch_size
    for i in range(0, len(texts), bs):
        out.extend(await _gemini_batch(texts[i : i + bs], task))
    return out


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        if "PerDay" in exc.response.text:  # daily quota: retrying won't help until it resets
            return False
        return exc.response.status_code in _RETRYABLE_STATUS
    return isinstance(exc, httpx.TransportError)


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential_jitter(multiplier=2, max=60),
    stop=stop_after_attempt(6),
    reraise=True,
    before_sleep=lambda rs: log.warning(
        "embedding call failed (%s), retry #%d", rs.outcome.exception(), rs.attempt_number
    ),
)
async def _gemini_batch(texts: list[str], task: TaskType) -> list[list[float]]:
    model = settings.embedding_model
    body = {
        "requests": [
            {
                "model": f"models/{model}",
                "content": {"parts": [{"text": t}]},
                "taskType": task,
                "outputDimensionality": settings.embedding_dim,
            }
            for t in texts
        ]
    }
    resp = await _client().post(
        _GEMINI_URL.format(model=model),
        json=body,
        headers={"x-goog-api-key": settings.gemini_api_key},
    )
    if resp.status_code in _RETRYABLE_STATUS:
        resp.raise_for_status()  # -> HTTPStatusError, retried with backoff
    if resp.status_code >= 400:
        raise EmbeddingError(f"Gemini embedding error {resp.status_code}: {resp.text[:500]}")

    embeddings = resp.json()["embeddings"]
    # Truncated (non-default-size) Gemini embeddings aren't unit length; normalize.
    return [_normalize(e["values"]) for e in embeddings]


def _fake_embed(text: str) -> list[float]:
    vec = [0.0] * settings.embedding_dim
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        h = int.from_bytes(hashlib.md5(token.encode()).digest()[:4], "little")
        vec[h % settings.embedding_dim] += 1.0
    return _normalize(vec)

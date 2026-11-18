"""Redis-backed rate limiting and answer caching.

Both live in Redis (not process memory) so they work across API processes
and replicas: a per-process counter would let a user get N x the limit by
hitting N workers.

Rate limit: fixed window. One counter per user per minute (INCR + EXPIRE in
one round trip). Simple and cheap; its known flaw is that a user can burst
up to 2x the limit across a window boundary (end of minute 1 + start of
minute 2). A sliding window or token bucket fixes that at the cost of a Lua
script; for protecting an LLM quota, fixed windows are fine.

Answer cache: exact match on (user, corpus version, mode, documents,
normalized question). Only for questions without chat history, since a
follow-up's meaning depends on the conversation.

Invalidation is the hard part of caching. Instead of finding and deleting
stale entries, every user has a "corpus version" counter, bumped whenever
their searchable content changes (a document finishes ingesting or is
deleted). The version is part of the key, so old entries simply stop being
looked up and expire by TTL. No scan, no race with in-flight requests.

Why not a *semantic* cache (match similar questions by embedding)? It hits
more often, but "PTO for full-time staff" and "PTO for part-time staff" are
near-identical vectors with different answers. A wrong cached answer is
worse than a slow right one, so it would need a very high similarity
threshold and its own evaluation.
"""

import hashlib
import json
import re
import time
import uuid

from redis.asyncio import Redis

from app.config import settings

_CORPUS_VERSION = "corpus_ver:{user_id}"


async def check_rate_limit(redis: Redis, key: str, limit: int, window_s: int = 60) -> int | None:
    """Count one request. Returns None if allowed, else seconds until the window resets."""
    if limit <= 0:
        return None
    now = time.time()
    window = int(now // window_s)
    counter = f"rl:{key}:{window}"
    async with redis.pipeline(transaction=True) as pipe:
        pipe.incr(counter)
        pipe.expire(counter, window_s + 1)
        count, _ = await pipe.execute()
    if count > limit:
        return max(1, int((window + 1) * window_s - now))
    return None


async def bump_corpus_version(redis: Redis, user_id: uuid.UUID) -> None:
    await redis.incr(_CORPUS_VERSION.format(user_id=user_id))


def normalize_question(q: str) -> str:
    """Case, whitespace and trailing punctuation don't change the meaning."""
    return re.sub(r"\s+", " ", q).strip().lower().rstrip("?!. ")


async def answer_cache_key(
    redis: Redis, user_id: uuid.UUID, question: str, mode: str, document_ids: list | None
) -> str:
    version = await redis.get(_CORPUS_VERSION.format(user_id=user_id)) or b"0"
    docs = ",".join(sorted(str(d) for d in document_ids)) if document_ids else "*"
    ident = f"{user_id}|{version.decode() if isinstance(version, bytes) else version}|{mode}|{docs}|{normalize_question(question)}"
    return "answer:" + hashlib.sha256(ident.encode()).hexdigest()


async def get_cached_answer(redis: Redis, key: str) -> dict | None:
    raw = await redis.get(key)
    return json.loads(raw) if raw else None


async def store_answer(redis: Redis, key: str, value: dict) -> None:
    await redis.set(key, json.dumps(value), ex=settings.answer_cache_ttl_seconds)

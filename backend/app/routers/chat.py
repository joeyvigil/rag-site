import json
import logging
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache import answer_cache_key, check_rate_limit, get_cached_answer, store_answer
from app.config import settings
from app.db import get_session
from app.embeddings import EmbeddingError
from app.llm import stream_chat
from app.models import User
from app.observability import ANSWER_CACHE, CHAT_ERRORS, CHAT_STAGE, LLM_TOKENS, RATE_LIMITED
from app.pipeline import prepare_answer
from app.rag import ContextTooLarge
from app.schemas import ChatRequest, SourceOut
from app.security import get_current_user

log = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])

NO_SOURCES_MESSAGE = "I couldn't find anything relevant. Upload some documents and wait until they show as ready."


def _sse(event: str, data: object) -> str:
    """Format one Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.post("/chat")
async def chat(
    body: ChatRequest,
    request: Request,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    """Answer a question about the user's documents, streamed as SSE.

    Events: `sources` (once, first) -> `token` (many) -> `done` | `error`.
    The strategy (RAG / agent / long-context) comes from ANSWER_MODE.
    """
    started = time.perf_counter()
    redis = request.app.state.redis

    retry_after = await check_rate_limit(redis, f"chat:{user.id}", settings.rate_limit_chat_per_minute)
    if retry_after is not None:
        RATE_LIMITED.labels("chat").inc()
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"Too many questions: limit is {settings.rate_limit_chat_per_minute} per minute.",
            headers={"Retry-After": str(retry_after)},
        )

    # --- Answer cache (only for questions without history: a follow-up's
    # meaning depends on the conversation).
    mode = body.answer_mode or settings.answer_mode
    cache_key = None
    if settings.answer_cache_ttl_seconds > 0 and not body.history:
        cache_key = await answer_cache_key(redis, user.id, body.question, mode, body.document_ids)
        cached = await get_cached_answer(redis, cache_key)
        ANSWER_CACHE.labels("hit" if cached else "miss").inc()
        if cached:
            await session.close()
            return _replay(cached, started)

    # --- Preparation: retrieval, or the whole agent loop. Done before streaming
    # starts, so failures return a clean HTTP error.
    try:
        prep = await prepare_answer(session, user.id, body.question, body.history, document_ids=body.document_ids,
                                     mode=mode)
    except EmbeddingError as e:
        CHAT_ERRORS.labels("prep").inc()
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e)) from e
    except ContextTooLarge as e:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, str(e)) from e
    except Exception as e:
        CHAT_ERRORS.labels("prep").inc()
        log.exception("answer preparation failed")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Retrieval failed: {e}") from e

    CHAT_STAGE.labels("prep", prep.mode).observe(prep.prep_ms / 1000)
    # Give the DB connection back to the pool before the (slow) LLM stream.
    # Phase 5 suspected the session dependency kept its connection checked out
    # for the whole stream, capping concurrent chats at the pool size. Measured:
    # it doesn't; FastAPI 0.163 closes yield-dependencies before the streamed
    # body runs. Closing explicitly keeps that guarantee even if the framework's
    # behavior changes (it has changed between FastAPI versions).
    await session.close()
    sources = [
        SourceOut(
            n=i,
            document_id=c.document_id,
            filename=c.filename,
            page=c.page,
            chunk_index=c.chunk_index,
            score=round(c.score, 4),
            snippet=c.content[:500],
        ).model_dump(mode="json")
        for i, c in enumerate(prep.sources, start=1)
    ]

    async def event_stream() -> AsyncIterator[str]:
        yield _sse("sources", sources)
        first_token_ms: float | None = None
        meta: dict = {}  # filled by stream_chat: model actually used + token usage
        answer_parts: list[str] = []

        if prep.answer is not None:  # agent: the answer was produced during preparation
            meta = {"model": prep.agent["model"], "usage": prep.agent["usage"]}
            first_token_ms = (time.perf_counter() - started) * 1000
            answer_parts.append(prep.answer)
            yield _sse("token", {"text": prep.answer})
        elif not prep.messages:
            # Don't spend an LLM call when there's nothing to ground the answer in.
            yield _sse("token", {"text": NO_SOURCES_MESSAGE})
        else:
            try:
                async for fragment in stream_chat(prep.messages, meta=meta):
                    if first_token_ms is None:
                        first_token_ms = (time.perf_counter() - started) * 1000
                    answer_parts.append(fragment)
                    yield _sse("token", {"text": fragment})
            except Exception as e:  # headers are already sent; report in-band
                CHAT_ERRORS.labels("llm").inc()
                log.exception("LLM call failed")
                yield _sse("error", {"message": f"LLM error: {e}"})
                return

        total_ms = (time.perf_counter() - started) * 1000
        if first_token_ms is not None:
            CHAT_STAGE.labels("ttft", prep.mode).observe(first_token_ms / 1000)
        CHAT_STAGE.labels("total", prep.mode).observe(total_ms / 1000)
        for kind in ("prompt_tokens", "completion_tokens"):
            if meta.get("usage"):
                LLM_TOKENS.labels(kind.split("_")[0], meta.get("model", "?")).inc(meta["usage"].get(kind) or 0)
        log.info(
            "chat user=%s mode=%s model=%s sources=%d query=%r prep=%.0fms first_token=%sms total=%.0fms usage=%s extra=%s",
            user.id, prep.mode, meta.get("model"), len(prep.sources), prep.search_query, prep.prep_ms,
            f"{first_token_ms:.0f}" if first_token_ms else "-", total_ms, meta.get("usage"), prep.extra_usage,
        )
        done = {
            "mode": prep.mode,
            "retrieval_ms": round(prep.prep_ms),
            "first_token_ms": round(first_token_ms) if first_token_ms else None,
            "total_ms": round(total_ms),
            "model": meta.get("model", "none"),
            "usage": meta.get("usage"),
            "search_query": prep.search_query,
            "agent_steps": prep.agent["steps"] if prep.agent else None,
            "cached": False,
        }
        if cache_key and prep.sources and answer_parts:
            try:
                await store_answer(redis, cache_key, {"sources": sources, "answer": "".join(answer_parts), "done": done})
            except Exception:
                log.warning("answer cache store failed", exc_info=True)
        yield _sse("done", done)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _replay(cached: dict, started: float) -> StreamingResponse:
    """Serve a cached answer with the same event sequence as a fresh one."""
    async def event_stream() -> AsyncIterator[str]:
        yield _sse("sources", cached["sources"])
        yield _sse("token", {"text": cached["answer"]})
        total_ms = (time.perf_counter() - started) * 1000
        CHAT_STAGE.labels("total", "cache").observe(total_ms / 1000)
        yield _sse("done", {**cached["done"], "cached": True, "retrieval_ms": 0,
                            "first_token_ms": round(total_ms), "total_ms": round(total_ms), "usage": None})

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

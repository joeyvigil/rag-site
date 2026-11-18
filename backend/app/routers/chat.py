import json
import logging
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import get_session
from app.embeddings import EmbeddingError
from app.llm import stream_chat
from app.models import User
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
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    """Answer a question about the user's documents, streamed as SSE.

    Events: `sources` (once, first) -> `token` (many) -> `done` | `error`.
    The strategy (RAG / agent / long-context) comes from ANSWER_MODE.
    """
    started = time.perf_counter()

    # --- Preparation: retrieval, or the whole agent loop. Done before streaming
    # starts, so failures return a clean HTTP error.
    try:
        prep = await prepare_answer(session, user.id, body.question, body.history, document_ids=body.document_ids,
                                     mode=body.answer_mode)
    except EmbeddingError as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e)) from e
    except ContextTooLarge as e:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, str(e)) from e
    except Exception as e:
        log.exception("answer preparation failed")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Retrieval failed: {e}") from e

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

        if prep.answer is not None:  # agent: the answer was produced during preparation
            meta = {"model": prep.agent["model"], "usage": prep.agent["usage"]}
            first_token_ms = (time.perf_counter() - started) * 1000
            yield _sse("token", {"text": prep.answer})
        elif not prep.messages:
            # Don't spend an LLM call when there's nothing to ground the answer in.
            yield _sse("token", {"text": NO_SOURCES_MESSAGE})
        else:
            try:
                async for fragment in stream_chat(prep.messages, meta=meta):
                    if first_token_ms is None:
                        first_token_ms = (time.perf_counter() - started) * 1000
                    yield _sse("token", {"text": fragment})
            except Exception as e:  # headers are already sent; report in-band
                log.exception("LLM call failed")
                yield _sse("error", {"message": f"LLM error: {e}"})
                return

        total_ms = (time.perf_counter() - started) * 1000
        log.info(
            "chat user=%s mode=%s model=%s sources=%d query=%r prep=%.0fms first_token=%sms total=%.0fms usage=%s extra=%s",
            user.id, prep.mode, meta.get("model"), len(prep.sources), prep.search_query, prep.prep_ms,
            f"{first_token_ms:.0f}" if first_token_ms else "-", total_ms, meta.get("usage"), prep.extra_usage,
        )
        yield _sse("done", {
            "mode": prep.mode,
            "retrieval_ms": round(prep.prep_ms),
            "first_token_ms": round(first_token_ms) if first_token_ms else None,
            "total_ms": round(total_ms),
            "model": meta.get("model", "none"),
            "usage": meta.get("usage"),
            "search_query": prep.search_query,
            "agent_steps": prep.agent["steps"] if prep.agent else None,
        })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

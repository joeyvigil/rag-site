import json
import logging
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import get_session
from app.embeddings import EmbeddingError, embed_query
from app.llm import stream_chat
from app.models import User
from app.rag import build_messages, retrieve
from app.schemas import ChatRequest, SourceOut
from app.security import get_current_user

log = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


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
    """
    started = time.perf_counter()

    # --- Retrieval (before streaming starts, so failures return a clean HTTP error)
    try:
        query_vec = await embed_query(session, body.question)
        chunks = await retrieve(
            session, user.id, query_vec, settings.retrieval_top_k, body.document_ids
        )
        await session.commit()  # persist the query embedding in the cache
    except EmbeddingError as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e)) from e
    except Exception as e:
        log.exception("retrieval failed")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Retrieval failed: {e}") from e
    retrieval_ms = (time.perf_counter() - started) * 1000

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
        for i, c in enumerate(chunks, start=1)
    ]
    messages = build_messages(body.question, body.history, chunks)

    async def event_stream() -> AsyncIterator[str]:
        yield _sse("sources", sources)
        first_token_ms: float | None = None

        if not chunks:
            # Don't spend an LLM call when there's nothing to ground the answer in.
            yield _sse("token", {"text": "I couldn't find anything relevant. Upload some documents and wait until they show as ready."})
        else:
            try:
                async for fragment in stream_chat(messages):
                    if first_token_ms is None:
                        first_token_ms = (time.perf_counter() - started) * 1000
                    yield _sse("token", {"text": fragment})
            except Exception as e:  # headers are already sent; report in-band
                log.exception("LLM call failed")
                yield _sse("error", {"message": f"LLM error: {e}"})
                return

        total_ms = (time.perf_counter() - started) * 1000
        log.info(
            "chat user=%s chunks=%d retrieval=%.0fms first_token=%sms total=%.0fms",
            user.id, len(chunks), retrieval_ms,
            f"{first_token_ms:.0f}" if first_token_ms else "-", total_ms,
        )
        yield _sse("done", {
            "retrieval_ms": round(retrieval_ms),
            "first_token_ms": round(first_token_ms) if first_token_ms else None,
            "total_ms": round(total_ms),
            "model": settings.llm_model if settings.llm_provider != "fake" else "fake",
        })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

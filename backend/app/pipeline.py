"""One entry point for every answering strategy, shared by the chat endpoint
and the evaluation harness, so the eval always measures the production code.

    prepare_answer(...) -> Prepared
        mode="rag":          [rewrite query] -> search -> prompt (streamed after)
        mode="long_context": all documents -> prompt (streamed after)
        mode="agent":        tool loop -> final answer (already computed)
"""

import time
import uuid
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.embeddings import embed_query
from app.rag import RetrievedChunk, build_messages, load_whole_documents, rewrite_query, search


@dataclass
class Prepared:
    mode: str
    sources: list[RetrievedChunk]  # what the answer may cite, numbered [1..n]
    messages: list[dict] | None = None  # rag / long_context: prompt to stream
    answer: str | None = None  # agent: final answer, already generated
    retrieved: list[RetrievedChunk] = field(default_factory=list)  # rag: full ranked list (for eval metrics)
    search_query: str | None = None  # rag: the query actually searched (after rewriting)
    prep_ms: float = 0.0  # time before generation starts (retrieval / agent loop)
    # LLM calls made during preparation: [{"stage", "model", "prompt_tokens", "completion_tokens"}]
    extra_usage: list[dict] = field(default_factory=list)
    agent: dict | None = None  # agent: steps, tool calls, usage


async def prepare_answer(
    session: AsyncSession,
    user_id: uuid.UUID,
    question: str,
    history: list,
    *,
    document_ids: list[uuid.UUID] | None = None,
    mode: str | None = None,
    k: int | None = None,
    retrieve_n: int | None = None,
    retrieval_mode: str | None = None,
    reranker: str | None = None,
    query_rewrite: bool | None = None,
) -> Prepared:
    mode = mode or settings.answer_mode
    k = k or settings.retrieval_top_k
    started = time.perf_counter()
    # End the auth query's transaction so the connection goes back to the pool
    # before any slow external call (query rewrite, embedding, agent LLM calls).
    await session.commit()

    if mode == "long_context":
        docs = await load_whole_documents(session, user_id, document_ids)
        return Prepared(mode, docs, messages=build_messages(question, history, docs),
                        prep_ms=(time.perf_counter() - started) * 1000)

    if mode == "agent":
        from app.agent import run_agent

        meta: dict = {}
        answer, sources = await run_agent(session, user_id, question, history, meta)
        return Prepared(mode, sources, answer=answer, agent=meta,
                        prep_ms=(time.perf_counter() - started) * 1000)

    if mode != "rag":
        raise ValueError(f"Unknown answer mode: {mode}")

    retrieval_mode = retrieval_mode or settings.retrieval_mode
    query_rewrite = settings.query_rewrite if query_rewrite is None else query_rewrite
    extra: list[dict] = []
    search_query = question
    if query_rewrite and history:
        rewritten = await rewrite_query(question, history)
        if rewritten.text.strip():
            search_query = rewritten.text.strip()
        extra.append({"stage": "rewrite", "model": rewritten.model, **rewritten.usage})

    qvec = await embed_query(session, search_query) if retrieval_mode != "keyword" else None
    search_meta: dict = {}
    retrieved = await search(
        session, user_id, search_query, qvec, max(k, retrieve_n or 0), document_ids,
        mode=retrieval_mode, reranker=reranker, meta=search_meta,
    )
    await session.commit()  # persist cached query embeddings
    if "rerank_usage" in search_meta:
        extra.append({"stage": "rerank", **search_meta["rerank_usage"]})

    context = retrieved[:k]
    return Prepared(
        mode, context, messages=build_messages(question, history, context) if context else None,
        retrieved=retrieved, search_query=search_query,
        prep_ms=(time.perf_counter() - started) * 1000, extra_usage=extra,
    )

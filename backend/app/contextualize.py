"""Contextual chunks: give each chunk the context it lost when it was cut out.

A chunk like "Overall status: Green. Route planning and telemetry are
complete" doesn't say WHICH project it's about. Its embedding can't match
"What is Osprey's status?" well, and keyword search can't match "Osprey"
at all. Two fixes, compared in the evaluation:

- "title": prefix a label with the filename (and page): free and deterministic.
- "llm": Anthropic's "contextual retrieval". An LLM reads the whole document
  plus the chunk and writes 1-2 sentences situating the chunk ("This chunk is
  from the Project Osprey Q3 2026 status report and describes its overall
  status."). That costs one LLM call per chunk at ingest time (never at query
  time), so results are cached in the llm_cache table.

The context is stored as part of the chunk's content, so it's embedded and
full-text indexed along with the chunk, and the answering LLM sees it too.
"""

import hashlib
import logging

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.chunking import ChunkData
from app.config import settings
from app.llm import complete
from app.models import LLMCache

log = logging.getLogger(__name__)

# The document goes FIRST and is identical for every chunk of that document,
# so providers with prefix caching can reuse it across calls.
CONTEXT_PROMPT = """<document name="{filename}">
{document}
</document>

Here is a chunk from the document above:
<chunk>
{chunk}
</chunk>

Write 1-2 short sentences that situate this chunk within the document, to improve search retrieval of the chunk. Name the document's subject and any entity the chunk is about but doesn't name itself (project, policy, product category, time period). Output only the sentences."""

MAX_DOC_CHARS = 30_000


async def contextualize(
    session: AsyncSession, filename: str, full_text: str, chunks: list[ChunkData], mode: str
) -> list[str]:
    """Return the text to store for each chunk (context + original chunk)."""
    if mode == "none":
        return [c.text for c in chunks]
    if mode == "title":
        return [f"[{filename}{f', page {c.page}' if c.page else ''}]\n{c.text}" for c in chunks]
    if mode != "llm":
        raise ValueError(f"Unknown chunk_context mode: {mode}")

    document = full_text[:MAX_DOC_CHARS]
    out = []
    for i, c in enumerate(chunks):
        prompt = CONTEXT_PROMPT.format(filename=filename, document=document, chunk=c.text)
        key = hashlib.sha256(f"{settings.llm_model}|{prompt}".encode()).hexdigest()
        cached = await session.get(LLMCache, key)
        if cached:
            context = cached.value
        else:
            result = await complete([{"role": "user", "content": prompt}])
            context = " ".join(result.text.split())
            await session.execute(pg_insert(LLMCache).values(key=key, value=context).on_conflict_do_nothing())
            log.info("%s: context %d/%d: %s", filename, i + 1, len(chunks), context[:100])
        out.append(f"{context}\n\n{c.text}")
    return out

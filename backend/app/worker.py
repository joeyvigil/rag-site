"""Background ingestion worker (arq + Redis).

Run with: arq app.worker.WorkerSettings

Uploads return immediately; the slow work (parse -> chunk -> embed -> index)
happens here. This keeps API latency flat under load, and you scale
ingestion independently by running more worker containers.
"""

import logging
import uuid

import anyio
from arq.connections import RedisSettings
from sqlalchemy import delete

from app.chunking import chunk_sections
from app.config import settings
from app.db import SessionLocal, ensure_vector_extension
from app.embeddings import embed_documents
from app.models import Chunk, Document, DocumentStatus
from app.ocr import ocr_document
from app.parsing import ParseError, Section, parse_document
from app.storage import storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("worker")


async def ingest_document(ctx: dict, document_id: str) -> str:
    doc_uuid = uuid.UUID(document_id)

    async with SessionLocal() as session:
        doc = await session.get(Document, doc_uuid)
        if doc is None:
            return "deleted"  # removed before we got to it
        doc.status = DocumentStatus.PROCESSING
        doc.error = None
        await session.commit()

        try:
            data = await storage.read(doc.storage_key)
            # Parsing is CPU-bound; a thread keeps the worker's event loop responsive.
            parsed = await anyio.to_thread.run_sync(parse_document, doc.filename, data)

            if parsed.ocr_pages:  # scanned pages / images: no text layer
                ocr_text = await ocr_document(doc.filename, data, parsed.ocr_pages)
                parsed.sections += [Section(t, page=p) for p, t in ocr_text.items() if t.strip()]
                parsed.sections.sort(key=lambda s: s.page or 0)
                log.info("%s: OCR'd %d page(s) via %s", doc.filename, len(ocr_text), settings.ocr_provider)
            if not parsed.sections:
                raise ParseError("No extractable text found, even after OCR")

            chunks = chunk_sections(
                parsed.sections, settings.chunk_size, settings.chunk_overlap,
                repeat_headers=settings.csv_header_in_chunks,
            )
            log.info("%s: %d sections -> %d chunks", doc.filename, len(parsed.sections), len(chunks))

            vectors = await embed_documents(session, [c.text for c in chunks])

            # Idempotent: replace any chunks from a previous attempt.
            await session.execute(delete(Chunk).where(Chunk.document_id == doc.id))
            session.add_all(
                Chunk(
                    document_id=doc.id,
                    user_id=doc.user_id,
                    chunk_index=c.index,
                    page=c.page,
                    content=c.text,
                    token_count=c.token_estimate,
                    embedding=vec,
                )
                for c, vec in zip(chunks, vectors, strict=True)
            )
            doc.status = DocumentStatus.READY
            doc.num_chunks = len(chunks)
            doc.num_pages = parsed.num_pages
            await session.commit()
            log.info("%s: ready", doc.filename)
            return "ready"

        except Exception as e:
            log.exception("ingestion failed for %s", document_id)
            await session.rollback()
            doc = await session.get(Document, doc_uuid)
            if doc is not None:
                doc.status = DocumentStatus.FAILED
                doc.error = str(e)[:1000] or e.__class__.__name__
                await session.commit()
            return "failed"


async def startup(ctx: dict) -> None:
    await ensure_vector_extension()


class WorkerSettings:
    functions = [ingest_document]
    on_startup = startup
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    max_jobs = 4  # concurrent ingestions per worker process
    job_timeout = 900  # seconds; big PDFs on a rate-limited free tier can be slow

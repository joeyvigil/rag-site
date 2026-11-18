import hashlib
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache import bump_corpus_version
from app.config import settings
from app.db import get_session
from app.models import Document, DocumentStatus, User
from app.parsing import SUPPORTED_EXTENSIONS
from app.schemas import DocumentOut
from app.security import get_current_user
from app.storage import storage

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=list[DocumentOut], status_code=status.HTTP_201_CREATED)
async def upload_documents(
    request: Request,
    files: list[UploadFile] = File(...),
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> list[Document]:
    """Store the files and queue them for ingestion. Returns immediately;
    parsing + embedding happen in the background worker."""
    max_bytes = settings.max_upload_mb * 1024 * 1024

    # Pass 1: validate everything before writing anything.
    incoming: list[tuple[str, str, bytes]] = []
    for f in files:
        name = Path(f.filename or "untitled").name
        ext = Path(name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                f"Unsupported file type: {name}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            )
        data = await f.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(
                413, f"{name} is larger than {settings.max_upload_mb} MB"
            )
        if not data:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, f"{name} is empty")
        incoming.append((name, f.content_type or "application/octet-stream", data))

    # Pass 2: store + create rows.
    results: list[Document] = []
    to_enqueue: list[uuid.UUID] = []
    for name, content_type, data in incoming:
        digest = hashlib.sha256(data).hexdigest()
        existing = await session.scalar(
            select(Document).where(
                Document.user_id == user.id,
                Document.sha256 == digest,
                Document.status != DocumentStatus.FAILED,
            )
        )
        if existing:  # same bytes already uploaded: don't pay to embed them again
            results.append(existing)
            continue

        doc_id = uuid.uuid4()
        doc = Document(
            id=doc_id,
            user_id=user.id,
            filename=name,
            content_type=content_type,
            size_bytes=len(data),
            sha256=digest,
            storage_key=f"{user.id}/{doc_id}{Path(name).suffix.lower()}",
            status=DocumentStatus.PENDING,
            num_chunks=0,
        )
        await storage.save(doc.storage_key, data)
        session.add(doc)
        results.append(doc)
        to_enqueue.append(doc_id)

    await session.commit()
    for doc_id in to_enqueue:
        await request.app.state.arq.enqueue_job("ingest_document", str(doc_id))
    return results


@router.get("", response_model=list[DocumentOut])
async def list_documents(
    user: User = Depends(get_current_user), session: AsyncSession = Depends(get_session)
) -> list[Document]:
    rows = await session.scalars(
        select(Document).where(Document.user_id == user.id).order_by(Document.created_at.desc())
    )
    return list(rows)


async def _get_owned(session: AsyncSession, user: User, document_id: uuid.UUID) -> Document:
    doc = await session.get(Document, document_id)
    # 404 (not 403) for other users' docs so IDs can't be probed.
    if doc is None or doc.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return doc


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(
    document_id: uuid.UUID,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> Document:
    return await _get_owned(session, user, document_id)


@router.post("/{document_id}/reprocess", response_model=DocumentOut)
async def reprocess_document(
    document_id: uuid.UUID,
    request: Request,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> Document:
    doc = await _get_owned(session, user, document_id)
    if doc.status in (DocumentStatus.PENDING, DocumentStatus.PROCESSING):
        raise HTTPException(status.HTTP_409_CONFLICT, "Document is already being processed")
    doc.status = DocumentStatus.PENDING
    doc.error = None
    await session.commit()
    await request.app.state.arq.enqueue_job("ingest_document", str(doc.id))
    return doc


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID,
    request: Request,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> Response:
    doc = await _get_owned(session, user, document_id)
    key = doc.storage_key
    await session.delete(doc)  # chunks go via ON DELETE CASCADE
    await session.commit()
    await bump_corpus_version(request.app.state.redis, user.id)  # cached answers may cite it
    await storage.delete(key)
    return Response(status_code=status.HTTP_204_NO_CONTENT)

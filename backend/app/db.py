from collections.abc import AsyncIterator

import asyncpg
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

# Arbitrary constant used as a Postgres advisory-lock id so that several
# processes starting at once (API replicas, workers) don't race on DDL.
_INIT_LOCK_ID = 7243901

engine = create_async_engine(
    settings.database_url,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session


def _raw_dsn() -> str:
    return settings.database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


async def ensure_vector_extension() -> None:
    """Create the pgvector extension (idempotent, safe with concurrent starts).

    Note: pgvector's SQLAlchemy `Vector` type sends vectors as text
    ('[0.1,0.2,...]'), so no asyncpg codec registration is needed.
    """
    conn = await asyncpg.connect(_raw_dsn())
    try:
        async with conn.transaction():
            await conn.execute(f"SELECT pg_advisory_xact_lock({_INIT_LOCK_ID})")
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    finally:
        await conn.close()


async def init_db() -> None:
    """Create tables. Phase 1 uses create_all; switch to Alembic migrations later."""
    from app import models  # noqa: F401  (registers tables on Base.metadata)

    await ensure_vector_extension()
    async with engine.begin() as conn:
        await conn.execute(text(f"SELECT pg_advisory_xact_lock({_INIT_LOCK_ID})"))
        await conn.run_sync(Base.metadata.create_all)
        # create_all only creates missing *tables*. Columns added after the
        # first run need explicit migration (a sign it's time for Alembic).
        await conn.execute(text(
            "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_tsv tsvector "
            f"GENERATED ALWAYS AS ({models.TSV_EXPRESSION}) STORED"
        ))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_chunks_content_tsv ON chunks USING gin (content_tsv)"
        ))
        await conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS full_text text"))

import asyncio
import logging
from contextlib import asynccontextmanager

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI

from app.config import settings
from app.db import engine, init_db
from app.routers import auth, chat, documents

from app.observability import (
    ObservabilityMiddleware,
    RequestIdLogFilter,
    instrument_pool,
    metrics_endpoint,
    monitor_event_loop,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s")
for _h in logging.getLogger().handlers:
    _h.addFilter(RequestIdLogFilter())
instrument_pool(engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    app.state.arq = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    app.state.redis = app.state.arq  # ArqRedis is a regular redis.asyncio client too
    lag_task = asyncio.create_task(monitor_event_loop())
    yield
    lag_task.cancel()
    await app.state.arq.aclose()
    await engine.dispose()


app = FastAPI(title="RAG Site", version="0.1.0", lifespan=lifespan)
app.add_middleware(ObservabilityMiddleware)
# Prometheus scrapes this. Not under /api and not authenticated: in production
# it would be reachable only from the internal network.
app.add_route("/metrics", metrics_endpoint, include_in_schema=False)

app.include_router(auth.router, prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(chat.router, prefix="/api")


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "llm": settings.llm_provider if settings.llm_provider == "fake" else settings.llm_model,
        "embeddings": settings.embedding_provider,
    }

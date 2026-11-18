"""Metrics (Prometheus) and request IDs.

Why Prometheus: it's the de-facto standard. The app exposes counters and
histograms at /metrics as plain text; a Prometheus server scrapes them every
few seconds and stores time series; Grafana draws them. The app stays cheap:
recording a metric is an in-memory increment.

Multiple processes: with `uvicorn --workers N`, each worker has its own
memory, so a scrape would only see one worker's numbers. prometheus_client's
"multiprocess mode" (enabled by setting PROMETHEUS_MULTIPROC_DIR) makes each
process write its metrics to files in that directory, and /metrics merges
them. Gauges then need a rule for merging: "livesum" = sum over live processes.

Labels must have LOW cardinality: use the route *template*
("/api/documents/{doc_id}"), never the raw path, or every document ID becomes
a new time series.
"""

import logging
import os
import re
import time
import uuid
from contextvars import ContextVar

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.requests import Request
from starlette.responses import Response

# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

# Latency buckets (seconds) spanning fast DB calls to slow LLM answers.
_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 13, 21, 34, 60)

HTTP_REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "route", "status"])
HTTP_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Time until the LAST byte of the response (includes the whole SSE stream)",
    ["method", "route"],
    buckets=_BUCKETS,
)
HTTP_IN_FLIGHT = Gauge("http_requests_in_flight", "Requests being processed", multiprocess_mode="livesum")

CHAT_STAGE = Histogram(
    "chat_stage_seconds",
    "Chat latency by stage: prep (retrieval or agent loop), ttft, total",
    ["stage", "mode"],
    buckets=_BUCKETS,
)
LLM_TOKENS = Counter("llm_tokens_total", "LLM tokens used by chat answers", ["kind", "model"])
CHAT_ERRORS = Counter("chat_errors_total", "Chat failures", ["stage"])
ANSWER_CACHE = Counter("answer_cache_total", "Answer cache lookups", ["result"])
RATE_LIMITED = Counter("rate_limited_total", "Requests rejected by the per-user rate limit", ["endpoint"])

DB_POOL_CHECKED_OUT = Gauge(
    "db_pool_checked_out", "DB connections currently checked out of the pool", multiprocess_mode="livesum"
)
DB_CONN_WAIT = Histogram(
    "db_connection_wait_seconds",
    "Time a request waited to get a DB connection from the pool",
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)


EVENT_LOOP_LAG = Gauge(
    "event_loop_lag_seconds",
    "How late a 0.25 s timer fires (max over the last 5 s), per process. The event loop runs "
    "one thing at a time, so lag = how long every waiting request is delayed by CPU work",
    multiprocess_mode="all",  # one series per worker process
)


async def monitor_event_loop(interval: float = 0.25, window: float = 5.0) -> None:
    """Background task: if a 0.25 s sleep takes 0.4 s, something hogged the loop for 0.15 s."""
    import asyncio

    worst, since = 0.0, time.perf_counter()
    while True:
        t = time.perf_counter()
        await asyncio.sleep(interval)
        worst = max(worst, time.perf_counter() - t - interval)
        if time.perf_counter() - since >= window:
            EVENT_LOOP_LAG.set(worst)
            worst, since = 0.0, time.perf_counter()


def instrument_pool(engine) -> None:
    """Track how many pooled connections are in use (the classic hidden bottleneck)."""
    from sqlalchemy import event

    @event.listens_for(engine.sync_engine, "checkout")
    def _checkout(*_):
        DB_POOL_CHECKED_OUT.inc()

    @event.listens_for(engine.sync_engine, "checkin")
    def _checkin(*_):
        DB_POOL_CHECKED_OUT.dec()


def metrics_response() -> Response:
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        from prometheus_client import multiprocess

        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        data = generate_latest(registry)
    else:
        from prometheus_client import REGISTRY

        data = generate_latest(REGISTRY)
    return Response(data, media_type=CONTENT_TYPE_LATEST)


# --------------------------------------------------------------------------
# Request IDs + timing middleware
# --------------------------------------------------------------------------

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
_VALID_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class RequestIdLogFilter(logging.Filter):
    """Adds `request_id` to every log record, so one request's lines can be grepped together."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class ObservabilityMiddleware:
    """Pure ASGI middleware (not BaseHTTPMiddleware) so it can see when a
    *streaming* response really ends, and doesn't buffer it."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] == "/metrics":
            return await self.app(scope, receive, send)

        incoming = dict(scope["headers"]).get(b"x-request-id", b"").decode(errors="ignore")
        rid = incoming if _VALID_ID.match(incoming) else uuid.uuid4().hex[:16]
        token = request_id_var.set(rid)
        started = time.perf_counter()
        status = 500
        HTTP_IN_FLIGHT.inc()

        async def send_wrapper(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message.setdefault("headers", [])
                message["headers"] = [*message["headers"], (b"x-request-id", rid.encode())]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            HTTP_IN_FLIGHT.dec()
            route = scope.get("route")
            route_label = getattr(route, "path", None) or "unmatched"
            method = scope["method"]
            HTTP_REQUESTS.labels(method, route_label, str(status)).inc()
            HTTP_LATENCY.labels(method, route_label).observe(time.perf_counter() - started)
            request_id_var.reset(token)


async def metrics_endpoint(_: Request) -> Response:
    return metrics_response()

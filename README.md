# Doc Q&A: a RAG learning project

Upload documents (PDF incl. scanned, DOCX, TXT, MD, CSV, JSON, HTML, images), then ask questions and get
streamed answers with citations. Built from scratch, without LangChain/LlamaIndex,
so every step of the pipeline is visible and explainable.

> 📘 **Learning guide:** [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) explains in detail
> how every part works, the trade-offs, and likely interview questions. It is updated
> at the end of each phase.

## Quick start

```bash
cp .env.example .env          # then paste your free GEMINI_API_KEY into .env
docker compose up --build
```

- App: http://localhost:5173
- API docs (Swagger): http://localhost:8000/docs

No key yet? Set `LLM_PROVIDER=fake` and `EMBEDDING_PROVIDER=fake` in `.env` to run
the whole pipeline offline.

Tests: `cd backend && uv venv -p 3.12 .venv && uv pip install -r requirements-dev.txt && .venv/bin/pytest`

## Architecture

```
 Browser (React/Vite)
    │  /api/*  (JWT bearer auth, SSE for streamed answers)
    ▼
 FastAPI ──────────────┬──────────────► Postgres + pgvector
    │ upload: save file│                 users, documents, chunks (+HNSW index),
    │ + enqueue job    │                 embedding_cache
    ▼                  │
 Redis (arq queue)     │
    │                  │
    ▼                  │
 Worker: parse → chunk → embed (Gemini, cached) → insert chunks
```

**Ingestion (async):** `POST /api/documents` validates the files, stores the originals,
creates `pending` rows and enqueues a job, then returns immediately. The worker then:

1. parses the file into text sections, keeping page numbers (`app/parsing.py`); scanned pages and images are OCR'd (`app/ocr.py`)
2. splits each section into ~1200-character chunks with 200 characters of overlap, cutting on paragraph, then sentence, then word boundaries; CSV chunks repeat the header row (`app/chunking.py`)
3. embeds the chunks in batches. Embeddings are cached by content hash, and rate-limit errors (429) are retried with backoff (`app/embeddings.py`)
4. writes the chunks and vectors and marks the document `ready` (`app/worker.py`)

**Query (streamed):** `POST /api/chat` does the following (`app/rag.py`, `app/routers/chat.py`):

1. embeds the question (with the `RETRIEVAL_QUERY` task type)
2. runs **hybrid search** over **only this user's** chunks: vector (HNSW cosine) + keyword (Postgres full-text), merged with Reciprocal Rank Fusion, with optional reranking (`RERANKER=cross_encoder|llm`)
3. builds a prompt with the numbered excerpts, plus the rules: answer only from them, cite as `[n]`, and treat the excerpts as data, not instructions
4. streams the answer back as SSE events (`sources`, then `token`s, then `done`), with timing info

## Code map

| File | What |
|---|---|
| `backend/app/config.py` | All settings (env vars) |
| `backend/app/models.py` | Tables: users, documents, chunks, embedding_cache |
| `backend/app/security.py` | bcrypt passwords, JWT, `get_current_user` |
| `backend/app/parsing.py` | File → text sections |
| `backend/app/chunking.py` | Recursive chunker |
| `backend/app/embeddings.py` | Gemini / fake embeddings + cache + retries |
| `backend/app/llm.py` | OpenAI-compatible streaming chat (Gemini, Groq, …) |
| `backend/app/rag.py` | Retrieval (vector, keyword, RRF), query rewriting, whole-document loading, prompt construction |
| `backend/app/pipeline.py` | `prepare_answer`: the three answer modes (rag / agent / long_context), shared by chat and eval |
| `backend/app/agent.py` | Agentic RAG: tool-calling loop with search / list / read tools |
| `backend/app/contextualize.py` | Contextual chunks (`none` / `title` / `llm`) at ingest |
| `backend/app/reranking.py` | Cross-encoder (local ONNX) and LLM rerankers |
| `backend/app/ocr.py` | OCR: Gemini vision or Tesseract |
| `backend/app/worker.py` | Ingestion job |
| `backend/app/observability.py` | Prometheus metrics, request IDs, event-loop lag, DB-pool gauges |
| `backend/app/cache.py` | Per-user rate limiting and the answer cache (Redis) |
| `loadtest/` | Load-test stack: Locust users, step runner, fake-provider config, Prometheus/Grafana |
| `backend/eval/` | Evaluation harness: corpus, golden dataset, metrics, LLM judge, results |
| `frontend/src/` | React UI (auth, documents, chat) |

## Evaluation

A fictional-company corpus (`backend/eval/corpus/`) and 57 hand-written questions
(`backend/eval/dataset.jsonl`) measure retrieval, answer quality, latency and cost.
Commands run inside the `api` container:

```bash
docker compose exec api python -m eval.run --validate-only                # check dataset vs corpus
docker compose exec api python -m eval.run --retrieval-only --label quick # retrieval metrics only (fast)
docker compose exec api python -m eval.run --label baseline               # + answers + LLM judge (~7 min)
docker compose exec api python -m eval.run --retrieval-only --chunk-size 600 --label chunk600
docker compose exec api python -m eval.run --retrieval-only --retrieval-mode vector --reranker cross_encoder --label ce
docker compose exec api python -m eval.run --answer-mode agent --concurrency 1 --rpm 10 --label agent
docker compose exec api python -m eval.run --retrieval-only --chunk-context llm --label ctx-llm
docker compose exec api python -m eval.compare                            # table of all runs
docker compose exec api python -m eval.ocr_eval                           # Gemini vs Tesseract OCR
```

Results are saved to `backend/eval/results/` with per-question detail. See
[docs/HOW_IT_WORKS.md §6](docs/HOW_IT_WORKS.md#6-evaluation) for what each metric means
and the results.

| | Correctness | hit@6 | $/1k queries |
|---|---|---|---|
| Phase 2 baseline (vector search) | 0.917 | 0.944 | $0.67 |
| **Phase 3 default (hybrid + CSV headers)** | **0.976** | **1.000** | $0.74 |
| Phase 3 + LLM reranker | 0.988 | 1.000 | $2.60 |

Phase 4 added 15 harder questions (follow-ups, cross-document synthesis, counting), 57 total:

| Answer mode | Correctness | p50 latency | $/1k queries |
|---|---|---|---|
| RAG, no query rewriting | 0.833 | 6.8 s | $0.81 |
| **RAG + query rewriting (default)** | **≈0.90** (follow-ups 3/7 → 7/7) | 6.8 s | ~$0.82 |
| Long context (no retrieval) | 0.939 | 6.8 s | $9.93 |
| Agent (tool-calling loop) | **0.974** | 16.4 s | $3.06 |

(The API was slow that day; compare latencies within the table, not with Phase 3. No mode
could reliably count rows in a 500-row CSV: that needs a SQL/code tool.)

## Load testing

An isolated stack (its own database) with **fake** LLM and embeddings that wait like real
ones (0.8 s to first token, 30 tokens/s), so load tests cost nothing and measure our code:

```bash
docker compose -f loadtest/docker-compose.yml up -d --build
API_WORKERS=2 docker compose -f loadtest/docker-compose.yml up -d --force-recreate api
API_WORKERS=2 docker compose -f loadtest/docker-compose.yml run --rm steps --label mytest --users 150,300,450
docker compose -f loadtest/docker-compose.yml --profile monitoring up -d   # Prometheus :9090, Grafana :3001
docker compose -f loadtest/docker-compose.yml down -v                      # remove it all
```

(Pass the same env vars to `run` as to `up`: `compose run` recreates the API if they differ.)
Interactive Locust UI: http://localhost:8089. Results: `loadtest/results/`.

| 1 API container (2 CPUs of a laptop) | chats/s | total p95 at 750 users |
|---|---|---|
| Before: DB connection held across external calls, 1 worker | ~59 (pool exhausted, 1.2 s DB wait) | – |
| 1 worker, connection fix | ~55 (CPU-bound) | 17 s |
| **2 workers (default for load tests)** | **~72** | **6.9 s** |
| 2 workers + answer cache, repeated questions | ~89 (p50 0.77 s) | – |

Details and the debugging story: [docs/HOW_IT_WORKS.md §7](docs/HOW_IT_WORKS.md#7-scaling-what-breaks-first-and-how-to-fix-it).

## Design decisions worth explaining in an interview

- **Background ingestion queue:** parsing and embedding are slow and rate-limited. Doing them inside the HTTP request would tie up API capacity and time out on large files. Worker capacity scales separately from the API.
- **pgvector instead of a dedicated vector DB:** one database holds metadata, vectors and (soon) full-text search, so tenant filters, joins and deletes are ordinary SQL in one transaction. A dedicated vector DB is worth it at a much larger scale.
- **HNSW + filtered search:** approximate nearest-neighbour search applies the `WHERE user_id = …` filter after finding neighbours, which can return fewer than k rows. `hnsw.iterative_scan` (pgvector ≥ 0.8) fixes that.
- **Asymmetric embeddings:** documents and queries are embedded with different task types.
- **Embedding cache:** content-addressed (`sha256(provider|model|dim|task|text)`), so re-uploads and repeated questions cost nothing.
- **Tenant isolation:** `user_id` is copied onto every chunk and every query filters on it. Other users' documents return 404 rather than 403, so document IDs can't be probed.
- **Hybrid search + RRF:** embeddings miss exact IDs; keywords miss paraphrases. Rank fusion merges them without comparing incomparable scores.
- **Answer modes:** RAG, agent and long-context share one pipeline, so they're compared on identical questions; the user picks per question.
- **No DB connection held across slow calls:** sessions end their transaction before LLM/embedding calls, so the pool isn't drained by connections that are just waiting.
- **Answer cache with version-based invalidation:** per-user corpus version in the key; no cache scans.
- **Measure before shipping:** the reranker doubled rank-1 accuracy but barely moved answer correctness, at 3.5× the cost, so it's off by default.
- **Prompt injection:** document text is untrusted. Excerpts are wrapped in tags and the system prompt tells the model to ignore instructions inside them.
- **Provider-agnostic LLM layer:** any OpenAI-compatible endpoint works. That makes the planned model comparison a config change.
- **Event loop hygiene:** bcrypt and PDF parsing run in threads so they don't block the async server.

## Known limitations (intentional; later phases fix them)

- Counting/aggregation questions over big tables fail in every mode (needs a text-to-SQL tool).
- Agent mode is the most accurate but slow (p95 ~55 s), so it's opt-in per question.
- Chunks never cross page boundaries, so very short pages produce tiny chunks.
- Keyword ranking is `ts_rank_cd`, not true BM25 (no IDF).
- OCR was evaluated on synthetic scans only.
- Tables are created with `create_all`; switch to Alembic migrations before the schema changes much.
- Changing `EMBEDDING_DIM` requires wiping the DB (`docker compose down -v`).

## Roadmap

1. ✅ Phase 1: naive RAG MVP
2. ✅ Phase 2: golden Q&A dataset + evaluation harness (hit@k, recall@k, MRR, correctness, faithfulness, latency, cost)
3. ✅ Phase 3: hybrid search (Postgres FTS + vectors, RRF), rerankers (cross-encoder, LLM), CSV-aware chunking, OCR (Gemini vision / Tesseract), all measured
4. ✅ Phase 4: harder benchmark, query rewriting, contextual chunks, agentic RAG, long-context baseline, all measured
5. ✅ Phase 5: observability (Prometheus, request IDs, Grafana), load testing with fake providers, connection-pool fix, multiple workers, answer cache, rate limiting

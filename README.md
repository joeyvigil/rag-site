# Doc Q&A: a RAG learning project

Upload documents (PDF, DOCX, TXT, MD, CSV, JSON, HTML), then ask questions and get
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

1. parses the file into text sections, keeping page numbers (`app/parsing.py`)
2. splits each section into ~1200-character chunks with 200 characters of overlap, cutting on paragraph, then sentence, then word boundaries (`app/chunking.py`)
3. embeds the chunks in batches. Embeddings are cached by content hash, and rate-limit errors (429) are retried with backoff (`app/embeddings.py`)
4. writes the chunks and vectors and marks the document `ready` (`app/worker.py`)

**Query (streamed):** `POST /api/chat` does the following (`app/rag.py`, `app/routers/chat.py`):

1. embeds the question (with the `RETRIEVAL_QUERY` task type)
2. runs an HNSW cosine search over **only this user's** chunks and returns the top k
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
| `backend/app/rag.py` | Retrieval query + prompt construction |
| `backend/app/worker.py` | Ingestion job |
| `frontend/src/` | React UI (auth, documents, chat) |

## Design decisions worth explaining in an interview

- **Background ingestion queue:** parsing and embedding are slow and rate-limited. Doing them inside the HTTP request would tie up API capacity and time out on large files. Worker capacity scales separately from the API.
- **pgvector instead of a dedicated vector DB:** one database holds metadata, vectors and (soon) full-text search, so tenant filters, joins and deletes are ordinary SQL in one transaction. A dedicated vector DB is worth it at a much larger scale.
- **HNSW + filtered search:** approximate nearest-neighbour search applies the `WHERE user_id = …` filter after finding neighbours, which can return fewer than k rows. `hnsw.iterative_scan` (pgvector ≥ 0.8) fixes that.
- **Asymmetric embeddings:** documents and queries are embedded with different task types.
- **Embedding cache:** content-addressed (`sha256(provider|model|dim|task|text)`), so re-uploads and repeated questions cost nothing.
- **Tenant isolation:** `user_id` is copied onto every chunk and every query filters on it. Other users' documents return 404 rather than 403, so document IDs can't be probed.
- **Prompt injection:** document text is untrusted. Excerpts are wrapped in tags and the system prompt tells the model to ignore instructions inside them.
- **Provider-agnostic LLM layer:** any OpenAI-compatible endpoint works. That makes the planned model comparison a config change.
- **Event loop hygiene:** bcrypt and PDF parsing run in threads so they don't block the async server.

## Known Phase 1 limitations (intentional; later phases fix them)

- Naive retrieval: one vector search on the raw question. No keyword/hybrid search, no reranking.
- Follow-up questions ("what about the second one?") retrieve poorly, because there is no query rewriting yet.
- Chunks never cross page boundaries, so very short pages produce tiny chunks.
- Scanned PDFs (images only) fail, since there's no OCR.
- Tables are created with `create_all`; switch to Alembic migrations before the schema changes much.
- Changing `EMBEDDING_DIM` requires wiping the DB (`docker compose down -v`).

## Roadmap

1. ✅ Phase 1: naive RAG MVP (this)
2. Golden Q&A dataset + evaluation script (recall@k, MRR, faithfulness, latency, cost)
3. Hybrid search (Postgres full-text search + vectors, Reciprocal Rank Fusion) → reranking → better chunking
   → OCR for scanned PDFs/images (Gemini vision on pages with no text layer; Tesseract as a local baseline)
4. Query rewriting, contextual retrieval, agentic RAG, long-context baseline
5. Load testing (Locust/k6 with the fake LLM), caching, rate limits, observability

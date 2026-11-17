"""Run the RAG evaluation.

Run inside the api container (it has the DB, Redis and API keys):

  docker compose exec api python -m eval.run --label baseline
  docker compose exec api python -m eval.run --retrieval-only --label quick-check
  docker compose exec api python -m eval.run --retrieval-only --chunk-size 600 --label chunk600
  docker compose exec api python -m eval.run --validate-only

How it works:
1. Ingest the corpus for a dedicated eval user, using the same parse -> chunk
   -> embed code as production. There's one eval user per configuration
   (chunk size, embedding model, ...), so switching configs back and forth
   doesn't re-embed. Re-runs reuse it.
2. For each question: retrieve the top 10 chunks and score retrieval against
   the evidence strings. Then (unless --retrieval-only) generate an answer
   from the top-k chunks, and have an LLM judge grade it.
3. Print a summary and save everything to eval/results/<timestamp>_<label>.json.
   Compare runs with: python -m eval.compare
"""

import argparse
import asyncio
import hashlib
import json
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import func, select

from app.config import settings
from app.db import SessionLocal, init_db
from app.embeddings import embed_query
from app.llm import stream_chat
from app.models import Chunk, Document, DocumentStatus, User
from app.parsing import parse_document
from app.rag import build_messages, search
from app.storage import storage
from eval import metrics
from eval.build_corpus import CORPUS_DIR
from eval.judge import Judge
from eval.pricing import cost_usd

EVAL_DIR = Path(__file__).parent
DATASET_PATH = EVAL_DIR / "dataset.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
RETRIEVE_N = 10  # always retrieve this many, so hit@1/3/5/10 can all be computed


# ---------------------------------------------------------------- dataset & corpus

def load_dataset() -> list[dict]:
    return [json.loads(line) for line in DATASET_PATH.read_text().splitlines() if line.strip()]


def corpus_files() -> list[Path]:
    files = sorted(p for p in CORPUS_DIR.iterdir() if p.is_file() and not p.name.startswith("."))
    if not files:
        raise SystemExit("Corpus is empty. Run: python -m eval.build_corpus")
    return files


def parsed_corpus() -> dict[str, str]:
    """filename -> full extracted text (what the system actually 'sees')."""
    return {
        f.name: "\n".join(s.text for s in parse_document(f.name, f.read_bytes()).sections)
        for f in corpus_files()
    }


def validate_dataset(questions: list[dict], corpus: dict[str, str]) -> list[str]:
    """Every evidence string must appear verbatim (modulo whitespace/case) in one document."""
    problems = []
    norm_docs = {name: metrics.normalize(text) for name, text in corpus.items()}
    seen_ids = set()
    for q in questions:
        if q["id"] in seen_ids:
            problems.append(f"{q['id']}: duplicate id")
        seen_ids.add(q["id"])
        if q["category"] == "unanswerable" and q["evidence"]:
            problems.append(f"{q['id']}: unanswerable questions must have no evidence")
        if q["category"] != "unanswerable" and not q["evidence"]:
            problems.append(f"{q['id']}: answerable question has no evidence")
        for ev in q["evidence"]:
            if not any(metrics.normalize(ev) in d for d in norm_docs.values()):
                problems.append(f"{q['id']}: evidence not found in corpus: {ev!r}")
    return problems


# ---------------------------------------------------------------- ingestion

def ingest_config(corpus: dict[str, str]) -> dict:
    fingerprint = hashlib.sha256(
        json.dumps(sorted(corpus.items())).encode()
    ).hexdigest()[:12]
    return {
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "csv_header_in_chunks": settings.csv_header_in_chunks,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "embedding_dim": settings.embedding_dim,
        "corpus": fingerprint,
    }


async def ensure_ingested(config: dict) -> tuple[uuid.UUID, int]:
    """Return (eval user id, chunk count), ingesting the corpus if needed."""
    from app.worker import ingest_document  # imported here: sets up worker logging

    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    email = f"eval-{config_hash}@eval.local"
    files = corpus_files()

    async with SessionLocal() as s:
        user = await s.scalar(select(User).where(User.email == email))
        if user is None:
            user = User(email=email, password_hash="!disabled")  # can't log in
            s.add(user)
            await s.commit()

        docs = list(await s.scalars(select(Document).where(Document.user_id == user.id)))
        ready = {d.filename for d in docs if d.status == DocumentStatus.READY}
        if ready != {f.name for f in files} or len(docs) != len(files):
            print(f"Ingesting {len(files)} corpus files for config {config_hash} …")
            for d in docs:
                await storage.delete(d.storage_key)
                await s.delete(d)
            await s.commit()

            new_ids = []
            for f in files:
                data = f.read_bytes()
                doc_id = uuid.uuid4()
                doc = Document(
                    id=doc_id, user_id=user.id, filename=f.name,
                    content_type="application/octet-stream", size_bytes=len(data),
                    sha256=hashlib.sha256(data).hexdigest(),
                    storage_key=f"{user.id}/{doc_id}{f.suffix.lower()}",
                    status=DocumentStatus.PENDING, num_chunks=0,
                )
                await storage.save(doc.storage_key, data)
                s.add(doc)
                new_ids.append(doc_id)
            await s.commit()

            # Same function the background worker runs, called directly.
            for doc_id in new_ids:
                if await ingest_document({}, str(doc_id)) != "ready":
                    raise SystemExit(f"Ingestion failed for document {doc_id}; see logs above.")

        n_chunks = await s.scalar(select(func.count()).select_from(Chunk).where(Chunk.user_id == user.id))
    return user.id, int(n_chunks or 0)


# ---------------------------------------------------------------- per-question

class RateLimiter:
    """Spaces out API calls to stay under a requests-per-minute limit."""

    def __init__(self, rpm: float) -> None:
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next = 0.0

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if self._next > now:
                await asyncio.sleep(self._next - now)
            self._next = max(now, self._next) + self.interval


async def run_question(q: dict, user_id: uuid.UUID, args, judge: Judge | None, limiter: RateLimiter) -> dict:
    rec: dict = {k: q[k] for k in ("id", "category", "question", "reference", "evidence")}

    # --- Retrieval
    if args.reranker == "llm":
        await limiter.wait()  # the LLM reranker is an API call too
    t0 = time.perf_counter()
    search_meta: dict = {}
    async with SessionLocal() as s:
        qvec = await embed_query(s, q["question"]) if args.retrieval_mode != "keyword" else None
        chunks = await search(
            s, user_id, q["question"], qvec, max(RETRIEVE_N, args.top_k),
            mode=args.retrieval_mode, reranker=args.reranker, meta=search_meta,
        )
        await s.commit()
    retrieval_ms = (time.perf_counter() - t0) * 1000
    context = chunks[: args.top_k]

    rec["retrieval_ms"] = round(retrieval_ms, 1)
    if "rerank_usage" in search_meta:
        ru = search_meta["rerank_usage"]
        rec["rerank_usage"] = {**ru, "cost_usd": cost_usd(ru["model"], ru["prompt_tokens"], ru["completion_tokens"])}
    rec["retrieved"] = [
        {"filename": c.filename, "page": c.page, "chunk_index": c.chunk_index, "score": round(c.score, 4)}
        for c in chunks
    ]
    if q["evidence"]:
        ranks = metrics.evidence_ranks(q["evidence"], [c.content for c in chunks])
        rec["retrieval"] = {
            "ranks": ranks,
            "hit@1": metrics.hit_at_k(ranks, 1),
            "hit@3": metrics.hit_at_k(ranks, 3),
            "hit@5": metrics.hit_at_k(ranks, 5),
            "hit@10": metrics.hit_at_k(ranks, 10),
            "hit@k": metrics.hit_at_k(ranks, args.top_k),
            "recall@k": metrics.recall_at_k(ranks, args.top_k),
            "rr": metrics.reciprocal_rank(ranks),
        }
    if args.retrieval_only:
        return rec

    # --- Generation (same prompt as production)
    messages = build_messages(q["question"], [], context)
    answer, meta, gen_error = "", {}, None
    ttft_ms = total_ms = None
    if not context:
        answer = "I couldn't find anything relevant."
    else:
        for attempt in range(4):
            meta, parts, first = {}, [], None
            await limiter.wait()
            start = time.perf_counter()
            try:
                async for fragment in stream_chat(messages, meta=meta, models=[args.model]):
                    if first is None:
                        first = time.perf_counter()
                    parts.append(fragment)
                end = time.perf_counter()
                answer = "".join(parts)
                # End-to-end, as a user would experience it: retrieval + generation.
                ttft_ms = retrieval_ms + ((first or end) - start) * 1000
                total_ms = retrieval_ms + (end - start) * 1000
                gen_error = None
                break
            except Exception as e:  # overloaded / rate limited: back off and retry the SAME model
                gen_error = f"{type(e).__name__}: {e}"[:300]
                await asyncio.sleep(5 * (attempt + 1) ** 2)
    if gen_error:
        rec["error"] = gen_error
        return rec

    usage = meta.get("usage") or {"prompt_tokens": 0, "completion_tokens": 0}
    model_used = meta.get("model", args.model)
    rec["generation"] = {
        "answer": answer,
        "model": model_used,
        "usage": usage,
        "cost_usd": cost_usd(model_used, usage["prompt_tokens"], usage["completion_tokens"]),
        "ttft_ms": round(ttft_ms, 1) if ttft_ms is not None else None,
        "total_ms": round(total_ms, 1) if total_ms is not None else None,
        **metrics.citation_check(answer, len(context)),
    }

    # --- Judge
    if judge:
        for attempt in range(3):
            await limiter.wait()
            try:
                rec["generation"].update(
                    await judge.grade(
                        q["question"], q["reference"], [c.content for c in context], answer,
                        answerable=bool(q["evidence"]),
                    )
                )
                break
            except Exception as e:
                rec["judge_error"] = f"{type(e).__name__}: {e}"[:300]
                await asyncio.sleep(5 * (attempt + 1) ** 2)
        else:
            return rec
        rec.pop("judge_error", None)
    return rec


# ---------------------------------------------------------------- summary

def _r(x: float | None, digits: int = 3) -> float | None:
    return round(x, digits) if x is not None else None


def summarize(records: list[dict]) -> dict:
    answerable = [r for r in records if r["evidence"]]
    retrieval_recs = [r["retrieval"] for r in answerable if "retrieval" in r]
    summary: dict = {
        "n_questions": len(records),
        "n_answerable": len(answerable),
        "errors": sum(1 for r in records if "error" in r or "judge_error" in r),
        "retrieval": {},
        "generation": {},
        "by_category": {},
    }

    for key in ("hit@1", "hit@3", "hit@5", "hit@10", "hit@k", "recall@k"):
        summary["retrieval"][key] = _r(metrics.mean([x[key] for x in retrieval_recs]))
    summary["retrieval"]["mrr"] = _r(metrics.mean([x["rr"] for x in retrieval_recs]))
    summary["retrieval"]["recall@k_ci95"] = metrics.bootstrap_ci([x["recall@k"] for x in retrieval_recs])
    summary["retrieval"]["retrieval_ms_p50"] = _r(metrics.percentile([r["retrieval_ms"] for r in records], 50), 1)
    summary["retrieval"]["retrieval_ms_p95"] = _r(metrics.percentile([r["retrieval_ms"] for r in records], 95), 1)

    gens = [r for r in records if "generation" in r]
    judged = [r for r in gens if "correctness_score" in r["generation"]]
    if gens:
        g = summary["generation"]
        g["n_answered"] = len(gens)
        g["n_judged"] = len(judged)
        if judged:
            scores = [r["generation"]["correctness_score"] for r in judged]
            g["correctness"] = _r(metrics.mean(scores))
            g["correctness_ci95"] = metrics.bootstrap_ci(scores)
            g["faithfulness"] = _r(metrics.mean([float(r["generation"]["faithful"]) for r in judged]))
            ans_j = [r for r in judged if r["evidence"]]
            unans_j = [r for r in judged if not r["evidence"]]
            g["correctness_answerable"] = _r(metrics.mean([r["generation"]["correctness_score"] for r in ans_j]))
            g["false_refusal_rate"] = _r(metrics.mean([float(r["generation"]["refused"]) for r in ans_j]))
            g["correct_refusal_rate"] = _r(metrics.mean([r["generation"]["correctness_score"] for r in unans_j]))
        answered_ans = [r for r in gens if r["evidence"] and not r["generation"].get("refused")]
        g["citation_rate"] = _r(metrics.mean([float(r["generation"]["has_citation"]) for r in answered_ans]))
        g["invalid_citations"] = sum(len(r["generation"]["invalid_citations"]) for r in gens)
        ttft = [r["generation"]["ttft_ms"] for r in gens if r["generation"]["ttft_ms"] is not None]
        total = [r["generation"]["total_ms"] for r in gens if r["generation"]["total_ms"] is not None]
        g["ttft_ms_p50"] = _r(metrics.percentile(ttft, 50), 0)
        g["ttft_ms_p95"] = _r(metrics.percentile(ttft, 95), 0)
        g["total_ms_p50"] = _r(metrics.percentile(total, 50), 0)
        g["total_ms_p95"] = _r(metrics.percentile(total, 95), 0)
        g["prompt_tokens_mean"] = _r(metrics.mean([r["generation"]["usage"]["prompt_tokens"] for r in gens]), 0)
        g["completion_tokens_mean"] = _r(metrics.mean([r["generation"]["usage"]["completion_tokens"] for r in gens]), 0)
        # Total cost per query = generation + LLM reranking (if used).
        costs = [r["generation"]["cost_usd"] for r in gens]
        rerank_costs = [r.get("rerank_usage", {}).get("cost_usd", 0.0) for r in gens]
        if all(c is not None for c in costs + rerank_costs):
            g["cost_per_1k_queries_usd"] = _r(metrics.mean([a + b for a, b in zip(costs, rerank_costs)]) * 1000, 4)
        reranks = [r["rerank_usage"] for r in gens if "rerank_usage" in r]
        if reranks:
            g["rerank_prompt_tokens_mean"] = _r(metrics.mean([x["prompt_tokens"] for x in reranks]), 0)
        g["models_used"] = sorted({r["generation"]["model"] for r in gens})

    for cat in sorted({r["category"] for r in records}):
        recs = [r for r in records if r["category"] == cat]
        row: dict = {"n": len(recs)}
        rr = [r["retrieval"] for r in recs if "retrieval" in r]
        if rr:
            row["hit@k"] = _r(metrics.mean([x["hit@k"] for x in rr]))
            row["recall@k"] = _r(metrics.mean([x["recall@k"] for x in rr]))
        cj = [r["generation"]["correctness_score"] for r in recs if "correctness_score" in r.get("generation", {})]
        if cj:
            row["correctness"] = _r(metrics.mean(cj))
        summary["by_category"][cat] = row
    return summary


def print_summary(summary: dict, config: dict) -> None:
    def fmt(v):
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.3f}"
        if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, float) for x in v):
            return f"[{v[0]:.2f}, {v[1]:.2f}]"
        return str(v)

    print("\n" + "=" * 64)
    print(f"Config: {json.dumps(config)}")
    print(f"Questions: {summary['n_questions']} ({summary['n_answerable']} answerable), errors: {summary['errors']}")
    print("\nRetrieval (answerable questions)")
    for k, v in summary["retrieval"].items():
        print(f"  {k:24} {fmt(v)}")
    if summary["generation"]:
        print("\nGeneration")
        for k, v in summary["generation"].items():
            print(f"  {k:24} {fmt(v)}")
    print("\nBy category")
    cols = ["n", "hit@k", "recall@k", "correctness"]
    print(f"  {'category':14}" + "".join(f"{c:>13}" for c in cols))
    for cat, row in summary["by_category"].items():
        print(f"  {cat:14}" + "".join(f"{fmt(row.get(c)):>13}" for c in cols))
    print("=" * 64)


# ---------------------------------------------------------------- main

async def main(args) -> None:
    # CLI overrides apply to the shared settings object, so production code
    # (chunking, retrieval) picks them up unchanged.
    if args.chunk_size:
        settings.chunk_size = args.chunk_size
    if args.chunk_overlap is not None:
        settings.chunk_overlap = args.chunk_overlap
    if args.csv_header is not None:
        settings.csv_header_in_chunks = args.csv_header == "on"
    settings.rerank_candidates = args.rerank_candidates

    questions = load_dataset()
    corpus = parsed_corpus()
    problems = validate_dataset(questions, corpus)
    if problems:
        print("Dataset problems:\n  " + "\n  ".join(problems))
        raise SystemExit(1)
    print(f"Dataset OK: {len(questions)} questions, all evidence found in {len(corpus)} corpus files.")
    if args.validate_only:
        return

    if args.ids:
        wanted = set(args.ids.split(","))
        questions = [q for q in questions if q["id"] in wanted]
    if args.category:
        questions = [q for q in questions if q["category"] == args.category]
    if args.limit:
        questions = questions[: args.limit]

    await init_db()
    config = ingest_config(corpus)
    user_id, n_chunks = await ensure_ingested(config)
    print(f"Corpus indexed: {n_chunks} chunks.")

    use_judge = not args.retrieval_only and not args.no_judge and settings.llm_provider != "fake"
    judge = Judge(args.judge_model) if use_judge else None
    limiter = RateLimiter(args.rpm)
    sem = asyncio.Semaphore(args.concurrency)
    done = 0

    async def one(q: dict) -> dict:
        nonlocal done
        async with sem:
            rec = await run_question(q, user_id, args, judge, limiter)
        done += 1
        bits = [f"[{done:>2}/{len(questions)}] {q['id']} {q['category']:12}"]
        if "retrieval" in rec:
            bits.append(f"hit@k={rec['retrieval']['hit@k']:.0f} rr={rec['retrieval']['rr']:.2f}")
        gen = rec.get("generation", {})
        if "correctness" in gen:
            bits.append(f"{gen['correctness']:9} faithful={gen['faithful']}")
        if "error" in rec or "judge_error" in rec:
            bits.append(f"ERROR {rec.get('error') or rec.get('judge_error')}")
        print("  ".join(bits), flush=True)
        return rec

    started = time.perf_counter()
    records = await asyncio.gather(*(one(q) for q in questions))
    records.sort(key=lambda r: r["id"])

    run_config = {
        **config,
        "top_k": args.top_k,
        "retrieval_mode": args.retrieval_mode,
        "reranker": args.reranker,
        "rerank_candidates": args.rerank_candidates if args.reranker != "none" else None,
        "model": None if args.retrieval_only else args.model,
        "judge_model": judge.model if judge else None,
        "retrieval_only": args.retrieval_only,
        "concurrency": args.concurrency,
        "rpm": args.rpm,
        "n_chunks": n_chunks,
    }
    summary = summarize(records)
    summary["wall_time_s"] = round(time.perf_counter() - started, 1)
    print_summary(summary, run_config)

    if not args.no_save:
        RESULTS_DIR.mkdir(exist_ok=True)
        ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        path = RESULTS_DIR / f"{ts}_{args.label}.json"
        path.write_text(json.dumps(
            {"label": args.label, "timestamp": ts, "config": run_config, "summary": summary, "questions": records},
            indent=2, default=str,
        ))
        _match_owner(RESULTS_DIR, path)
        print(f"Saved {path.relative_to(EVAL_DIR.parent)}")


def _match_owner(*paths: Path) -> None:
    """The container runs as root; give files written into the bind-mounted
    source tree back to whoever owns the project, so they're editable on the host."""
    if os.getuid() != 0:
        return
    st = EVAL_DIR.stat()
    for p in paths:
        os.chown(p, st.st_uid, st.st_gid)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", default="run", help="name for this run (used in the results filename)")
    p.add_argument("--retrieval-only", action="store_true", help="skip generation and judging (fast, ~free)")
    p.add_argument("--no-judge", action="store_true", help="generate answers but don't grade them")
    p.add_argument("--validate-only", action="store_true", help="only check the dataset against the corpus")
    p.add_argument("--top-k", type=int, default=settings.retrieval_top_k, help="chunks given to the LLM")
    p.add_argument("--chunk-size", type=int, help="override CHUNK_SIZE (triggers re-ingestion)")
    p.add_argument("--chunk-overlap", type=int, help="override CHUNK_OVERLAP (triggers re-ingestion)")
    p.add_argument("--retrieval-mode", choices=["vector", "keyword", "hybrid"], default=settings.retrieval_mode)
    p.add_argument("--reranker", choices=["none", "cross_encoder", "llm"], default=settings.reranker)
    p.add_argument("--rerank-candidates", type=int, default=settings.rerank_candidates,
                   help="first-stage results the reranker reorders")
    p.add_argument("--csv-header", choices=["on", "off"], help="repeat CSV header in every chunk (re-ingests)")
    p.add_argument("--model", default=settings.llm_model, help="answer model (no fallbacks during eval)")
    p.add_argument("--judge-model", help="judge model (default: JUDGE_MODEL env or the answer model)")
    p.add_argument("--ids", help="comma-separated question ids, e.g. q01,q22")
    p.add_argument("--category", help="only run one category")
    p.add_argument("--limit", type=int, help="only run the first N questions")
    p.add_argument("--concurrency", type=int, default=2, help="questions in flight at once")
    p.add_argument("--rpm", type=float, default=12, help="max LLM requests per minute (answer + judge)")
    p.add_argument("--no-save", action="store_true", help="don't write a results file")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))

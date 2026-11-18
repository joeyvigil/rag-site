"""Compare saved evaluation runs side by side.

  docker compose exec api python -m eval.compare            # all runs
  docker compose exec api python -m eval.compare baseline   # runs whose label contains "baseline"
"""

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

COLUMNS = [
    ("run", lambda r: f"{r['timestamp']} {r['label']}"),
    ("model", lambda r: r["config"].get("model") or "(retrieval)"),
    ("chunk", lambda r: f"{r['config']['chunk_size']}/{r['config']['chunk_overlap']}"
                        + ("+hdr" if r["config"].get("csv_header_in_chunks") else "")),
    ("mode", lambda r: r["config"].get("answer_mode", "rag")
                       + ("+rw" if r["config"].get("query_rewrite") else "")
                       + (f"+ctx:{r['config']['chunk_context']}" if r["config"].get("chunk_context", "none") != "none" else "")),
    ("retrieval", lambda r: r["config"].get("retrieval_mode", "vector")
                            + (f"+{r['config']['reranker']}" if r["config"].get("reranker", "none") != "none" else "")),
    ("n", lambda r: r["summary"]["n_questions"]),
    ("k", lambda r: r["config"]["top_k"]),
    ("hit@k", lambda r: r["summary"]["retrieval"].get("hit@k")),
    ("recall@k", lambda r: r["summary"]["retrieval"].get("recall@k")),
    ("hit@1", lambda r: r["summary"]["retrieval"].get("hit@1")),
    ("MRR", lambda r: r["summary"]["retrieval"].get("mrr")),
    ("ret p50", lambda r: r["summary"]["retrieval"].get("retrieval_ms_p50")),
    ("correct", lambda r: r["summary"]["generation"].get("correctness")),
    ("faithful", lambda r: r["summary"]["generation"].get("faithfulness")),
    ("p50 ms", lambda r: r["summary"]["generation"].get("total_ms_p50")),
    ("p95 ms", lambda r: r["summary"]["generation"].get("total_ms_p95")),
    ("tok in", lambda r: r["summary"]["generation"].get("prompt_tokens_mean")),
    ("$/1k q", lambda r: r["summary"]["generation"].get("cost_per_1k_queries_usd")),
]


def fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.0f}" if v >= 100 else f"{v:.3f}"
    return str(v)


def main() -> None:
    filt = sys.argv[1] if len(sys.argv) > 1 else ""
    runs = [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("*.json"))]
    runs = [r for r in runs if filt in r["label"]]
    if not runs:
        print("No results yet. Run: python -m eval.run --label baseline")
        return
    print("| " + " | ".join(name for name, _ in COLUMNS) + " |")
    print("|" + "|".join("---" for _ in COLUMNS) + "|")
    for r in runs:
        print("| " + " | ".join(fmt(get(r)) for _, get in COLUMNS) + " |")


if __name__ == "__main__":
    main()

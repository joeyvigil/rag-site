"""Step-load runner: run Locust headless at increasing user counts and record
client-side latency (Locust) next to server-side causes (/metrics).

    docker compose -f loadtest/docker-compose.yml run --rm steps --label baseline
    docker compose -f loadtest/docker-compose.yml run --rm steps --label x --users 25,50 --duration 60

Writes loadtest/results/<label>.json and prints a markdown table.

Why steps instead of one big run: the interesting thing is the *knee*, the
load at which latency stops being flat and starts climbing (a queue is
forming somewhere). One number at one load level can't show that.
"""

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
import re
import subprocess
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
API = "http://api:8000"


def scrape() -> dict[str, float]:
    """Parse the Prometheus text format into {series: value} (enough for our needs)."""
    text = urllib.request.urlopen(f"{API}/metrics", timeout=5).read().decode()
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            name, _, value = line.rpartition(" ")
            try:
                out[name] = float(value)
            except ValueError:
                pass
    return out


def total(metrics: dict[str, float], prefix: str) -> float:
    """Sum every series of a metric (all label combinations / processes)."""
    pat = re.compile(rf"^{re.escape(prefix)}(\{{.*\}})?$")
    return sum(v for k, v in metrics.items() if pat.match(k))


class Sampler(threading.Thread):
    """Polls /metrics every second during a step to catch peaks of gauges."""

    def __init__(self):
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.peak_pool = 0.0
        self.peak_in_flight = 0.0
        self.peak_loop_lag = 0.0

    def run(self):
        while not self.stop.wait(1):
            try:
                m = scrape()
            except Exception:
                continue
            self.peak_pool = max(self.peak_pool, total(m, "db_pool_checked_out"))
            self.peak_in_flight = max(self.peak_in_flight, total(m, "http_requests_in_flight"))
            self.peak_loop_lag = max([self.peak_loop_lag] + [v for k, v in m.items() if k.startswith("event_loop_lag_seconds")])


ACCOUNTS = HERE / "results" / "raw" / "accounts.json"


def seed_accounts(n: int) -> None:
    """Create n accounts, each with one ingested document, BEFORE measuring.

    Reused across steps and runs (same database). Registration is slow on
    purpose (bcrypt), and ingestion is background work; neither belongs in a
    measurement of steady-state chat traffic.
    """
    import requests

    sys.path.insert(0, str(HERE))
    from docgen import make_document

    tokens: list[str] = json.loads(ACCOUNTS.read_text()) if ACCOUNTS.exists() else []
    if tokens:  # still valid for this database? (it may have been wiped with `down -v`)
        r = requests.get(f"{API}/api/documents", headers={"Authorization": f"Bearer {tokens[0]}"}, timeout=10)
        if r.status_code != 200:
            tokens = []
    missing = n - len(tokens)
    if missing <= 0:
        return
    print(f"seeding {missing} accounts...", flush=True)

    def create(_):
        email = f"seed-{os.urandom(6).hex()}@example.com"
        r = requests.post(f"{API}/api/auth/register", json={"email": email, "password": "loadtest-pass-1"}, timeout=60)
        r.raise_for_status()
        token = r.json()["access_token"]
        files = {"files": ("policy.md", make_document().encode(), "text/markdown")}
        requests.post(f"{API}/api/documents", files=files, headers={"Authorization": f"Bearer {token}"},
                      timeout=60).raise_for_status()
        return token

    with ThreadPoolExecutor(8) as pool:
        new = list(pool.map(create, range(missing)))
    deadline = time.time() + 300
    for token in new:  # wait for ingestion
        while time.time() < deadline:
            docs = requests.get(f"{API}/api/documents", headers={"Authorization": f"Bearer {token}"}, timeout=10).json()
            if docs and docs[0]["status"] in ("ready", "failed"):
                break
            time.sleep(0.5)
    ACCOUNTS.parent.mkdir(parents=True, exist_ok=True)
    ACCOUNTS.write_text(json.dumps(tokens + new))


def read_stats(prefix: Path) -> dict[str, dict]:
    rows = {}
    with open(f"{prefix}_stats.csv") as f:
        for r in csv.DictReader(f):
            rows[r["Name"]] = r
    return rows


def read_failures(prefix: Path) -> list[dict]:
    try:
        with open(f"{prefix}_failures.csv") as f:
            return [
                {"name": r["Name"], "error": r["Error"][:200], "count": int(r["Occurrences"])}
                for r in csv.DictReader(f)
            ]
    except FileNotFoundError:
        return []


def run_step(label: str, users: int, duration: int) -> dict:
    raw = HERE / "results" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    prefix = raw / f"{label}_u{users}"

    before = scrape()
    sampler = Sampler()
    sampler.start()
    spawn_rate = max(1, users // 10)  # ~10 s ramp-up
    subprocess.run(
        env={**os.environ, "TOKENS_FILE": str(ACCOUNTS)},
        args=[
            "locust", "-f", str(HERE / "locustfile.py"), "--headless", "--host", API,
            "-u", str(users), "-r", str(spawn_rate), "-t", f"{duration}s",
            # Drop stats from the ~10 s ramp-up once all users are running.
            "--reset-stats", "--csv", str(prefix), "--only-summary", "--loglevel", "WARNING",
        ],
        check=False,
    )
    sampler.stop.set()
    after = scrape()

    def delta(name: str) -> float:
        return total(after, name) - total(before, name)

    stats = read_stats(prefix)

    def pick(name: str) -> dict:
        r = stats.get(name)
        if not r or int(r["Request Count"]) == 0:
            return {"n": 0}
        return {
            "n": int(r["Request Count"]),
            "fail": int(r["Failure Count"]),
            "rps": round(float(r["Requests/s"]), 2),
            "p50": float(r["50%"]),
            "p95": float(r["95%"]),
            "p99": float(r["99%"]),
        }

    waits = delta("db_connection_wait_seconds_count")
    return {
        "users": users,
        "chat_headers": pick("POST /api/chat [headers]"),
        "chat_ttft": pick("chat TTFT"),
        "chat_total": pick("chat total"),
        "list_documents": pick("GET /api/documents"),
        "aggregated": pick("Aggregated"),
        "server": {
            "peak_db_pool_checked_out": sampler.peak_pool,
            "peak_requests_in_flight": sampler.peak_in_flight,
            "peak_event_loop_lag_ms": round(1000 * sampler.peak_loop_lag),
            "db_wait_mean_ms": round(1000 * delta("db_connection_wait_seconds_sum") / waits, 1) if waits else None,
            "answer_cache_hits": after.get('answer_cache_total{result="hit"}', 0) - before.get('answer_cache_total{result="hit"}', 0),
            "answer_cache_lookups": delta("answer_cache_total"),
            "rate_limited": delta("rate_limited_total"),
        },
        "failures": read_failures(prefix),
    }


def fmt(step: dict) -> str:
    c, t, h, s = step["chat_total"], step["chat_ttft"], step["chat_headers"], step["server"]
    if not c.get("n"):
        return f"| {step['users']} | no completed chats | | | | | | | |"
    err = 100 * c["fail"] / c["n"]
    return (
        f"| {step['users']} | {c['rps']:.1f} | {h.get('p50', 0):.0f} / {h.get('p95', 0):.0f} "
        f"| {t.get('p50', 0):.0f} / {t.get('p95', 0):.0f} | {c['p50']:.0f} / {c['p95']:.0f} "
        f"| {err:.1f}% | {s['peak_db_pool_checked_out']:.0f} | {s['db_wait_mean_ms']} | {s['peak_requests_in_flight']:.0f} |"
    )


HEADER = (
    "| users | chats/s | prep p50/p95 ms | TTFT p50/p95 ms | total p50/p95 ms | errors "
    "| peak DB conns | DB wait mean ms | peak in-flight |\n|---|---|---|---|---|---|---|---|---|"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--users", default="10,25,50,100,150")
    ap.add_argument("--duration", type=int, default=60, help="seconds per step")
    ap.add_argument("--note", default="", help="what's different about this run")
    args = ap.parse_args()

    health = json.loads(urllib.request.urlopen(f"{API}/api/health", timeout=5).read())
    assert health.get("llm") == "fake", f"Refusing to load-test a real LLM (quota/cost!): {health}"

    levels = [int(u) for u in args.users.split(",")]
    seed_accounts(max(levels))

    steps = []
    print(HEADER)
    for users in levels:
        step = run_step(args.label, users, args.duration)
        steps.append(step)
        print(fmt(step), flush=True)
        time.sleep(5)  # let in-flight requests drain between steps

    out = {
        "label": args.label,
        "note": args.note,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "duration_s": args.duration,
        "steps": steps,
    }
    path = HERE / "results" / f"{datetime.now():%Y%m%d-%H%M%S}_{args.label}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\nsaved {path.name}")


if __name__ == "__main__":
    main()

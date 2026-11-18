"""Locust load test: simulated users who upload a document and chat about it.

Each simulated user:
  on_start: registers (unique email), uploads one generated policy document,
            waits until ingestion finishes ("ready").
  then, with 2-6 s of "think time" between actions:
    ask a question (weight 6), ask a follow-up with history (2), list documents (2).

Locust times a streamed request only until the response *headers* arrive, so
we report three rows for chat:
  "POST /api/chat [headers]"  = server-side preparation (auth + retrieval)
  "chat TTFT"                 = time to the first token: what the user feels
  "chat total"                = until the `done` event: the whole answer

QUESTION_POOL=unique (default) makes every question text unique, so an
answer cache can't flatter the results; QUESTION_POOL=fixed reuses 8
questions, which is the best case for caching.
"""

import itertools
import json
import os
import random
import time
import uuid

from docgen import make_document
from locust import HttpUser, between, events, task

QUESTIONS = [
    "How many PTO days do full-time employees get?",
    "What is the remote work stipend?",
    "Who approves expense reports over the limit?",
    "What is the laptop refresh cycle?",
    "How long is the parental leave?",
    "What is the on-call compensation?",
    "Which VPN client is required?",
    "When are performance reviews held?",
]
FOLLOW_UPS = ["And for part-time employees?", "Who do I ask about that?", "Has that changed recently?"]
UNIQUE = os.environ.get("QUESTION_POOL", "unique") == "unique"

# steps.py pre-creates accounts (with an ingested document each) and lists
# their tokens here, so registration (bcrypt is deliberately slow) and
# ingestion don't pollute the measurement. Without the file, users set
# themselves up (fine for the interactive UI).
_TOKENS = json.load(open(os.environ["TOKENS_FILE"])) if os.environ.get("TOKENS_FILE") else []
_next_token = itertools.count()


def _report(environment, name: str, ms: float, exc: Exception | None = None, length: int = 0) -> None:
    environment.events.request.fire(
        request_type="SSE", name=name, response_time=ms, response_length=length, exception=exc, context={}
    )


class ChatUser(HttpUser):
    wait_time = between(2, 6)

    def on_start(self) -> None:
        self.history: list[dict] = []
        if _TOKENS:
            self.client.headers["Authorization"] = f"Bearer {_TOKENS[next(_next_token) % len(_TOKENS)]}"
            return
        email = f"load-{uuid.uuid4().hex[:12]}@example.com"
        resp = self.client.post("/api/auth/register", json={"email": email, "password": "loadtest-pass-1"},
                                name="setup: register")
        resp.raise_for_status()
        self.client.headers["Authorization"] = f"Bearer {resp.json()['access_token']}"

        files = {"files": ("policy.md", make_document().encode(), "text/markdown")}
        self.client.post("/api/documents", files=files, name="setup: upload").raise_for_status()

        deadline = time.time() + 120
        while time.time() < deadline:
            docs = self.client.get("/api/documents", name="setup: poll status").json()
            if docs and all(d["status"] in ("ready", "failed") for d in docs):
                break
            time.sleep(1)
        self.history: list[dict] = []

    def _chat(self, question: str, history: list[dict]) -> str | None:
        if UNIQUE:
            question = f"{question} (ref {random.randint(0, 10**9)})"
        started = time.perf_counter()
        ttft = None
        answer = []
        error: Exception | None = None
        try:
            with self.client.post(
                "/api/chat",
                json={"question": question, "history": history},
                stream=True,
                name="POST /api/chat [headers]",
                catch_response=True,
            ) as resp:
                if resp.status_code != 200:
                    resp.failure(f"HTTP {resp.status_code}")
                    error = RuntimeError(f"HTTP {resp.status_code}")
                else:
                    resp.success()
                    event = None
                    done = False
                    for raw in resp.iter_lines(decode_unicode=True):
                        if raw.startswith("event: "):
                            event = raw[7:]
                        elif raw.startswith("data: "):
                            if event == "token":
                                if ttft is None:
                                    ttft = (time.perf_counter() - started) * 1000
                                answer.append(json.loads(raw[6:])["text"])
                            elif event == "error":
                                error = RuntimeError("SSE error event: " + raw[6:200])
                            elif event == "done":
                                done = True
                    if not done and error is None:
                        error = RuntimeError("stream ended without a done event")
        except Exception as e:  # connection reset, timeout...
            error = e

        total = (time.perf_counter() - started) * 1000
        if ttft is not None:
            _report(self.environment, "chat TTFT", ttft)
        _report(self.environment, "chat total", total, error, sum(map(len, answer)))
        return None if error else "".join(answer)

    @task(6)
    def ask(self) -> None:
        q = random.choice(QUESTIONS)
        answer = self._chat(q, [])
        if answer is not None:
            self.history = [{"role": "user", "content": q}, {"role": "assistant", "content": answer}]

    @task(2)
    def follow_up(self) -> None:
        if not self.history:
            return self.ask()
        self._chat(random.choice(FOLLOW_UPS), self.history)

    @task(2)
    def list_documents(self) -> None:
        self.client.get("/api/documents", name="GET /api/documents")


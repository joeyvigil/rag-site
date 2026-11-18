import uuid

from app.agent import _Sources
from app.parsing import Section
from app.rag import RetrievedChunk
from app.worker import document_text


def _chunk(doc: uuid.UUID, cid: int) -> RetrievedChunk:
    return RetrievedChunk(cid, doc, "f.txt", None, cid, f"text {cid}", 0.0)


def test_document_text_has_page_markers_and_csv_header():
    sections = [
        Section("first page", page=1),
        Section("second page", page=2),
        Section("A-1,10\nB-2,20", header="sku,price"),
    ]
    assert document_text(sections) == "[Page 1]\nfirst page\n\n[Page 2]\nsecond page\n\nsku,price\nA-1,10\nB-2,20"


def test_agent_sources_number_each_chunk_once():
    doc = uuid.uuid4()
    s = _Sources()
    assert s.add(_chunk(doc, 7)) == 1
    assert s.add(_chunk(doc, 9)) == 2
    assert s.add(_chunk(doc, 7)) == 1  # seen again in a later search: same citation number
    assert [c.chunk_id for c in s.items] == [7, 9]


def test_daily_quota_detection():
    from app.llm import is_daily_quota_error

    daily = Exception("Error code: 429 - quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    per_minute = Exception("Error code: 429 - quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier")
    assert is_daily_quota_error(daily)
    assert not is_daily_quota_error(per_minute)  # per-minute limits are worth retrying


def test_daily_quota_detected_through_wrapper_exceptions():
    from app.llm import LLMError, is_daily_quota_error

    class FakeSDKError(Exception):
        body = [{"error": {"details": [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}}]

    try:
        try:
            raise FakeSDKError("Error code: 429")  # str() alone lacks the quota details
        except FakeSDKError as inner:
            raise LLMError("All models failed. Last error: Error code: 429") from inner
    except LLMError as wrapped:
        assert is_daily_quota_error(wrapped)


def test_agent_falls_back_when_a_model_is_overloaded(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from app import agent
    from app.config import settings
    from app.llm import LLMError

    calls = []

    class FakeCompletions:
        def __init__(self, down):
            self.down = down

        async def create(self, model, **kwargs):
            calls.append(model)
            if model in self.down:
                raise RuntimeError("Error code: 503 - high demand")
            msg = SimpleNamespace(content="answer", tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)

    class FakeSession:
        async def commit(self):
            pass

    monkeypatch.setattr(settings, "llm_model", "primary")
    monkeypatch.setattr(settings, "llm_fallback_models", "backup")

    down = {"primary"}
    monkeypatch.setattr(agent, "_get_client", lambda: SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(down))))
    meta: dict = {}
    answer, _ = asyncio.run(agent.run_agent(FakeSession(), uuid.uuid4(), "q?", [], meta))
    assert answer == "answer" and calls == ["primary", "backup"] and meta["model"] == "backup"

    down.update({"backup"})  # everything overloaded -> a clear LLMError (the API turns it into a 503)
    try:
        asyncio.run(agent.run_agent(FakeSession(), uuid.uuid4(), "q?", [], {}))
        raise AssertionError("expected LLMError")
    except LLMError:
        pass

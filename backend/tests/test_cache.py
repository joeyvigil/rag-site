import asyncio
import uuid

from app.cache import answer_cache_key, bump_corpus_version, check_rate_limit, normalize_question


class FakeRedis:
    """Just enough of redis.asyncio for app.cache."""

    def __init__(self):
        self.data: dict[str, int | bytes] = {}

    async def incr(self, key):
        self.data[key] = int(self.data.get(key, 0)) + 1
        return self.data[key]

    async def get(self, key):
        v = self.data.get(key)
        return str(v).encode() if isinstance(v, int) else v

    def pipeline(self, transaction=True):
        redis = self

        class Pipe:
            def __init__(self):
                self.ops = []

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            def incr(self, key):
                self.ops.append(("incr", key))

            def expire(self, key, seconds):
                self.ops.append(("expire", key))

            async def execute(self):
                return [await redis.incr(k) if op == "incr" else True for op, k in self.ops]

        return Pipe()


def test_rate_limit_allows_then_blocks():
    async def go():
        r = FakeRedis()
        results = [await check_rate_limit(r, "chat:u1", limit=3) for _ in range(5)]
        assert results[:3] == [None, None, None]
        assert all(isinstance(x, int) and 1 <= x <= 60 for x in results[3:])  # Retry-After seconds
        assert await check_rate_limit(r, "chat:u2", limit=3) is None  # other users unaffected
        assert await check_rate_limit(r, "chat:u1", limit=0) is None  # 0 = disabled

    asyncio.run(go())


def test_normalize_question():
    assert normalize_question("  What is  the PTO policy?? ") == "what is the pto policy"


def test_cache_key_changes_when_corpus_changes():
    async def go():
        r = FakeRedis()
        user = uuid.uuid4()
        k1 = await answer_cache_key(r, user, "What is PTO?", "rag", None)
        assert k1 == await answer_cache_key(r, user, "what is pto", "rag", None)  # normalized
        assert k1 != await answer_cache_key(r, user, "What is PTO?", "agent", None)  # mode matters
        assert k1 != await answer_cache_key(r, uuid.uuid4(), "What is PTO?", "rag", None)  # per user
        await bump_corpus_version(r, user)  # e.g. a document finished ingesting
        assert k1 != await answer_cache_key(r, user, "What is PTO?", "rag", None)

    asyncio.run(go())

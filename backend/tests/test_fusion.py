import uuid

from app.rag import RetrievedChunk, rrf_fuse


def _chunk(cid: int, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(cid, uuid.uuid4(), "f.txt", None, cid, f"chunk {cid}", score)


def test_rrf_rewards_agreement_between_retrievers():
    vector = [_chunk(1), _chunk(2), _chunk(3)]
    keyword = [_chunk(3), _chunk(4), _chunk(1)]
    fused = rrf_fuse([vector, keyword], k=60)
    ids = [c.chunk_id for c in fused]
    # 1 (ranks 1 and 3) and 3 (ranks 3 and 1) appear in both lists -> top two.
    assert set(ids[:2]) == {1, 3}
    assert set(ids) == {1, 2, 3, 4}  # union, no duplicates


def test_rrf_ignores_raw_scores():
    # Wildly different score scales must not matter; only ranks do.
    a = [_chunk(1, score=0.9), _chunk(2, score=0.8)]
    b = [_chunk(2, score=500.0), _chunk(1, score=0.001)]
    fused = rrf_fuse([a, b], k=60)
    assert abs(fused[0].score - fused[1].score) < 1e-12  # perfectly symmetric -> tie


def test_rrf_score_formula():
    fused = rrf_fuse([[_chunk(7)]], k=60)
    assert fused[0].score == 1 / 61

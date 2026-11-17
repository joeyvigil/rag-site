from eval import metrics


def test_evidence_ranks_normalize_whitespace_and_case():
    chunks = ["intro text", "Full-time employees accrue 22 days\nof paid time off", "other"]
    assert metrics.evidence_ranks(["accrue 22 days of PAID time off", "missing"], chunks) == [2, None]


def test_hit_recall_rr():
    ranks = [2, 7]  # two evidence strings, found at ranks 2 and 7
    assert metrics.hit_at_k(ranks, 1) == 0.0
    assert metrics.hit_at_k(ranks, 3) == 1.0
    assert metrics.recall_at_k(ranks, 3) == 0.5
    assert metrics.recall_at_k(ranks, 10) == 1.0
    assert metrics.reciprocal_rank(ranks) == 0.5
    assert metrics.reciprocal_rank([None]) == 0.0


def test_percentile():
    assert metrics.percentile([1, 2, 3, 4, 5], 50) == 3
    assert metrics.percentile([10], 95) == 10
    assert metrics.percentile([], 50) is None
    assert metrics.percentile([0, 100], 95) == 95


def test_bootstrap_ci_brackets_mean():
    values = [1.0] * 30 + [0.0] * 10
    lo, hi = metrics.bootstrap_ci(values)
    assert lo < 0.75 < hi
    assert metrics.bootstrap_ci([1.0]) is None


def test_citation_check():
    assert metrics.citation_check("A [1] and B [2][5].", num_sources=3) == {
        "has_citation": True,
        "invalid_citations": [5],
    }
    assert metrics.citation_check("No citations.", 3)["has_citation"] is False

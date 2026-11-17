"""Pure metric functions (no I/O), so they're easy to unit test.

Relevance is judged by *evidence strings*, not chunk IDs: a retrieved chunk
counts as relevant if it contains one of the question's evidence strings.
That keeps the dataset valid when chunking changes (Phase 3 will change it).
"""

import random
import re
import statistics


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def evidence_ranks(evidence: list[str], chunk_texts: list[str]) -> list[int | None]:
    """For each evidence string, the 1-based rank of the first chunk containing it (or None)."""
    norm_chunks = [normalize(c) for c in chunk_texts]
    ranks: list[int | None] = []
    for ev in evidence:
        needle = normalize(ev)
        ranks.append(next((i + 1 for i, c in enumerate(norm_chunks) if needle in c), None))
    return ranks


def hit_at_k(ranks: list[int | None], k: int) -> float:
    """1 if any evidence was retrieved in the top k, else 0."""
    return 1.0 if any(r is not None and r <= k for r in ranks) else 0.0


def recall_at_k(ranks: list[int | None], k: int) -> float:
    """Fraction of evidence strings retrieved in the top k (matters for multi-hop)."""
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def reciprocal_rank(ranks: list[int | None]) -> float:
    """1 / rank of the first relevant chunk (0 if none). Averaged over questions = MRR."""
    found = [r for r in ranks if r is not None]
    return 1.0 / min(found) if found else 0.0


def percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolated percentile, p in [0, 100]."""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * p / 100
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def bootstrap_ci(values: list[float], iterations: int = 2000, seed: int = 0) -> tuple[float, float] | None:
    """95% confidence interval for the mean, by resampling with replacement.

    With ~40 questions, intervals are wide (often +/- 10 points). If two runs'
    intervals overlap heavily, the difference may just be noise.
    """
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.fmean(rng.choices(values, k=n)) for _ in range(iterations))
    return means[int(0.025 * iterations)], means[int(0.975 * iterations) - 1]


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


CITATION_RE = re.compile(r"\[(\d+)\]")


def citation_check(answer: str, num_sources: int) -> dict:
    """Programmatic citation sanity checks (cheap; complements the LLM judge)."""
    cited = [int(n) for n in CITATION_RE.findall(answer)]
    return {
        "has_citation": bool(cited),
        "invalid_citations": sorted({n for n in cited if n < 1 or n > num_sources}),
    }

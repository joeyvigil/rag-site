"""Paid-tier prices, in USD per 1M tokens: (input, output).

We run on the free tier ($0), but cost per query at paid prices is what you'd
compare in a real system. Verify current prices before quoting them:
https://ai.google.dev/gemini-api/docs/pricing
"""

PRICES: dict[str, tuple[float, float]] = {
    "fake": (0.0, 0.0),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    # Third-party sources disagreed on 3.6 Flash (1.50/7.50 vs 0.75/3.75); unverified.
    "gemini-3.6-flash": (1.50, 7.50),
}


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    if model not in PRICES:
        return None
    price_in, price_out = PRICES[model]
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000

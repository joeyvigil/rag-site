"""Chat model client.

Gemini, Groq, OpenRouter, OpenAI, Together, etc. all expose the OpenAI
chat-completions API, so one client covers them all. Switching provider is
just LLM_BASE_URL + LLM_MODEL + LLM_API_KEY. That's what makes the later
"compare models on accuracy / speed / cost" phase cheap to do.
"""

import asyncio
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass

from openai import AsyncOpenAI

from app.config import settings

log = logging.getLogger(__name__)

class LLMError(RuntimeError):
    pass


class QuotaExhausted(LLMError):
    """A DAILY quota is used up. Retrying is pointless until it resets
    (free tier: midnight Pacific), so callers should stop, not back off."""


def is_daily_quota_error(e: BaseException | None) -> bool:
    """True if `e` (or any exception it was raised from) is a DAILY quota 429.

    Gemini's 429 body names the violated quota, e.g.
    "GenerateRequestsPerDayPerProjectPerModel-FreeTier". Per-minute limits
    say "PerMinute" and ARE worth retrying. The details live in the original
    SDK exception's body, so we follow the __cause__/__context__ chain: a
    wrapper like LLMError("All models failed") has lost them otherwise.
    """
    seen = 0
    while e is not None and seen < 10:
        if isinstance(e, QuotaExhausted) or "PerDay" in f"{e} {getattr(e, 'body', None)}":
            return True
        e = e.__cause__ or e.__context__
        seen += 1
    return False


_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        if not settings.effective_llm_api_key:
            raise LLMError("No LLM API key. Set GEMINI_API_KEY (or LLM_API_KEY) in .env, or LLM_PROVIDER=fake.")
        _client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.effective_llm_api_key,
            # Fail fast and fall back to another model instead of retrying
            # an overloaded one for minutes.
            max_retries=1,
            timeout=settings.llm_timeout_seconds,
        )
    return _client


@dataclass
class Completion:
    text: str
    model: str
    usage: dict  # {"prompt_tokens": int, "completion_tokens": int}


async def complete(
    messages: list[dict],
    *,
    model: str | None = None,
    json_mode: bool = False,
    temperature: float = 0.0,
    attempts: int = 5,
) -> Completion:
    """Non-streaming call for helper tasks (query rewriting, chunk context).

    Retries the same model with exponential backoff: helper calls are often
    made in bursts (one per chunk at ingest), which is exactly when free-tier
    rate limits (HTTP 429) hit.
    """
    if settings.llm_provider == "fake":
        await asyncio.sleep(settings.fake_llm_ttft_ms / 1000)
        text = messages[-1]["content"].strip().splitlines()[-1][:500]  # echo: a plausible "rewrite"
        return Completion(text, "fake", {"prompt_tokens": sum(len(m["content"]) for m in messages) // 4,
                                         "completion_tokens": len(text) // 4})
    model = model or settings.llm_model
    kwargs: dict = {"model": model, "messages": messages, "temperature": temperature}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    for attempt in range(attempts):
        try:
            resp = await _get_client().chat.completions.create(**kwargs)
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
            }
            return Completion(resp.choices[0].message.content or "", model, usage)
        except Exception as e:
            if is_daily_quota_error(e):
                raise QuotaExhausted(f"Daily quota exhausted for {model}") from e
            if attempt == attempts - 1:
                raise
            delay = min(60, 2 ** (attempt + 1))
            log.warning("LLM call failed (%s); retrying in %ss", e, delay)
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")


async def stream_chat(
    messages: list[dict[str, str]],
    *,
    meta: dict | None = None,
    models: list[str] | None = None,
) -> AsyncIterator[str]:
    """Yield the answer as text fragments as they're generated.

    Tries LLM_MODEL, then each of LLM_FALLBACK_MODELS in order (or exactly
    `models`, if given; the eval harness passes a single model so results
    aren't silently mixed across models). Falling back is only possible
    before the first token is sent: once the user has seen part of an
    answer, switching models mid-stream would garble it.

    If `meta` is given, it's filled with the model that actually answered
    and token usage: {"model": ..., "usage": {"prompt_tokens", "completion_tokens"}}.
    """
    meta = meta if meta is not None else {}

    if settings.llm_provider == "fake":
        meta["model"] = "fake"
        count = 0
        async for token in _fake_stream(messages):
            count += 1
            yield token
        meta["usage"] = {
            "prompt_tokens": sum(len(m["content"]) for m in messages) // 4,
            "completion_tokens": count,
        }
        return

    models = models or [settings.llm_model, *settings.llm_fallback_model_list]
    last_error: Exception | None = None
    for model in models:
        started = False
        try:
            stream = await _get_client().chat.completions.create(
                model=model,
                messages=messages,
                temperature=settings.llm_temperature,
                stream=True,
                stream_options={"include_usage": True},  # final chunk carries token counts
            )
            async for chunk in stream:
                if chunk.usage:
                    meta["usage"] = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                    }
                if chunk.choices and chunk.choices[0].delta.content:
                    if not started:
                        started = True
                        meta["model"] = model
                    yield chunk.choices[0].delta.content
            meta.setdefault("model", model)
            return
        except Exception as e:
            if started:
                raise
            log.warning("model %s failed (%s); trying next fallback", model, e)
            last_error = e
    raise LLMError(f"All models failed ({', '.join(models)}). Last error: {last_error}") from last_error


async def _fake_stream(messages: list[dict[str, str]]) -> AsyncIterator[str]:
    """Deterministic stand-in for an LLM: echoes the top excerpt.

    FAKE_LLM_TTFT_MS / FAKE_LLM_TOKENS_PER_S make it *wait* like a real API
    (the request holds its resources while waiting, which is what a load
    test needs to reproduce), without spending any quota.
    """
    prompt = messages[-1]["content"]
    match = re.search(r'<excerpt n="1"[^>]*>\n?(.*?)</excerpt>', prompt, re.S)
    excerpt = " ".join(match.group(1).split())[:300] if match else "(no excerpt)"
    answer = f"[fake LLM] The most relevant excerpt says: {excerpt} [1]"
    per_token = 1 / settings.fake_llm_tokens_per_s if settings.fake_llm_tokens_per_s else 0.01
    await asyncio.sleep(settings.fake_llm_ttft_ms / 1000)
    for i, word in enumerate(answer.split(" ")):
        if i:
            await asyncio.sleep(per_token)
        yield word + " "

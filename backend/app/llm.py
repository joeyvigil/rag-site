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

from openai import AsyncOpenAI

from app.config import settings

log = logging.getLogger(__name__)

class LLMError(RuntimeError):
    pass


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
    raise LLMError(f"All models failed ({', '.join(models)}). Last error: {last_error}")


async def _fake_stream(messages: list[dict[str, str]]) -> AsyncIterator[str]:
    """Deterministic stand-in for an LLM: echoes the top excerpt."""
    prompt = messages[-1]["content"]
    match = re.search(r'<excerpt n="1"[^>]*>\n?(.*?)</excerpt>', prompt, re.S)
    excerpt = " ".join(match.group(1).split())[:300] if match else "(no excerpt)"
    answer = f"[fake LLM] The most relevant excerpt says: {excerpt} [1]"
    for word in answer.split(" "):
        await asyncio.sleep(0.01)
        yield word + " "

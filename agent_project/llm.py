"""Thin wrapper around an OpenAI-compatible chat API (DashScope / Kimi / ...).

Everything LLM-related goes through `chat()` so tests can monkeypatch a
single function and the graph nodes stay provider-agnostic.

Token usage of every successful call is recorded in a process-local log
(`usage_summary()` / `reset_usage()`), so the eval runner and the UI can
report real cost numbers instead of guesses.
"""

from __future__ import annotations

import contextvars
import time

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

from . import config

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not config.LLM_API_KEY:
            raise RuntimeError(
                "LLM_API_KEY 未配置：请在环境变量或 .env 中设置 "
                "（DashScope 或 Kimi 的 API Key，见 .env.example）。"
            )
        _client = OpenAI(
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY,
            timeout=config.LLM_TIMEOUT,
        )
    return _client


# ---------------------------------------------------------------------------
# Token usage tracking (process-local; tests monkeypatch chat() so they never
# touch this log). Listeners get every recorded call — the API layer uses one
# to persist per-call usage into SQLite. `_session_ctx` tags calls with the
# serving session; anyio/run_in_threadpool propagates contextvars into the
# worker thread, so the tag survives the sync-offload in the API layer.
# ---------------------------------------------------------------------------

_usage_log: list[dict] = []
_usage_listeners: list = []
_session_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "llm_session_id", default=None
)


def set_session_context(session_id: str | None):
    """Tag subsequent LLM calls in this context with a session id."""
    return _session_ctx.set(session_id)


def reset_session_context(token) -> None:
    _session_ctx.reset(token)


def add_usage_listener(fn) -> None:
    """fn(record) is called for every successful call's usage; listener
    failures are swallowed so metering can never break the serving path.
    Idempotent: re-registering the same function is a no-op (lifespan
    restarts must not pile up duplicate writes)."""
    if fn not in _usage_listeners:
        _usage_listeners.append(fn)


def remove_usage_listener(fn) -> None:
    if fn in _usage_listeners:
        _usage_listeners.remove(fn)


def _record_usage(model: str, resp) -> None:
    u = getattr(resp, "usage", None)
    if u is None:
        return
    record = {
        "model": model,
        "prompt_tokens": getattr(u, "prompt_tokens", None),
        "completion_tokens": getattr(u, "completion_tokens", None),
        "total_tokens": getattr(u, "total_tokens", None),
        "session_id": _session_ctx.get(),
    }
    _usage_log.append(record)
    for fn in _usage_listeners:
        try:
            fn(record)
        except Exception:
            pass


def reset_usage() -> None:
    _usage_log.clear()


def usage_summary() -> dict:
    """Aggregate tokens since the last reset; None fields mean the provider
    did not return usage for those calls."""
    def _total(key: str):
        vals = [r[key] for r in _usage_log if r.get(key) is not None]
        return sum(vals) if vals else None

    return {
        "calls": len(_usage_log),
        "prompt_tokens": _total("prompt_tokens"),
        "completion_tokens": _total("completion_tokens"),
        "total_tokens": _total("total_tokens"),
    }


def chat(
    messages: list[dict],
    *,
    temperature: float | None = None,
    model: str | None = None,
    max_retries: int = 4,
    json_mode: bool = False,
) -> str:
    """Single-turn chat completion; returns the assistant message text.

    Retries with exponential backoff on 429 and on transient network errors
    (read timeout / connection failure) — free-tier keys (e.g. Kimi's 3 RPM
    org cap) and slow reasoning models otherwise fail whole turns on a
    transient hiccup.

    json_mode=True 时使用 OpenAI 兼容的 response_format={"type": "json_object"}，
    强制输出纯 JSON（槽位抽取等结构化场景用；模型仍可能输出非法 JSON，
    调用方须做 Pydantic 校验，见 nodes._extract_slots）。
    """
    delay = 5.0
    for attempt in range(max_retries + 1):
        try:
            used_model = model or config.LLM_MODEL
            kwargs: dict = dict(
                model=used_model,
                messages=messages,
                temperature=config.LLM_TEMPERATURE if temperature is None else temperature,
            )
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = get_client().chat.completions.create(**kwargs)
            _record_usage(used_model, resp)
            return (resp.choices[0].message.content or "").strip()
        except (RateLimitError, APITimeoutError, APIConnectionError):
            if attempt >= max_retries:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    raise AssertionError("unreachable")


def chat_stream(
    messages: list[dict],
    *,
    temperature: float | None = None,
    model: str | None = None,
    max_retries: int = 2,
):
    """流式版本：逐块产出回答文本（生成器）。

    只在建连阶段重试（流开始之后不重试，避免重复输出）。
    stream_options include_usage 让末块携带 token 用量，照常落计量。
    """
    delay = 5.0
    stream = None
    used_model = model or config.LLM_MODEL
    for attempt in range(max_retries + 1):
        try:
            stream = get_client().chat.completions.create(
                model=used_model,
                messages=messages,
                temperature=config.LLM_TEMPERATURE if temperature is None else temperature,
                stream=True,
                stream_options={"include_usage": True},
            )
            break
        except (RateLimitError, APITimeoutError, APIConnectionError):
            if attempt >= max_retries:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    if stream is None:
        raise AssertionError("unreachable")
    for chunk in stream:
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            _record_usage(used_model, chunk)
        choices = getattr(chunk, "choices", None) or []
        if choices:
            delta = getattr(choices[0].delta, "content", None)
            if delta:
                yield delta

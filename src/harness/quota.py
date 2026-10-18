"""Daily quota ledger for the free-tier providers (revision-2 runs).

Every live call made by a revision-2 runner is appended to
results/quota_ledger.jsonl (timestamp, provider, model, tokens, cached flag).
Before a call, a runner asks `Budget.allow(...)`; it refuses once the
remaining allowance would be exceeded, so runs stop cleanly at the cap
instead of burning requests on 429 errors.

Caps (free tier, confirmed from provider error bodies during the study):
  Gemini   20 requests per model per day; the day resets at midnight
           US Pacific time.
  Groq     200,000 tokens per model per rolling 24 hours.
  OpenRouter 50 requests per day (not used in revision-2).

A safety margin is kept below each cap. Cached responses cost nothing and
are recorded with cached=True but not counted.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

LEDGER = Path("results/quota_ledger.jsonl")
GEMINI_DAILY_REQUESTS = 20
GEMINI_MARGIN = 1  # keep one request spare
GROQ_TOKENS_24H = 200_000
GROQ_MARGIN = 8_000

PACIFIC = timezone(timedelta(hours=-7))  # PDT; Gemini daily reset at local midnight


def _rows() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]


def record(provider: str, model: str, tokens: int, cached: bool, note: str = "") -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts": time.time(), "iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "provider": provider, "model": model, "tokens": int(tokens or 0), "cached": bool(cached), "note": note}
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def gemini_requests_today(model: str) -> int:
    start = datetime.now(PACIFIC).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    return sum(1 for r in _rows() if r["provider"] == "gemini" and r["model"] == model and not r["cached"] and r["ts"] >= start)


def groq_tokens_24h(model: str) -> int:
    since = time.time() - 24 * 3600
    return sum(r["tokens"] for r in _rows() if r["provider"] == "groq" and r["model"] == model and not r["cached"] and r["ts"] >= since)


def allow(provider: str, model: str, est_tokens: int = 0, est_requests: int = 1) -> bool:
    """True if the next call(s) fit within today's remaining allowance."""
    if provider == "gemini":
        return gemini_requests_today(model) + est_requests <= GEMINI_DAILY_REQUESTS - GEMINI_MARGIN
    if provider == "groq":
        return groq_tokens_24h(model) + est_tokens <= GROQ_TOKENS_24H - GROQ_MARGIN
    return True


def status() -> str:
    return (f"gemini-2.5-flash requests today: {gemini_requests_today('gemini-2.5-flash')}/{GEMINI_DAILY_REQUESTS}; "
            f"qwen3.8 tokens 24h: {groq_tokens_24h('qwen/qwen3.8-27b'):,}/{GROQ_TOKENS_24H:,}; "
            f"gpt-oss-120b tokens 24h: {groq_tokens_24h('openai/gpt-oss-120b'):,}/{GROQ_TOKENS_24H:,}")


class BudgetExhausted(RuntimeError):
    """Raised when the next call would exceed today's remaining allowance."""


def metered(provider: str, model: str, fn, est_tokens: int = 0, note: str = ""):
    """Run fn() (a provider call) only if it fits the budget; record its usage.

    Groq calls are counted by prompt + completion tokens; Gemini calls by
    request. Calls answered from the response cache are recorded but not
    counted.
    """
    from harness import providers

    if not allow(provider, model, est_tokens=est_tokens):
        raise BudgetExhausted(f"{provider}/{model}: budget reached ({status()})")
    providers.LAST_FROM_CACHE = False
    result = fn()
    cached = providers.LAST_FROM_CACHE
    tokens = (result.input_tokens or 0) + (result.output_tokens or 0)
    record(provider, model, tokens, cached, note)
    return result

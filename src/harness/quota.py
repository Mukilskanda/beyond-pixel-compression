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
import re
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


def record_block(provider: str, model: str, until_ts: float, msg: str) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts": time.time(), "iso": datetime.now(timezone.utc).isoformat(timespec="seconds"), "provider": provider,
           "model": model, "tokens": 0, "cached": False, "block_until": until_ts, "note": "provider limit: " + msg}
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def blocked_until(provider: str, model: str) -> float:
    return max([r.get("block_until", 0) for r in _rows() if r["provider"] == provider and r["model"] == model] or [0])


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
    if blocked_until(provider, model) > time.time():
        until = datetime.fromtimestamp(blocked_until(provider, model)).strftime("%H:%M")
        raise BudgetExhausted(f"{provider}/{model}: provider-reported limit, blocked until {until} ({status()})")
    providers.LAST_FROM_CACHE = False
    try:
        for attempt in range(5):
            try:
                result = fn()
                break
            except Exception as exc:
                # Per-minute limits (e.g. Groq OTPM) clear in seconds: wait and retry.
                msg = str(exc)
                if attempt < 4 and "per minute" in msg:
                    m = re.search(r"try again in ([\d.]+)s", msg)
                    time.sleep((float(m.group(1)) if m else 65.0) + 5)
                    continue
                raise
    except Exception as exc:
        # The provider's own meter is authoritative (Groq counts more than the
        # usage fields show). On a daily-limit 429, record a block until the
        # time the provider gives and stop cleanly.
        msg = str(exc)
        if "tokens per day" in msg or "TPD" in msg or "per day" in msg.lower() or "RESOURCE_EXHAUSTED" in msg:
            wait = 3600.0
            m = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", msg)
            if m:
                wait = int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60 + float(m.group(3)) + 30
            record_block(provider, model, time.time() + wait, msg[:300])
            raise BudgetExhausted(f"{provider}/{model}: provider daily limit, retry after {wait / 60:.0f} min") from exc
        raise
    cached = providers.LAST_FROM_CACHE
    tokens = (result.input_tokens or 0) + (result.output_tokens or 0)
    record(provider, model, tokens, cached, note)
    return result

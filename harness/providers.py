"""Provider wrappers for vision API calls. Version-stamped for reproducibility.

Four backends behind one interface:
- OllamaProvider: free, local, unlimited — use for pipeline development.
- AnthropicProvider: real hosted API, paid — use for measurement runs that
  produce the paper's numbers (token counts are ground truth from usage field).
- GeminiProvider: real hosted API, free tier (Gemini 2.5 Flash) — use for
  validating cost-model logic (e.g. pixelpath/packing.py) against real
  provider token counts at zero cost. NOTE: free-tier behavior (rate limits,
  possibly different tokenization/billing than paid tier) may not represent
  paid-tier production behavior — flag this explicitly wherever free-tier
  numbers inform a methodology decision, don't silently generalize them.
  Also NOTE: free-tier quota is 20 requests/day/model (confirmed from a live
  429 error body), not the 15 RPM/1500 RPD figure general Gemini API docs
  state — don't rediscover this the hard way in a future session.
- GroqProvider: real hosted API, free tier (no credit card at
  console.groq.com). Second provider for the cross-provider mechanism check
  (paper/draft.md §9) — chosen after OpenAI was dropped (no approved budget)
  and the originally-planned model (Llama 4 Scout vision) turned out to have
  been shut down by Groq on 2026-07-17, before this project could use it.
  Currently configured for qwen/qwen3.6-27b, Groq's vision-capable model as
  of this writing. NOTE: verify token usage is actually populated before
  trusting any number from this provider — preview/vision models on fast
  inference platforms have been known to have incomplete metering; this was
  explicitly checked before use (see paper/draft.md §9).
"""
from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from harness.cache import ResponseCache
except ImportError:  # running as a script from inside harness/
    from cache import ResponseCache

# Loads GEMINI_API_KEY / GROQ_API_KEY / ANTHROPIC_API_KEY from a .env file at
# the repo root, if one exists, so keys persist across sessions instead of
# needing re-export in every new shell (the cause of every provider sweep in
# prior sessions silently never running - see paper/draft.md §4.3/§8). Real
# key VALUES are never committed - .env is gitignored; see .env.example for
# the expected format. Safe no-op if .env doesn't exist or dotenv isn't
# installed (falls back to whatever's already in the environment).
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


def credentials_status() -> dict[str, bool]:
    """Real (not assumed) check of which provider credentials are actually
    readable in this process right now - call this at the start of any
    sweep before spending quota, per the standing rule that a sweep must
    confirm credentials load before proceeding to spend-dependent work."""
    return {
        "GEMINI_API_KEY": bool(os.environ.get("GEMINI_API_KEY")),
        "GROQ_API_KEY": bool(os.environ.get("GROQ_API_KEY")),
        "ANTHROPIC_API_KEY": bool(os.environ.get("ANTHROPIC_API_KEY")),
    }


@dataclass
class CallResult:
    text: str
    input_tokens: int | None  # None when the backend has no real token accounting (e.g. local)
    output_tokens: int | None
    latency_s: float
    model: str
    raw: dict[str, Any]
    # Provider-reported stop/finish reason, normalized to lowercase strings
    # ("stop", "length", "max_tokens", "end_turn", etc. depending on provider)
    # where available, else None. Added after discovering that qwen/qwen3.6-27b's
    # <think> preamble silently consumed an entire max_tokens budget and produced
    # a truncated (not wrong) answer that scored anls=0.0 indistinguishably from
    # a genuine wrong answer (paper/draft.md §9.4). This field lets callers (see
    # harness/metrics.is_truncated) detect that case instead of re-discovering it
    # ad hoc per provider. Defaults to None for providers/paths where it isn't
    # populated yet (e.g. OllamaProvider) - treat None as "unknown", not "not truncated".
    finish_reason: str | None = None


class OllamaProvider:
    """Local, free, unlimited. No real token cost — dev/test only, never for measurement runs."""

    def __init__(self, model: str = "llava", base_url: str = "http://localhost:11434", cache: ResponseCache | None = None):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.cache = cache

    def call(self, image_path: str | Path, prompt: str, **settings: Any) -> CallResult:
        import requests

        image_bytes = Path(image_path).read_bytes()
        settings_key = {"prompt": prompt, **settings}

        if self.cache is not None:
            cached = self.cache.get(image_bytes, self.model, settings_key)
            if cached is not None:
                return CallResult(**cached)

        image_b64 = base64.b64encode(image_bytes).decode()
        start = time.monotonic()
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                **settings,
            },
            timeout=120,
        )
        resp.raise_for_status()
        latency = time.monotonic() - start
        data = resp.json()

        result = CallResult(
            text=data.get("response", ""),
            input_tokens=None,
            output_tokens=data.get("eval_count"),
            latency_s=latency,
            model=self.model,
            raw=data,
        )

        if self.cache is not None:
            self.cache.put(image_bytes, self.model, settings_key, result.__dict__)

        return result


class GeminiProvider:
    """Real hosted API, free tier (no billing required). Token counts come
    from response.usage_metadata — ground truth per §3.3, same as
    AnthropicProvider, but validate whether free-tier tokenization/billing
    matches paid-tier before treating these numbers as production-representative.

    Auth: reads GEMINI_API_KEY from the environment automatically (the
    google-genai SDK also accepts GOOGLE_API_KEY, with GOOGLE_API_KEY taking
    precedence if both are set — use GEMINI_API_KEY alone to avoid ambiguity).
    Get a free key with no credit card at aistudio.google.com.

    Free tier rate limits (gemini-2.5-flash, subject to change — check
    ai.google.dev/gemini-api/docs/rate-limits): 15 requests/minute, 1500
    requests/day. This class does not rate-limit for you — space calls out
    at the call site if you're running more than a handful.
    """

    def __init__(self, model: str = "gemini-2.5-flash", cache: ResponseCache | None = None):
        self.model = model
        self.cache = cache
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client()
        return self._client

    def call(self, image_path: str | Path, prompt: str, **settings: Any) -> CallResult:
        from google.genai import types
        from PIL import Image

        image_bytes = Path(image_path).read_bytes()
        settings_key = {"prompt": prompt, **settings}

        if self.cache is not None:
            cached = self.cache.get(image_bytes, self.model, settings_key)
            if cached is not None:
                return CallResult(**cached)

        img = Image.open(image_path)
        config = types.GenerateContentConfig(**settings) if settings else None

        start = time.monotonic()
        response = self.client.models.generate_content(
            model=self.model,
            contents=[img, prompt],
            config=config,
        )
        latency = time.monotonic() - start

        usage = response.usage_metadata
        finish_reason = None
        if getattr(response, "candidates", None):
            fr = getattr(response.candidates[0], "finish_reason", None)
            finish_reason = str(fr).lower() if fr is not None else None
        result = CallResult(
            text=response.text or "",
            input_tokens=usage.prompt_token_count if usage else None,
            output_tokens=usage.candidates_token_count if usage else None,
            latency_s=latency,
            model=self.model,
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
            finish_reason=finish_reason,
        )

        if self.cache is not None:
            self.cache.put(image_bytes, self.model, settings_key, result.__dict__)

        return result

    def call_text_only(self, text: str, prompt: str, **settings: Any) -> CallResult:
        """The pure-TEXT-route call this project's own runner never had:
        harness/runner.py's real sweeps only ever call provider.call(image_path,
        question), which always attaches an image — router/rules.py's Route.TEXT
        (image_bytes=None) has never actually reached a real API request before
        this method existed. Sends `text` (e.g. compressed OCR markdown) and
        `prompt` (the question) as text-only content, no image at all, so the
        real billed cost of skipping the image entirely can finally be measured
        against §5/§9.4's image-only and OCR-added-on-top-of-image numbers.

        Cached like image calls, but keyed on the text content's hash instead
        of an image hash (ResponseCache.get/put take `image_bytes` positionally
        — reused here by encoding the text; the parameter name is a misnomer
        for this call path, not a type mismatch).
        """
        from google.genai import types

        combined = f"{text}\n\n{prompt}"
        content_bytes = combined.encode("utf-8")
        settings_key = {"prompt": prompt, "text_only": True, **settings}

        if self.cache is not None:
            cached = self.cache.get(content_bytes, self.model, settings_key)
            if cached is not None:
                return CallResult(**cached)

        config = types.GenerateContentConfig(**settings) if settings else None

        start = time.monotonic()
        response = self.client.models.generate_content(
            model=self.model,
            contents=[combined],
            config=config,
        )
        latency = time.monotonic() - start

        usage = response.usage_metadata
        finish_reason = None
        if getattr(response, "candidates", None):
            fr = getattr(response.candidates[0], "finish_reason", None)
            finish_reason = str(fr).lower() if fr is not None else None
        result = CallResult(
            text=response.text or "",
            input_tokens=usage.prompt_token_count if usage else None,
            output_tokens=usage.candidates_token_count if usage else None,
            latency_s=latency,
            model=self.model,
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
            finish_reason=finish_reason,
        )

        if self.cache is not None:
            self.cache.put(content_bytes, self.model, settings_key, result.__dict__)

        return result


class GroqProvider:
    """Real hosted API, free tier (no credit card required at
    console.groq.com). OpenAI-compatible chat-completions shape, so this
    mirrors OpenAIProvider's request/response handling. Token counts come
    from response.usage — but VERIFY this is actually populated with real
    numbers before trusting it for cost-model work (see class-level
    docstring in providers.py module header — preview/vision models have
    been known to have incomplete metering; do not assume it works just
    because the SDK's type definition has the fields).

    Auth: reads GROQ_API_KEY from the environment automatically (standard
    groq-python SDK behavior, confirmed in SDK source).

    The `detail` parameter ("low", "high", or "auto") mirrors OpenAI's
    image_url.detail field structurally (Groq's API is OpenAI-compatible),
    but whether Qwen3.6-27B on Groq actually respects it and changes real
    cost has not been assumed — verify empirically before relying on it.
    """

    def __init__(self, model: str = "qwen/qwen3.6-27b", cache: ResponseCache | None = None):
        self.model = model
        self.cache = cache
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import groq
            self._client = groq.Groq()
        return self._client

    def call(self, image_path: str | Path, prompt: str, media_type: str = "image/png", **settings: Any) -> CallResult:
        image_bytes = Path(image_path).read_bytes()
        settings_key = {"prompt": prompt, **settings}

        if self.cache is not None:
            cached = self.cache.get(image_bytes, self.model, settings_key)
            if cached is not None:
                return CallResult(**cached)

        image_b64 = base64.standard_b64encode(image_bytes).decode()
        detail = settings.pop("detail", None)  # "low" | "high" | "auto" — may or may not be respected, verify
        max_tokens = settings.pop("max_tokens", 1024)

        image_url: dict[str, Any] = {"url": f"data:{media_type};base64,{image_b64}"}
        if detail is not None:
            image_url["detail"] = detail

        start = time.monotonic()
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": image_url},
                    {"type": "text", "text": prompt},
                ],
            }],
            **settings,
        )
        latency = time.monotonic() - start

        text = response.choices[0].message.content or ""
        usage = response.usage
        raw_finish_reason = getattr(response.choices[0], "finish_reason", None)
        result = CallResult(
            text=text,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            latency_s=latency,
            model=response.model,
            raw=response.model_dump(),
            finish_reason=str(raw_finish_reason).lower() if raw_finish_reason is not None else None,
        )

        if self.cache is not None:
            self.cache.put(image_bytes, self.model, settings_key, result.__dict__)

        return result

    def call_text_only(self, text: str, prompt: str, **settings: Any) -> CallResult:
        """See GeminiProvider.call_text_only's docstring — same rationale:
        harness/runner.py never exercises Route.TEXT against a real API,
        since provider.call() always attaches an image. This sends `text`
        (e.g. compressed OCR markdown) plus `prompt` with no image content
        at all, OpenAI-compatible text-only message shape."""
        combined = f"{text}\n\n{prompt}"
        content_bytes = combined.encode("utf-8")
        settings_key = {"prompt": prompt, "text_only": True, **settings}

        if self.cache is not None:
            cached = self.cache.get(content_bytes, self.model, settings_key)
            if cached is not None:
                return CallResult(**cached)

        max_tokens = settings.pop("max_tokens", 1024)

        start = time.monotonic()
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": combined}],
            **settings,
        )
        latency = time.monotonic() - start

        text_out = response.choices[0].message.content or ""
        usage = response.usage
        raw_finish_reason = getattr(response.choices[0], "finish_reason", None)
        result = CallResult(
            text=text_out,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            latency_s=latency,
            model=response.model,
            raw=response.model_dump(),
            finish_reason=str(raw_finish_reason).lower() if raw_finish_reason is not None else None,
        )

        if self.cache is not None:
            self.cache.put(content_bytes, self.model, settings_key, result.__dict__)

        return result


class AnthropicProvider:
    """Real hosted API. Token counts come from response.usage — the ground truth per §3.3."""

    def __init__(self, model: str = "claude-opus-5", cache: ResponseCache | None = None):
        self.model = model
        self.cache = cache
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def call(self, image_path: str | Path, prompt: str, media_type: str = "image/png", **settings: Any) -> CallResult:
        image_bytes = Path(image_path).read_bytes()
        settings_key = {"prompt": prompt, **settings}

        if self.cache is not None:
            cached = self.cache.get(image_bytes, self.model, settings_key)
            if cached is not None:
                return CallResult(**cached)

        image_b64 = base64.standard_b64encode(image_bytes).decode()
        max_tokens = settings.pop("max_tokens", 1024)

        start = time.monotonic()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
            **settings,
        )
        latency = time.monotonic() - start

        text = next((b.text for b in response.content if b.type == "text"), "")
        stop_reason = getattr(response, "stop_reason", None)
        result = CallResult(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_s=latency,
            model=response.model,
            raw=response.to_dict(),
            finish_reason=str(stop_reason).lower() if stop_reason is not None else None,
        )

        if self.cache is not None:
            self.cache.put(image_bytes, self.model, settings_key, result.__dict__)

        return result

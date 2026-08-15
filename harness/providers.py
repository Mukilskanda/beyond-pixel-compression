"""Provider wrappers for vision API calls. Version-stamped for reproducibility.

Three backends behind one interface:
- OllamaProvider: free, local, unlimited — use for pipeline development.
- AnthropicProvider: real hosted API, paid — use for measurement runs that
  produce the paper's numbers (token counts are ground truth from usage field).
- GeminiProvider: real hosted API, free tier (Gemini 2.5 Flash) — use for
  validating cost-model logic (e.g. pixelpath/packing.py) against real
  provider token counts at zero cost. NOTE: free-tier behavior (rate limits,
  possibly different tokenization/billing than paid tier) may not represent
  paid-tier production behavior — flag this explicitly wherever free-tier
  numbers inform a methodology decision, don't silently generalize them.
"""
from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from harness.cache import ResponseCache
except ImportError:  # running as a script from inside harness/
    from cache import ResponseCache


@dataclass
class CallResult:
    text: str
    input_tokens: int | None  # None when the backend has no real token accounting (e.g. local)
    output_tokens: int | None
    latency_s: float
    model: str
    raw: dict[str, Any]


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
        result = CallResult(
            text=response.text or "",
            input_tokens=usage.prompt_token_count if usage else None,
            output_tokens=usage.candidates_token_count if usage else None,
            latency_s=latency,
            model=self.model,
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )

        if self.cache is not None:
            self.cache.put(image_bytes, self.model, settings_key, result.__dict__)

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
        result = CallResult(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_s=latency,
            model=response.model,
            raw=response.to_dict(),
        )

        if self.cache is not None:
            self.cache.put(image_bytes, self.model, settings_key, result.__dict__)

        return result

"""Provider wrappers for vision API calls. Version-stamped for reproducibility.

Two backends behind one interface:
- OllamaProvider: free, local, unlimited — use for pipeline development.
- AnthropicProvider: real hosted API — use only for measurement runs that
  produce the paper's numbers (token counts are ground truth from usage field).
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

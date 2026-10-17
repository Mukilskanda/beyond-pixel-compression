"""Baseline preprocessing methods — every one must be implemented, per plan §3.2."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image


def unreduced(image_path: str | Path) -> bytes:
    """Baseline 1: full image, unchanged. The fidelity ceiling and cost ceiling."""
    return Path(image_path).read_bytes()


def uniform_downscale(image_path: str | Path, scale: float = 0.5) -> bytes:
    """Baseline 2: resize to `scale` fraction. Often works embarrassingly well."""
    img = Image.open(image_path)
    new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    resized = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    resized.save(buf, format=img.format or "PNG")
    return buf.getvalue()


def jpeg_quality_reduction(image_path: str | Path, quality: int = 50) -> bytes:
    """Baseline 3: same pixel dimensions, lower JPEG quality. Tests whether compression artifacts matter."""
    img = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def fixed_ratio_crop(image_path: str | Path, keep_fraction: float = 0.6) -> bytes:
    """Baseline 6: content-agnostic center crop."""
    img = Image.open(image_path)
    w, h = img.width, img.height
    cw, ch = int(w * keep_fraction), int(h * keep_fraction)
    left, top = (w - cw) // 2, (h - ch) // 2
    cropped = img.crop((left, top, left + cw, top + ch))
    buf = io.BytesIO()
    cropped.save(buf, format=img.format or "PNG")
    return buf.getvalue()


# Baseline 4 (provider low-detail mode) is not a pixel transform — it's a
# request-time flag on the provider call. For Gemini, that's the
# `media_resolution` config field, which GeminiProvider.call(**settings)
# already passes straight through to GenerateContentConfig (no code change
# needed there). Verified with real API calls (2026-08-16): on a 1000x1000
# image, MEDIA_RESOLUTION_LOW dropped Gemini's cost from the default flat 258
# tokens to 74 tokens — a real, working lever, and notably the FIRST thing
# found that changes Gemini's image cost at all (see pixelpath/packing.py's
# module docstring — the default/unspecified resolution is flat regardless
# of size/content, but explicit resolution levels do vary cost). Usage:
# provider.call(image_path, prompt, media_resolution="MEDIA_RESOLUTION_LOW").
# Anthropic doesn't expose an equivalent flag as of this writing.
GEMINI_LOW_DETAIL_SETTINGS = {"media_resolution": "MEDIA_RESOLUTION_LOW"}

# Baseline 5 (OCR-everything) lives in baselines/ocr_everything.py — separate
# module since it returns OCR text, not transformed image bytes.

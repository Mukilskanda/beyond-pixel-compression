"""Candidate 3 (multimodal caching) viability check — completes the
caching investigation from §11 (similarity-based visual caching, found
unsafe) with the OTHER half of the multimodal caching gap the literature
review (§2.5) named: exact-repeat visual caching, where a provider
literally caches identical previously-sent content.

Two independent checks, cheapest first:

1. Groq: is there a model that is BOTH vision-capable AND has documented
   prompt-caching support? Groq's caching docs list only GPT-OSS models
   (openai/gpt-oss-120b, gpt-oss-20b, gpt-oss-safeguard-20b) as
   cache-supported. This function sends ONE real image call to
   gpt-oss-120b to check directly (not just from docs) whether it even
   accepts image content.

2. Gemini: implicit caching requires a 2,048-token minimum per Gemini's
   own docs (ai.google.dev/gemini-api/docs/caching). This project's own
   real-measured single-image and tiled-image costs (§3.5/§7, §13.1) are
   258-322 tokens - compares that real number against the documented
   threshold rather than assuming.

Both checks are near-zero-cost (one real call at most) by design - this
is a viability gate, not the caching mechanism test itself. If either
comes back viable, a real cache-hit test (send the same image twice,
compare cost/cached_tokens) is the natural next step; if not, that is
reported as a real, informative negative result per this project's
standing discipline (§2.5 already argues this gap in the literature -
finding it also unaddressed on the two real providers tested here
completes that argument with evidence, not just citation).

Usage: python experiments/image_caching_viability_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.providers import GroqProvider

CACHE_SUPPORTED_GROQ_MODELS = {"openai/gpt-oss-120b", "openai/gpt-oss-20b", "openai/gpt-oss-safeguard-20b"}
GEMINI_IMPLICIT_CACHE_MIN_TOKENS = 2048
# Real, previously measured (paper/draft.md §3.5, §7, §13.1) - not re-derived here.
MEASURED_GEMINI_IMAGE_TOKENS = {"single_image": 258, "tiled_n2": 267, "tiled_n4": 267}


def check_groq_vision_caching_overlap() -> None:
    print("--- Groq: does a cache-supported model accept image content? ---")
    print(f"Documented cache-supported models: {sorted(CACHE_SUPPORTED_GROQ_MODELS)}")

    provider = GroqProvider(model="openai/gpt-oss-120b")
    try:
        provider.call("data/docvqa_cache/49153.png", "Describe this in one sentence.", max_tokens=50, reasoning_effort="low")
        print("RESULT: openai/gpt-oss-120b accepted image content - caching IS testable on Groq.")
    except Exception as exc:
        print(f"RESULT: openai/gpt-oss-120b rejected image content - {exc}")
        print("Groq's only documented cache-supported models are text-only (GPT-OSS family).")
        print("qwen/qwen3.6-27b (this project's vision-capable Groq model) has no documented caching support.")
        print("VERDICT: no valid (vision-capable, cache-supported) model combination exists on Groq with currently available models.")


def check_gemini_threshold_reachability() -> None:
    print("\n--- Gemini: does this project's real image-call size ever reach the implicit-caching threshold? ---")
    print(f"Documented implicit-caching minimum: {GEMINI_IMPLICIT_CACHE_MIN_TOKENS} tokens (gemini-2.5-flash)")
    for label, tokens in MEASURED_GEMINI_IMAGE_TOKENS.items():
        ratio = tokens / GEMINI_IMPLICIT_CACHE_MIN_TOKENS
        print(f"  {label}: {tokens} tokens measured ({ratio * 100:.1f}% of threshold) - {'REACHES' if tokens >= GEMINI_IMPLICIT_CACHE_MIN_TOKENS else 'below threshold'}")
    print("VERDICT: every real image-call size this project has ever measured on Gemini is far below the caching threshold.")


if __name__ == "__main__":
    check_groq_vision_caching_overlap()
    check_gemini_threshold_reachability()

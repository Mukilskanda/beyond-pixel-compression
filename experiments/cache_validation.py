"""Pre-spend gate: prove the cache actually dedupes repeated calls before any
real API budget is approved. Run this against a real provider on a handful of
samples — if the second pass shows zero real API calls (all cache hits), the
cache is doing its job and it's safe to move to a real measurement run.

Uses real DocVQA samples by default (no registration needed — see
data/docvqa_loader.py) rather than synthetic stand-ins, since the point is to
validate against the actual distribution of data the real experiments will use.

Usage (once ANTHROPIC_API_KEY is set and a real dollar cap is approved for
this specific run — see harness/providers.py:AnthropicProvider):

    python experiments/cache_validation.py --model claude-opus-5 --n 5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.cache import ResponseCache
from harness.providers import AnthropicProvider


def run_cache_validation(model: str, n: int, dataset: str = "docvqa", cache_path: str = "results/cache.sqlite") -> None:
    if dataset == "docvqa":
        from data.docvqa_loader import load_docvqa
        print(f"Loading {n} real DocVQA samples (no registration needed, streamed via HF Hub)...")
        samples = load_docvqa(limit=n)
        get_qa = lambda s: (s.image_path, s.question)
    else:
        from data.synthetic import generate_dataset
        print(f"Generating {n} synthetic samples for cache validation (--dataset synthetic)...")
        samples = generate_dataset(output_dir="data/scratch/cache_validation", per_category=max(1, n // 5))[:n]
        get_qa = lambda s: (s.image_path, s.question)

    cache = ResponseCache(cache_path)
    provider = AnthropicProvider(model=model, cache=cache)

    print(f"\n--- Pass 1: cold cache, expect {len(samples)} real API calls ---")
    pass1_times = []
    for s in samples:
        image_path, question = get_qa(s)
        start = time.monotonic()
        result = provider.call(image_path, question)
        elapsed = time.monotonic() - start
        pass1_times.append(elapsed)
        print(f"  {elapsed:.2f}s, input_tokens={result.input_tokens}  Q={question[:50]!r}")

    print(f"\n--- Pass 2: warm cache, expect {len(samples)} cache hits, ~0 real API calls ---")
    pass2_times = []
    for s in samples:
        image_path, question = get_qa(s)
        start = time.monotonic()
        result = provider.call(image_path, question)
        elapsed = time.monotonic() - start
        pass2_times.append(elapsed)
        print(f"  {elapsed:.2f}s, input_tokens={result.input_tokens}")

    avg1 = sum(pass1_times) / len(pass1_times)
    avg2 = sum(pass2_times) / len(pass2_times)
    print(f"\nMean latency: pass 1 (cold) = {avg1:.2f}s, pass 2 (warm) = {avg2:.3f}s")

    if avg2 < avg1 * 0.1:
        print("PASS: cache is dedupering correctly (pass 2 near-instant). Safe to proceed to a real measurement run.")
    else:
        print("WARNING: pass 2 wasn't meaningfully faster than pass 1 — cache may not be hitting. "
              "Do NOT approve further spend until this is investigated (check harness/cache.py key construction).")

    cache.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--n", type=int, default=5, help="number of samples (small on purpose — this is a smoke test, not a real experiment)")
    parser.add_argument("--dataset", choices=["docvqa", "synthetic"], default="docvqa")
    args = parser.parse_args()
    run_cache_validation(model=args.model, n=args.n, dataset=args.dataset)

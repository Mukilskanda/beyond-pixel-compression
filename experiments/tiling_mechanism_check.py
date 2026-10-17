"""Real-billing mechanism check for image tiling — does compositing N
images into one grid actually cost close to 1 flat-cost unit, or does the
larger composite cross a provider's resolution/tile boundary into a more
expensive bucket? This is an empirical question, not assumed: measure
before running any fidelity sweep, same discipline as every other real
finding in this project (§7, §9).

Two conditions per N, per provider:
  - separate: N individual calls, each unreduced (baseline pattern from
    baselines.baselines.unreduced) — the real cost of "N calls, flat cost each"
  - tiled: 1 call with the N images composited via pixelpath.tiling.tile_images

Usage:
    python experiments/tiling_mechanism_check.py --provider gemini --ns 2,4
    python experiments/tiling_mechanism_check.py --provider groq --ns 2,4,6,9
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.docvqa_loader import load_docvqa
from harness.cache import ResponseCache
from harness.providers import GeminiProvider, GroqProvider
from pixelpath.tiling import tile_images

PROVIDERS = {"gemini": GeminiProvider, "groq": GroqProvider}
GENERIC_PROMPT = "Describe what you see in this image."


def run(provider_name: str, ns: list[int], max_tokens: int = 100) -> None:
    max_n = max(ns)
    print(f"Loading {max_n} real DocVQA samples...")
    samples = load_docvqa(limit=max_n)

    cache = ResponseCache("results/cache.sqlite")
    extra = {"max_tokens": max_tokens} if provider_name == "groq" else {}
    provider = PROVIDERS[provider_name](cache=cache)

    print(f"\n{'N':>4}{'separate_total':>18}{'tiled_total':>14}{'ratio':>10}{'verdict':>30}")
    for n in ns:
        subset = samples[:n]

        separate_total = 0
        for s in subset:
            result = provider.call(s.image_path, GENERIC_PROMPT, **extra)
            separate_total += result.input_tokens

        grid = tile_images([s.image_path for s in subset], tile_size=(600, 800))
        tmp_path = Path("data/scratch") / f"tile_n{n}_{provider_name}.png"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        grid.save(tmp_path)

        tiled_result = provider.call(tmp_path, GENERIC_PROMPT, **extra)
        tiled_total = tiled_result.input_tokens

        ratio = tiled_total / separate_total if separate_total else float("nan")
        if ratio < 0.6:
            verdict = "REAL SAVING"
        elif ratio < 1.0:
            verdict = "partial saving"
        elif ratio < 1.5:
            verdict = "roughly a wash"
        else:
            verdict = "MORE EXPENSIVE"
        print(f"{n:>4}{separate_total:>18}{tiled_total:>14}{ratio:>10.2f}{verdict:>30}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=list(PROVIDERS), required=True)
    parser.add_argument("--ns", type=str, required=True, help="comma-separated tile counts, e.g. 2,4,6,9")
    parser.add_argument("--max-tokens", type=int, default=100)
    args = parser.parse_args()
    run(args.provider, [int(x) for x in args.ns.split(",")], max_tokens=args.max_tokens)

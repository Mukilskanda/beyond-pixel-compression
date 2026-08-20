"""Candidate 1: does mode-selection (Gemini's media_resolution="low", the
single lever §3.5-3.6/§7/§9.3 found that changes cost) compound with
tiling (§13, image-count reduction), or does compositing change how
Gemini's tiering logic treats the image (e.g. the composite is already
large enough that MEDIA_RESOLUTION_LOW behaves differently than on a
single image)? Gemini-only - Groq's `detail` parameter has zero measured
effect (§9.3), so there is no mode lever to compound with tiling there.

Four real calls per N: tiling alone (default resolution, already measured
in experiments/tiling_mechanism_check.py, re-measured here for a clean
side-by-side) vs. tiling + MEDIA_RESOLUTION_LOW, same N, same images.

Usage:
    python experiments/tiling_mode_compounding_check.py --ns 2,4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baselines.baselines import GEMINI_LOW_DETAIL_SETTINGS
from data.docvqa_loader import load_docvqa
from harness.cache import ResponseCache
from harness.providers import GeminiProvider
from pixelpath.tiling import tile_images

GENERIC_PROMPT = "Describe what you see in this image."


def run(ns: list[int]) -> None:
    max_n = max(ns)
    print(f"Loading {max_n} real DocVQA samples...")
    samples = load_docvqa(limit=max_n)

    cache = ResponseCache("results/cache.sqlite")
    provider = GeminiProvider(cache=cache)

    print(f"\n{'N':>4}{'tiled_default':>16}{'tiled_low_res':>16}{'ratio':>10}{'verdict':>30}")
    for n in ns:
        subset = samples[:n]
        grid = tile_images([s.image_path for s in subset], tile_size=(600, 800))
        tmp_path = Path("data/scratch") / f"compound_tile_n{n}.png"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        grid.save(tmp_path)

        default_result = provider.call(tmp_path, GENERIC_PROMPT)
        default_tokens = default_result.input_tokens

        low_res_result = provider.call(tmp_path, GENERIC_PROMPT, **GEMINI_LOW_DETAIL_SETTINGS)
        low_res_tokens = low_res_result.input_tokens

        ratio = low_res_tokens / default_tokens if default_tokens else float("nan")
        if ratio < 0.5:
            verdict = "COMPOUNDS (real additional saving)"
        elif ratio < 0.95:
            verdict = "partial additional saving"
        elif ratio <= 1.05:
            verdict = "no additional effect"
        else:
            verdict = "MORE EXPENSIVE with low-res"
        print(f"{n:>4}{default_tokens:>16}{low_res_tokens:>16}{ratio:>10.2f}{verdict:>30}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ns", type=str, required=True)
    args = parser.parse_args()
    run([int(x) for x in args.ns.split(",")])

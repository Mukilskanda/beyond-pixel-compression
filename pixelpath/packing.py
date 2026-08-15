"""Provider tiling cost model — turns region selection from plain knapsack into
geometric packing (plan §Stage 3b: "token cost is not linear in area").

STUB — week 5-6 work. Fill in once a provider's actual tiling grid is known
(inspect billed input_tokens vs. image dimensions from real API responses in
harness/results.sqlite). Until then, pixelpath/regions.py uses a flat linear
cost-per-cell placeholder.

VALIDATED AGAINST REAL DATA (2026-08-16, see experiments/packing_validation.py
and tests/fixtures/gemini_token_cost_2026-08-16.json) — NOT YET FIXED, logic
below is unchanged pending a deliberate follow-up: Gemini 2.5 Flash's real
image token cost is FLAT (258 tokens) across all tested dimensions, 100x100
through 3000x4000 pixels — it does NOT scale with pixel area or tile count in
this range. tile_cost() below assumes a step function that grows with image
size (an Anthropic-style formula per the comment below, never actually
calibrated against Anthropic either). The mismatch against Gemini is large
(diffs of -98% to -559%) and completely systematic, not noisy — the real
numbers are the same across replicate real API calls at wildly different
sizes. This means the cost-model SHAPE is wrong for Gemini specifically, not
just its constants. Do not tune tile_size/base_tokens to fit Gemini's numbers
without changing the functional form (Gemini's real behavior isn't a step
function of tile count at all, at least not within the range tested) —
whatever replaces this needs to be re-derived from Gemini's actual documented
image tokenization, and separately re-validated for Anthropic if that's used
for the paper's headline numbers, since this file has never been checked
against either provider's real numbers before now.
"""
from __future__ import annotations


def tile_cost(width: int, height: int, tile_size: int = 512, base_tokens: int = 85) -> int:
    """Placeholder step-function cost model: providers bill per tile, not per pixel.

    Replace `tile_size`/`base_tokens` with the target provider's actual documented
    tiling scheme (e.g. Anthropic bills roughly (w*h)/750 tokens for images —
    verify against real usage.input_tokens before trusting this).
    """
    tiles_x = -(-width // tile_size)   # ceil div
    tiles_y = -(-height // tile_size)
    return tiles_x * tiles_y * base_tokens


def merge_savings(region_a, region_b, tile_size: int = 512, base_tokens: int = 85) -> int:
    """Cost delta (positive = savings) of sending regions as one merged crop
    vs. two separate crops. Region merging is only worth it when the merged
    bounding box doesn't cross enough extra tile boundaries to offset the win.
    """
    separate = (
        tile_cost(region_a.w, region_a.h, tile_size, base_tokens)
        + tile_cost(region_b.w, region_b.h, tile_size, base_tokens)
    )
    merged_w = max(region_a.x + region_a.w, region_b.x + region_b.w) - min(region_a.x, region_b.x)
    merged_h = max(region_a.y + region_a.h, region_b.y + region_b.h) - min(region_a.y, region_b.y)
    merged = tile_cost(merged_w, merged_h, tile_size, base_tokens)
    return separate - merged

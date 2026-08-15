"""Provider tiling cost model — turns region selection from plain knapsack into
geometric packing (plan §Stage 3b: "token cost is not linear in area").

STUB — week 5-6 work. Fill in once a provider's actual tiling grid is known
(inspect billed input_tokens vs. image dimensions from real API responses in
harness/results.sqlite). Until then, pixelpath/regions.py uses a flat linear
cost-per-cell placeholder.
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

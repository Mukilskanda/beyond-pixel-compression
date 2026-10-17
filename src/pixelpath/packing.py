"""Provider tiling cost model — turns region selection from plain knapsack into
geometric packing (plan §Stage 3b: "token cost is not linear in area").

STUB — week 5-6 work. Fill in once a provider's actual tiling grid is known
(inspect billed input_tokens vs. image dimensions from real API responses in
harness/results.sqlite). Until then, pixelpath/regions.py uses a flat linear
cost-per-cell placeholder.

VALIDATED AGAINST REAL DATA (2026-08-16, see experiments/packing_validation.py
and tests/fixtures/gemini_token_cost_2026-08-16.json). tile_cost() below is a
GENERIC step-function placeholder — kept as-is, unvalidated against any real
provider, since it was originally written with an Anthropic-style formula in
mind (never actually checked against Anthropic either) and should not be
silently overwritten by a Gemini-specific finding. See
gemini_2_5_flash_image_cost() below for the Gemini-specific replacement,
which should NOT be assumed to generalize to any other provider — see its
own docstring for the mechanism investigation and why.
"""

from __future__ import annotations

# Gemini 2.5 Flash's real image token cost, empirically confirmed flat.
GEMINI_2_5_FLASH_FLAT_IMAGE_TOKENS = 258


def gemini_2_5_flash_image_cost(width: int, height: int) -> int:
    """Gemini 2.5 Flash's real image token cost — explicitly Gemini-specific,
    do not assume this generalizes to any other provider (see below).

    Mechanism investigation (2026-08-16, experiments/packing_validation.py):
    Google's public docs (ai.google.dev/gemini-api/docs/tokens and
    /image-understanding) state a tiered formula — "images <=384px in both
    dimensions count as 258 tokens; larger images are tiled into 768x768
    pixel tiles, each costing 258 tokens" — which predicts token cost SHOULD
    scale with image size above 384px (e.g. an 8000x8000 image should tile
    into roughly 11x11=121 tiles at 258 tokens each, ~31,218 tokens).

    Empirically, real gemini-2.5-flash calls returned a flat 258 image
    tokens (266 total minus an 8-token fixed prompt) across every size
    tested: 10x10, 32x32, 100x100, 404x538 up to 1653x2339 (from the original
    DocVQA-derived probe), 3000x4000, and 8000x8000 — 12 real API calls, zero
    variation. Also tested content variation at a fixed 1000x1000 resolution
    (blank white vs. dense random noise): identical 258 tokens for both,
    ruling out a content-adaptive quantization explanation. Ruled out
    client-side resizing as the cause by reading the google-genai SDK source
    directly (types.py's PIL.Image-to-Part conversion calls
    `value.save(bytes_io, ...)` with no resize/thumbnail step — the full
    original-resolution image bytes are what's actually sent).

    Net result: this empirical finding DIRECTLY CONTRADICTS the general
    tiling formula in Google's own public docs. No model-specific
    documentation was found (via WebFetch) that confirms gemini-2.5-flash
    overrides the general formula, so the mechanism is confirmed by data
    (robustly — 8 orders of magnitude in size, two content extremes) but not
    fully explained by public documentation. Plausible-but-unconfirmed
    explanations: gemini-2.5-flash may default to a fixed low-resolution
    internal representation unless a `media_resolution` parameter (seen
    referenced for newer Gemini models) is set explicitly; or the public
    tiling-formula docs may not reflect 2.5-era models' actual default
    behavior. Do not state either as fact without further verification.

    CROSS-PROVIDER CAVEAT: this is very likely provider-specific, possibly
    even specific to gemini-2.5-flash's default configuration. If this
    project adds a second real provider, re-run
    experiments/packing_validation.py's size/content sweep against it before
    assuming this flat-cost behavior applies — if a provider's real cost
    genuinely does not vary with image size or content, region-selection and
    budget-allocation (this entire project's core mechanism) may not reduce
    that provider's real cost AT ALL, regardless of how well-chosen the
    regions are. That would be a fundamental, provider-dependent limit on
    this project's applicability, not a tuning problem — worth checking
    early against any new provider, not assumed to be fine.
    """
    return GEMINI_2_5_FLASH_FLAT_IMAGE_TOKENS


def tile_cost(width: int, height: int, tile_size: int = 512, base_tokens: int = 85) -> int:
    """Placeholder step-function cost model: providers bill per tile, not per pixel.

    Replace `tile_size`/`base_tokens` with the target provider's actual documented
    tiling scheme (e.g. Anthropic bills roughly (w*h)/750 tokens for images —
    verify against real usage.input_tokens before trusting this).
    """
    tiles_x = -(-width // tile_size)  # ceil div
    tiles_y = -(-height // tile_size)
    return tiles_x * tiles_y * base_tokens


def merge_savings(region_a, region_b, tile_size: int = 512, base_tokens: int = 85) -> int:
    """Cost delta (positive = savings) of sending regions as one merged crop
    vs. two separate crops. Region merging is only worth it when the merged
    bounding box doesn't cross enough extra tile boundaries to offset the win.
    """
    separate = tile_cost(region_a.w, region_a.h, tile_size, base_tokens) + tile_cost(
        region_b.w, region_b.h, tile_size, base_tokens
    )
    merged_w = max(region_a.x + region_a.w, region_b.x + region_b.w) - min(region_a.x, region_b.x)
    merged_h = max(region_a.y + region_a.h, region_b.y + region_b.h) - min(region_a.y, region_b.y)
    merged = tile_cost(merged_w, merged_h, tile_size, base_tokens)
    return separate - merged

"""Stage 3 Assembler: profiler -> router -> text/pixel path -> final API payload.

This is "system v1" from the plan (§7 timeline, week 7): the rule-based router
wired to both paths end to end. Each function in isolation was already smoke-
tested; this ties them together and is the thing baselines get compared against.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from baselines.baselines import unreduced
from profiler.budget import allocate_budget
from profiler.features import ProfileFeatures, profile_image
from router.rules import Route, route


@dataclass
class Payload:
    route: Route
    image_bytes: bytes | None  # None for pure text-path (zero pixels sent)
    text_context: str | None  # OCR markdown, populated for text/hybrid routes
    features: ProfileFeatures
    pixel_budget: int  # actual budget used (adaptive or the caller's override) — exposed for logging/ablation


def build_payload(image_path: str | Path, query: str, pixel_budget: int | None = None) -> Payload:
    """Run the full v1 pipeline and produce whatever goes into the actual API call.

    `pixel_budget`: None (default) uses profiler/budget.py's adaptive
    per-image allocation. Pass an int to override with a fixed budget — this
    is the fixed_budget ablation's hook (experiments/ablations.yaml). Prior
    to this, the default was a hardcoded 400 for every image regardless of
    content, which is exactly the "fixed pruning ratio" problem the plan
    (and budget.py's own docstring) is meant to solve — budget.py existed
    and was tested in isolation but was never actually wired into the
    pipeline that uses it.
    """
    features = profile_image(image_path)
    chosen_route = route(features, query)
    resolved_budget = pixel_budget if pixel_budget is not None else allocate_budget(features)

    if chosen_route == Route.TEXT:
        from textpath.ocr import image_to_markdown
        ocr_result = image_to_markdown(image_path)
        return Payload(route=chosen_route, image_bytes=None, text_context=ocr_result.markdown, features=features, pixel_budget=resolved_budget)

    if chosen_route == Route.HYBRID:
        from textpath.ocr import image_to_markdown
        from pixelpath.regions import propose_regions, greedy_select

        ocr_result = image_to_markdown(image_path)
        regions = propose_regions(image_path)
        selected = greedy_select(regions, budget=resolved_budget)
        # v1: hybrid sends the OCR text plus a low-res full image, per plan
        # §Stage 2 ("often the best fidelity/cost point"). Selected regions
        # from greedy_select are computed but not yet used to crop — that's
        # the region-merging/packing upgrade (pixelpath/packing.py, week 6).
        from baselines.baselines import uniform_downscale
        low_res = uniform_downscale(image_path, scale=0.5)
        return Payload(route=chosen_route, image_bytes=low_res, text_context=ocr_result.markdown, features=features, pixel_budget=resolved_budget)

    # Route.PIXEL
    from pixelpath.regions import propose_regions, greedy_select

    regions = propose_regions(image_path)
    selected = greedy_select(regions, budget=resolved_budget)
    # v1: send full image; cropping to `selected` regions is the packing.py upgrade.
    image_bytes = unreduced(image_path)
    return Payload(route=chosen_route, image_bytes=image_bytes, text_context=None, features=features, pixel_budget=resolved_budget)

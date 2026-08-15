"""Stage 3 Assembler: profiler -> router -> text/pixel path -> final API payload.

This is "system v1" from the plan (§7 timeline, week 7): the rule-based router
wired to both paths end to end. Each function in isolation was already smoke-
tested; this ties them together and is the thing baselines get compared against.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from baselines.baselines import unreduced
from profiler.features import ProfileFeatures, profile_image
from router.rules import Route, route


@dataclass
class Payload:
    route: Route
    image_bytes: bytes | None  # None for pure text-path (zero pixels sent)
    text_context: str | None  # OCR markdown, populated for text/hybrid routes
    features: ProfileFeatures


def build_payload(image_path: str | Path, query: str, pixel_budget: int = 400) -> Payload:
    """Run the full v1 pipeline and produce whatever goes into the actual API call."""
    features = profile_image(image_path)
    chosen_route = route(features, query)

    if chosen_route == Route.TEXT:
        from textpath.ocr import image_to_markdown
        ocr_result = image_to_markdown(image_path)
        return Payload(route=chosen_route, image_bytes=None, text_context=ocr_result.markdown, features=features)

    if chosen_route == Route.HYBRID:
        from textpath.ocr import image_to_markdown
        from pixelpath.regions import propose_regions, greedy_select

        ocr_result = image_to_markdown(image_path)
        regions = propose_regions(image_path)
        selected = greedy_select(regions, budget=pixel_budget)
        # v1: hybrid sends the OCR text plus a low-res full image, per plan
        # §Stage 2 ("often the best fidelity/cost point"). Selected regions
        # from greedy_select are computed but not yet used to crop — that's
        # the region-merging/packing upgrade (pixelpath/packing.py, week 6).
        from baselines.baselines import uniform_downscale
        low_res = uniform_downscale(image_path, scale=0.5)
        return Payload(route=chosen_route, image_bytes=low_res, text_context=ocr_result.markdown, features=features)

    # Route.PIXEL
    from pixelpath.regions import propose_regions, greedy_select

    regions = propose_regions(image_path)
    selected = greedy_select(regions, budget=pixel_budget)
    # v1: send full image; cropping to `selected` regions is the packing.py upgrade.
    image_bytes = unreduced(image_path)
    return Payload(route=chosen_route, image_bytes=image_bytes, text_context=None, features=features)

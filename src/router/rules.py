"""Stage 2 router — v1 hand-tuned rule baseline. The learned router (week 9) must beat this."""

from __future__ import annotations

from enum import Enum

from profiler.features import ProfileFeatures


class Route(str, Enum):
    TEXT = "text"
    PIXEL = "pixel"
    HYBRID = "hybrid"


# Query keywords that indicate spatial/counting intent — never route these to text-only,
# since OCR discards layout and exact positions.
SPATIAL_INTENT_KEYWORDS = (
    "count",
    "how many",
    "where is",
    "where are",
    "left of",
    "right of",
    "above",
    "below",
    "top of",
    "bottom of",
    "color of",
    "shape of",
)


def _has_spatial_intent(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in SPATIAL_INTENT_KEYWORDS)


def route(features: ProfileFeatures, query: str, query_conditioning: bool = True) -> Route:
    """Hand-tuned v1 rules. Thresholds calibrated against data/synthetic.py fixtures
    (see profiler feature ranges there) — MUST be re-tuned against real dev-set
    labels in week 7 (§7 timeline). Real scanned documents/screenshots will have
    different absolute edge-density values than these synthetic renders.

    Note on text_density: it's global Canny edge density over the whole image,
    which is a weaker proxy than the OCR-confidence alternative the plan lists
    (§Stage 1 Profiler table) — it responds to "high local gradient," not
    "text-like structure." Per-pixel random noise pathologically maximizes it
    (worse than real text), which is why data/synthetic.py generates smooth,
    spatially-correlated photo fixtures rather than literal noise. Swap in an
    OCR-confidence-based measure once Tesseract is available if this keeps
    misfiring against real data — see plan's stated Stage-1 alternative.

    `query_conditioning=False` is the ablation switch (plan §3.5): ignore the
    query text entirely and route on image features alone, to measure whether
    query-awareness (the spatial-intent override below) is pulling its weight.

    GAP FIXED (re-tuned against 40 real DocVQA scans, not just synthetic):
    the previous whitespace_ratio > 0.85 bar was implicitly tuned against a
    synthetic fixture with huge blank margins (whitespace ~0.92) and never
    fired on real scans, whose measured distribution over these 40 samples
    is text_density in [0.009, 0.215] (median 0.048) and whitespace_ratio in
    [0.004, 0.773] (median 0.270) — every real sample fell through to
    HYBRID/PIXEL, meaning the image (flat-cost per §2.4/§9.4, but nonzero)
    was paid for on every single real request regardless of content.

    IMPORTANT — what this fix does and does not claim: this project's own
    real-billing measurements (§9.4) only ever tested `ocr_everything` (OCR
    text ADDED on top of the image) and pixel-space baselines (image ALONE)
    — a pure TEXT route that replaces the image with `image_bytes=None`
    (harness/pipeline.py's TEXT branch) has never been measured against real
    billing. Enabling TEXT to actually fire is what makes that measurement
    possible; it is not itself proof of savings. The threshold below
    (whitespace_ratio > 0.40) is chosen by inspecting the real 40-sample
    distribution — it separates a visibly sparser subgroup (13/40 samples,
    whitespace_ratio 0.40-0.77, mostly low text_density too) from the denser
    majority — not by any fidelity outcome, since no fidelity labels exist
    for these samples yet. This is still a heuristic pending the real
    week-7 process (tune against measured ANLS/exact-match once TEXT-route
    fidelity data exists) — just a heuristic now grounded in the real
    feature distribution instead of one that structurally never fires.
    """
    if query_conditioning and _has_spatial_intent(query):
        # Query needs exact spatial grounding; text path would discard it.
        return Route.PIXEL if features.text_density < 0.15 else Route.HYBRID

    if features.text_density > 0.006 and features.whitespace_ratio > 0.40:
        # Document signature: some text edges, and a sparser/lighter layout
        # than the HYBRID-territory majority (see real-data note above).
        return Route.TEXT

    if features.text_density > 0.008:
        # Enough text-like edge signal to be worth OCR, but not enough
        # whitespace to be a clean document — chart/table/screenshot territory.
        # (layout_regularity/Hough-lines is unreliable at synthetic-fixture
        # scale — omitted from this rule rather than gated on a noisy signal.)
        return Route.HYBRID

    return Route.PIXEL

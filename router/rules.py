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
    "count", "how many", "where is", "where are", "left of", "right of",
    "above", "below", "top of", "bottom of", "color of", "shape of",
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

    KNOWN GAP (confirmed against 5 real DocVQA scans, not just synthetic):
    real scanned documents currently route HYBRID, never TEXT, because the
    whitespace_ratio > 0.85 bar was implicitly tuned against a synthetic
    fixture with huge blank margins (whitespace ~0.92). Real business-letter-
    style scans measured 0.004-0.18 — genuinely denser, no huge margins. This
    contradicts the plan's stated expectation that "text path should dominate"
    on DocVQA (§3.1). HYBRID isn't wrong per se (plan §Stage 2 calls it "often
    the best fidelity/cost point"), but it's not what was expected here.
    Deliberately left unpatched rather than threshold-tuned against 5 examples
    — real calibration needs the week-7 process (tune against measured
    fidelity/cost outcomes, not eyeballed feature values). If you're reading
    this in week 7, this is the first thing to fix with real labeled data.
    """
    if query_conditioning and _has_spatial_intent(query):
        # Query needs exact spatial grounding; text path would discard it.
        return Route.PIXEL if features.text_density < 0.15 else Route.HYBRID

    if features.text_density > 0.006 and features.whitespace_ratio > 0.85:
        # Document signature: some text edges, and mostly uniform (white) background.
        return Route.TEXT

    if features.text_density > 0.008:
        # Enough text-like edge signal to be worth OCR, but not enough
        # whitespace to be a clean document — chart/table/screenshot territory.
        # (layout_regularity/Hough-lines is unreliable at synthetic-fixture
        # scale — omitted from this rule rather than gated on a noisy signal.)
        return Route.HYBRID

    return Route.PIXEL

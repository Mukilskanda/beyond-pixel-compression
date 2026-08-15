import shutil

import pytest

import harness.pipeline as pipeline_module
from harness.pipeline import Payload, build_payload
from router.rules import Route

TESSERACT_AVAILABLE = shutil.which("tesseract") is not None
requires_tesseract = pytest.mark.skipif(
    not TESSERACT_AVAILABLE, reason="tesseract binary not installed (see README setup)"
)


def test_build_payload_pixel_route_has_image_bytes(photo_like_image):
    if not TESSERACT_AVAILABLE:
        pytest.skip("route for this fixture may resolve to TEXT/HYBRID, which needs tesseract")
    payload = build_payload(photo_like_image, "describe this image")
    assert isinstance(payload, Payload)
    if payload.route == Route.PIXEL:
        assert payload.image_bytes is not None
        assert payload.text_context is None


def test_build_payload_pure_pixel_route_never_calls_ocr(photo_like_image, monkeypatch):
    """Force the PIXEL branch directly (no router dependency) and confirm it
    never touches textpath — this must pass with or without tesseract installed."""
    monkeypatch.setattr(pipeline_module, "route", lambda features, query: Route.PIXEL)
    payload = build_payload(photo_like_image, "describe this image")
    assert payload.route == Route.PIXEL
    assert payload.image_bytes is not None
    assert payload.text_context is None


def test_build_payload_always_returns_features(sample_image, monkeypatch):
    monkeypatch.setattr(pipeline_module, "route", lambda features, query: Route.PIXEL)
    payload = build_payload(sample_image, "what does this say?")
    assert payload.features is not None


def test_build_payload_uses_adaptive_budget_by_default(sample_image, photo_like_image, monkeypatch):
    """Regression test: profiler/budget.py's allocate_budget() must actually
    be called by build_payload() when pixel_budget is not overridden. Prior
    to this, build_payload() had a hardcoded pixel_budget=400 default and
    never called allocate_budget() at all — the adaptive-budget module
    existed and was tested in isolation but was dead code from the
    pipeline's perspective. Verify the resolved budget matches what
    allocate_budget() independently computes for the same features, and that
    it varies across genuinely different images (not coincidentally the same
    fixed number for everything)."""
    from profiler.budget import allocate_budget
    from profiler.features import profile_image

    monkeypatch.setattr(pipeline_module, "route", lambda features, query: Route.PIXEL)

    payload_a = build_payload(sample_image, "irrelevant query")
    payload_b = build_payload(photo_like_image, "irrelevant query")

    assert payload_a.pixel_budget == allocate_budget(profile_image(sample_image))
    assert payload_b.pixel_budget == allocate_budget(profile_image(photo_like_image))


def test_build_payload_explicit_budget_overrides_adaptive(sample_image, monkeypatch):
    """The fixed_budget ablation's hook: passing pixel_budget explicitly must
    bypass adaptive allocation entirely."""
    monkeypatch.setattr(pipeline_module, "route", lambda features, query: Route.PIXEL)
    payload = build_payload(sample_image, "irrelevant query", pixel_budget=999)
    assert payload.pixel_budget == 999


@requires_tesseract
def test_build_payload_text_route_has_no_image_bytes(sample_image, monkeypatch):
    monkeypatch.setattr(pipeline_module, "route", lambda features, query: Route.TEXT)
    payload = build_payload(sample_image, "what does this say?")
    assert payload.route == Route.TEXT
    assert payload.image_bytes is None
    assert payload.text_context is not None

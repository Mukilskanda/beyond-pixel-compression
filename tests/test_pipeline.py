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


@requires_tesseract
def test_build_payload_text_route_has_no_image_bytes(sample_image, monkeypatch):
    monkeypatch.setattr(pipeline_module, "route", lambda features, query: Route.TEXT)
    payload = build_payload(sample_image, "what does this say?")
    assert payload.route == Route.TEXT
    assert payload.image_bytes is None
    assert payload.text_context is not None

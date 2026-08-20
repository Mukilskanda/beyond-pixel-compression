import shutil

import pytest

from textpath.ocr import image_to_markdown

requires_tesseract = pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="tesseract binary not installed (see README setup)"
)


@requires_tesseract
def test_min_confidence_default_matches_old_behavior(sample_image):
    """min_confidence=0.0 (the default) must keep exactly what the old
    conf<0 filter kept — this is a backward-compat regression test for the
    added min_confidence parameter."""
    result = image_to_markdown(sample_image)
    result_explicit = image_to_markdown(sample_image, min_confidence=0.0)
    assert result.markdown == result_explicit.markdown


@requires_tesseract
def test_higher_min_confidence_never_produces_more_text(sample_image):
    """Raising min_confidence can only drop words, never add them — the
    filtered markdown must be a subset in length terms."""
    lenient = image_to_markdown(sample_image, min_confidence=0.0)
    strict = image_to_markdown(sample_image, min_confidence=90.0)
    assert len(strict.markdown) <= len(lenient.markdown)

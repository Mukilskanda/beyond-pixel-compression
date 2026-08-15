import shutil

import pytest

from baselines.ocr_everything import ocr_everything

requires_tesseract = pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="tesseract binary not installed (see README setup)"
)


@requires_tesseract
def test_ocr_everything_returns_ocr_result(sample_image):
    result = ocr_everything(sample_image)
    assert result.markdown  # sample_image has real text, should extract something


@requires_tesseract
def test_ocr_everything_runs_even_on_photo(photo_like_image):
    """Deliberately the wrong tool for this image — but it must not crash;
    that's the point of baseline 5 (informative failure, not an error)."""
    result = ocr_everything(photo_like_image)
    assert isinstance(result.markdown, str)  # may well be empty/garbage — that's expected

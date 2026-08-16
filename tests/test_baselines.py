from PIL import Image

from baselines.baselines import (
    GEMINI_LOW_DETAIL_SETTINGS,
    fixed_ratio_crop,
    jpeg_quality_reduction,
    uniform_downscale,
    unreduced,
)


def test_unreduced_returns_original_bytes(sample_image):
    from pathlib import Path

    assert unreduced(sample_image) == Path(sample_image).read_bytes()


def test_downscale_reduces_dimensions(sample_image):
    import io

    original = Image.open(sample_image)
    result_bytes = uniform_downscale(sample_image, scale=0.5)
    resized = Image.open(io.BytesIO(result_bytes))

    assert resized.width == original.width // 2
    assert resized.height == original.height // 2


def test_downscale_reduces_byte_size(sample_image):
    from pathlib import Path

    original_size = len(Path(sample_image).read_bytes())
    downscaled_size = len(uniform_downscale(sample_image, scale=0.5))
    assert downscaled_size < original_size


def test_jpeg_quality_reduction_produces_valid_jpeg(sample_image):
    import io

    result_bytes = jpeg_quality_reduction(sample_image, quality=30)
    img = Image.open(io.BytesIO(result_bytes))
    assert img.format == "JPEG"


def test_fixed_ratio_crop_smaller_than_original(sample_image):
    import io

    original = Image.open(sample_image)
    cropped_bytes = fixed_ratio_crop(sample_image, keep_fraction=0.5)
    cropped = Image.open(io.BytesIO(cropped_bytes))

    assert cropped.width < original.width
    assert cropped.height < original.height


def test_gemini_low_detail_settings_shape():
    """Baseline 4 (provider low-detail mode) is a request-time settings dict,
    not a pixel transform — verify it has the shape GeminiProvider.call()
    expects to pass through to GenerateContentConfig. Real-cost verification
    (258 -> 74 tokens on a 1000x1000 image, confirmed through GeminiProvider
    itself) was done with a live API call and is not re-run here to avoid
    spending free-tier quota on every test run — see baselines.py's comment
    and paper/draft.md for the real numbers."""
    assert GEMINI_LOW_DETAIL_SETTINGS == {"media_resolution": "MEDIA_RESOLUTION_LOW"}

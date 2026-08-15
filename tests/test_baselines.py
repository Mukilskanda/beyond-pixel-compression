from PIL import Image

from baselines.baselines import (
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

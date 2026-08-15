import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample_image(tmp_path) -> str:
    """A synthetic document-like image: text block + a colored region."""
    img = Image.new("RGB", (800, 600), "white")
    d = ImageDraw.Draw(img)
    d.text((50, 50), "Invoice #12345", fill="black")
    d.text((50, 80), "Total: $500.00", fill="black")
    d.text((50, 110), "Date: 2026-01-01", fill="black")
    d.rectangle([400, 300, 700, 500], fill="blue")
    path = tmp_path / "sample.png"
    img.save(path)
    return str(path)


@pytest.fixture
def photo_like_image(tmp_path) -> str:
    """A synthetic photo-like image: no text, high color variance."""
    import numpy as np

    rng = np.random.default_rng(42)
    arr = rng.integers(0, 255, (400, 400, 3), dtype="uint8")
    img = Image.fromarray(arr)
    path = tmp_path / "photo.png"
    img.save(path)
    return str(path)

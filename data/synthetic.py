"""Synthetic multi-category dataset generator — lets the pipeline be validated
across the plan's §3.1 failure spectrum without waiting on real dataset
downloads/registration. NOT a substitute for real benchmarks in the paper's
numbers — use only for harness/pipeline development and CI.

Categories mirror plan §3.1:
    document   -> DocVQA stand-in: dense text, white background
    chart      -> ChartQA stand-in: bars + axis labels
    screenshot -> UI stand-in: high whitespace, grid layout, buttons
    photo      -> VQAv2/GQA stand-in: high color variance, no text
    counting   -> TallyQA stand-in: deliberate failure case, N discrete objects
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# PIL's default font is a tiny ~11px bitmap font — its edge density is nothing
# like real document text and silently breaks profiler calibration (text_density
# reads near-zero even on "dense text" synthetic samples). Use a real TrueType
# font at a realistic size so profiler/router sanity checks mean something.
_FONT_CANDIDATES = ["C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Helvetica.ttc"]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()  # last resort — will under-trigger text_density


_FONT = _load_font(22)
_FONT_SMALL = _load_font(16)


@dataclass
class SyntheticSample:
    category: str
    image_path: Path
    question: str
    answers: list[str]


def _make_document(path: Path, rng: random.Random) -> tuple[str, list[str]]:
    img = Image.new("RGB", (800, 1000), "white")
    d = ImageDraw.Draw(img)
    total = rng.randint(100, 9999)
    lines = [
        f"Invoice #{rng.randint(10000, 99999)}",
        "Bill To: Acme Corp",
        f"Date: 2026-0{rng.randint(1,9)}-{rng.randint(10,28)}",
        "",
        "Item          Qty   Price",
        "Widget A       3    $10.00",
        "Widget B       2    $25.00",
        "",
        f"Total: ${total}.00",
    ]
    for i, line in enumerate(lines):
        d.text((60, 60 + i * 30), line, fill="black", font=_FONT)
    img.save(path)
    return "What is the total?", [f"${total}.00", str(total)]


def _make_chart(path: Path, rng: random.Random) -> tuple[str, list[str]]:
    img = Image.new("RGB", (800, 600), "white")
    d = ImageDraw.Draw(img)
    values = [rng.randint(20, 200) for _ in range(5)]
    max_v = max(values)
    bar_w = 100
    colors = ["red", "blue", "green", "orange", "purple"]
    for i, v in enumerate(values):
        x0 = 80 + i * (bar_w + 20)
        h = int(v / max_v * 400)
        d.rectangle([x0, 500 - h, x0 + bar_w, 500], fill=colors[i])
        d.text((x0, 505), f"Q{i+1}", fill="black", font=_FONT)
    d.line([50, 500, 750, 500], fill="black", width=2)
    tallest_idx = values.index(max(values))
    img.save(path)
    return "Which quarter had the highest value?", [f"Q{tallest_idx+1}"]


def _make_screenshot(path: Path, rng: random.Random) -> tuple[str, list[str]]:
    img = Image.new("RGB", (1200, 800), "#f0f0f0")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 1200, 60], fill="#2c3e50")
    d.text((20, 20), "MyApp Dashboard", fill="white", font=_FONT)
    n_buttons = rng.randint(3, 6)
    for i in range(n_buttons):
        y = 100 + i * 80
        d.rectangle([40, y, 300, y + 50], fill="#3498db")
        d.text((60, y + 15), f"Button {i+1}", fill="white", font=_FONT_SMALL)
    img.save(path)
    return "How many buttons are visible?", [str(n_buttons)]


def _make_photo(path: Path, rng: random.Random) -> tuple[str, list[str]]:
    """Smooth, spatially-correlated blobs — NOT per-pixel noise. Per-pixel
    random noise has no spatial correlation and pathologically maximizes
    Canny edge density (real photos are smooth almost everywhere), which
    made it score as more "text-like" than actual text under naive
    edge-density profiling. Real photos are locally smooth with a few
    strong edges at object boundaries — blurred blobs approximate that."""
    import numpy as np
    from PIL import ImageFilter

    seed = rng.randint(0, 1_000_000)
    local_rng = np.random.default_rng(seed)
    arr = np.zeros((500, 500, 3), dtype="uint8")
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    base_color = tuple(int(c) for c in local_rng.integers(60, 200, 3))
    d.rectangle([0, 0, 500, 500], fill=base_color)
    n_blobs = 6
    for _ in range(n_blobs):
        cx, cy = local_rng.integers(50, 450, 2)
        r = int(local_rng.integers(40, 120))
        color = tuple(int(c) for c in local_rng.integers(0, 255, 3))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    img = img.filter(ImageFilter.GaussianBlur(radius=8))  # smooth, correlated — like a real photo
    img.save(path)
    return "Describe the dominant colors in this image.", ["unknown"]  # no ground truth for a synthetic photo


def _make_counting(path: Path, rng: random.Random) -> tuple[str, list[str]]:
    """Deliberate failure case per plan §3.1 — pixel path only, small objects,
    exact count matters. Include this category specifically because it's
    expected to expose reduction methods' weaknesses."""
    img = Image.new("RGB", (800, 800), "white")
    d = ImageDraw.Draw(img)
    n = rng.randint(5, 25)
    for _ in range(n):
        x, y = rng.randint(20, 760), rng.randint(20, 760)
        r = rng.randint(8, 15)
        d.ellipse([x - r, y - r, x + r, y + r], fill="black")
    img.save(path)
    return "How many circles are in this image?", [str(n)]


_GENERATORS = {
    "document": _make_document,
    "chart": _make_chart,
    "screenshot": _make_screenshot,
    "photo": _make_photo,
    "counting": _make_counting,
}


def generate_dataset(
    output_dir: str | Path = "data/synthetic",
    per_category: int = 5,
    seed: int = 0,
) -> list[SyntheticSample]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    samples = []
    for category, gen_fn in _GENERATORS.items():
        cat_dir = output_dir / category
        cat_dir.mkdir(exist_ok=True)
        for i in range(per_category):
            img_path = cat_dir / f"{i:03d}.png"
            question, answers = gen_fn(img_path, rng)
            samples.append(SyntheticSample(category=category, image_path=img_path, question=question, answers=answers))

    return samples


if __name__ == "__main__":
    samples = generate_dataset(per_category=5)
    print(f"Generated {len(samples)} synthetic samples across {len(_GENERATORS)} categories")
    for s in samples[:3]:
        print(f"  [{s.category}] {s.image_path.name}: {s.question!r} -> {s.answers}")

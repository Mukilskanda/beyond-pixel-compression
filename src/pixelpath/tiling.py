"""Multi-image tiling — the lever this project's own flat-cost finding
(§5, §9.4) points to rather than contradicts: if per-image cost is a flat
constant regardless of content or pixel dimensions, then N separate calls
each pay that flat cost N times, but compositing N images into a single
grid and sending ONE call pays it once — a cost lever aimed at image
COUNT, not image content, consistent with (not fighting) the falsification
finding.

Real risk this module does not resolve on its own, stated per the honest-
reporting discipline used throughout this project: a large composite image
may cross a provider's resolution/tile boundary into a MORE expensive cost
bucket, partially or fully erasing the per-call saving. That is an
empirical question for the real-billing mechanism check
(experiments/tiling_mechanism_check.py), not assumed here.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _grid_dims(n: int) -> tuple[int, int]:
    """Pick a (rows, cols) grid as close to square as possible, rows <= cols."""
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def _load_label_font(size: int) -> ImageFont.FreeTypeFont:
    for path in ("C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def tile_images(
    image_paths: list[str | Path],
    tile_size: tuple[int, int] = (600, 800),
    label_tiles: bool = True,
    separator_px: int = 8,
) -> Image.Image:
    """Composite N images into a single grid image, each tile resized to
    `tile_size` (uniform tiles keep grid geometry simple and predictable -
    not claimed to be fidelity-optimal, just a reasonable v1). A thin
    separator line and an optional "Tile N" label are drawn per cell so a
    model asked about "the second image" has a real visual anchor, not
    just an unmarked grid position.
    """
    n = len(image_paths)
    if n == 0:
        raise ValueError("tile_images requires at least one image")

    rows, cols = _grid_dims(n)
    tw, th = tile_size
    label_h = 28 if label_tiles else 0
    cell_w, cell_h = tw + separator_px, th + label_h + separator_px

    grid = Image.new("RGB", (cell_w * cols + separator_px, cell_h * rows + separator_px), "white")
    draw = ImageDraw.Draw(grid)
    font = _load_label_font(20) if label_tiles else None

    for i, path in enumerate(image_paths):
        r, c = divmod(i, cols)
        x0 = separator_px + c * cell_w
        y0 = separator_px + r * cell_h

        if label_tiles:
            draw.text((x0, y0), f"Tile {i + 1}", fill="black", font=font)

        img = Image.open(path).convert("RGB").resize((tw, th), Image.LANCZOS)
        grid.paste(img, (x0, y0 + label_h))

        # Cell border, so tile boundaries are visually unambiguous even
        # without labels.
        draw.rectangle([x0 - 1, y0 - 1, x0 + tw + 1, y0 + label_h + th + 1], outline="black", width=1)

    return grid


def save_tiled(image_paths: list[str | Path], output_path: str | Path, **kwargs) -> Path:
    """Convenience wrapper: tile_images() + save to disk, returns the path."""
    grid = tile_images(image_paths, **kwargs)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    return output_path

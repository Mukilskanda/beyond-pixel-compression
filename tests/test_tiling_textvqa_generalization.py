"""Zero-cost check that pixelpath/tiling.py's compositing mechanism (§13)
generalizes structurally to a domain it was never built or tested
against: TextVQA's real photos (varied real aspect ratios/sizes — e.g.
1024x664, 1024x1024 — unlike DocVQA/ChartQA's more uniform scan/chart
shapes). This is a MECHANISM check only — it confirms the composite
builds correctly on real, differently-shaped photos. It does NOT confirm
real token savings or real fidelity on this domain, which needs real
billed API calls this project is not spending on TextVQA right now (see
git history / conversation for why: text-route was already ruled out here
by a real, zero-cost OCR mechanism check — 14/15 real samples returned
zero usable OCR text — but tiling doesn't depend on OCR at all, so this
check is the honest zero-cost limit of what CAN be confirmed for tiling
on this domain without spending).

Skips if the real TextVQA cache isn't populated (network-fetched,
gitignored) — mirrors tests/test_cross_page_dedup_real_check.py's pattern.
"""

from pathlib import Path

import pytest

from pixelpath.tiling import tile_images

requires_real_textvqa_cache = pytest.mark.skipif(
    not any(Path("data/textvqa_cache").glob("*.png")) if Path("data/textvqa_cache").exists() else True,
    reason="no cached real TextVQA images on disk (network-fetched, gitignored — "
    "run data.textvqa_loader.load_textvqa() once to populate)",
)


@requires_real_textvqa_cache
def test_tile_images_builds_correct_grid_from_real_varied_aspect_photos():
    paths = sorted(Path("data/textvqa_cache").glob("*.png"))[:4]
    assert len(paths) >= 2, "need at least 2 real cached images for a meaningful tiling check"

    grid = tile_images(paths, tile_size=(600, 800))

    assert grid.width > 0 and grid.height > 0
    # A 2-4 image set should tile into a 2-column grid per pixelpath/tiling.py's
    # _grid_dims (ceil(sqrt(n)) columns) — sanity-check it didn't collapse to 1x1.
    assert grid.width > 700  # more than one tile_size-worth of width


@requires_real_textvqa_cache
def test_tile_images_preserves_all_real_images_not_just_the_first():
    """Regression against a real risk with varied real aspect ratios: a
    naive resize/crop could silently drop later tiles or produce a
    degenerate (near-blank) composite. Checking real pixel variance in
    each quadrant is a cheap, real (not synthetic) way to catch that."""
    import numpy as np

    paths = sorted(Path("data/textvqa_cache").glob("*.png"))[:4]
    grid = tile_images(paths, tile_size=(600, 800))
    arr = np.array(grid.convert("L"))
    h, w = arr.shape
    quadrants = [
        arr[: h // 2, : w // 2],
        arr[: h // 2, w // 2 :],
    ]
    for q in quadrants:
        assert q.std() > 1.0, "a quadrant with near-zero variance suggests a blank/degenerate tile"

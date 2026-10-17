"""Stage 1 profiler: cheap local features, target <15ms. Must never call the API."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class ProfileFeatures:
    text_density: float       # edge-density proxy for text presence, 0-1
    whitespace_ratio: float   # fraction of near-uniform grid cells, 0-1
    layout_regularity: float  # fraction of pixels on detected lines, 0-1
    color_entropy: float      # histogram entropy, bits
    spatial_variance: float   # mean Laplacian energy (unnormalized)


def _text_density(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150)
    return float(np.count_nonzero(edges)) / edges.size


def _whitespace_ratio(gray: np.ndarray, grid: int = 16, brightness_threshold: float = 225.0, std_threshold: float = 30.0) -> float:
    """Fraction of grid cells that are blank/background: near-white AND locally
    smooth (low variance).

    First attempt used a fixed `cell.std() < 8.0` cutoff on its own, calibrated
    implicitly against clean synthetic PNGs. That failed on real DocVQA scans —
    scan/compression noise means even genuinely blank margins have std in the
    16-50 range depending on the document, so the fixed-8 threshold called
    almost every real scan "zero whitespace."

    A percentile-relative threshold (tried next) is worse: it's self-referential
    — by construction, ~N% of any image's cells fall below its own Nth
    percentile, so the feature collapses to a constant ~N/100 regardless of
    actual content and carries no signal at all (verified empirically).

    This version gates on BOTH near-white mean intensity (blank paper actually
    is near-white, robust to scan noise which mostly affects local variance,
    not the mean) AND a raised absolute variance threshold (as a floor against
    dense-but-pale content). Still an absolute-threshold heuristic pending
    proper calibration against labeled real data (plan's week 7) — better
    reasoned than either prior attempt, not a final answer.
    """
    h, w = gray.shape
    gh, gw = max(1, h // grid), max(1, w // grid)
    uniform_cells = 0
    total = 0
    for y in range(0, h - gh + 1, gh):
        for x in range(0, w - gw + 1, gw):
            cell = gray[y:y + gh, x:x + gw]
            if cell.mean() >= brightness_threshold and cell.std() < std_threshold:
                uniform_cells += 1
            total += 1
    return uniform_cells / total if total else 0.0


def _layout_regularity(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=gray.shape[1] // 4, maxLineGap=10)
    if lines is None:
        return 0.0
    mask = np.zeros_like(gray)
    for line in lines:
        x1, y1, x2, y2 = np.ravel(line)
        cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, 1)
    return float(np.count_nonzero(mask)) / mask.size


def _color_entropy(img: np.ndarray) -> float:
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = hist.flatten() / hist.sum()
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def _spatial_variance(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def profile_image(image_path: str | Path) -> ProfileFeatures:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return ProfileFeatures(
        text_density=_text_density(gray),
        whitespace_ratio=_whitespace_ratio(gray),
        layout_regularity=_layout_regularity(gray),
        color_entropy=_color_entropy(img),
        spatial_variance=_spatial_variance(gray),
    )

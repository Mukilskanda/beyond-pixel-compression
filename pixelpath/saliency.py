"""Fast saliency via OpenCV spectral residual. Compare against a slower/accurate
method (e.g. U2-Net) per plan §3.5 ablation once the greedy baseline is validated.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def saliency_map(image_path: str | Path, pre_downscale: float = 1.0) -> np.ndarray:
    """Returns a float32 saliency map in [0, 1], same H×W as the input.

    `pre_downscale` (early-vs-late discard ablation, plan literature review
    §1/STAR): when < 1.0, the image is downsampled BEFORE saliency is
    computed — the saliency pass itself works from less information ("early
    discard"). Default 1.0 computes saliency at full resolution and only
    discards information at the later crop/budget-selection step ("late
    discard" — the STAR finding: later-stage discard preserves more useful
    information than committing early). The output is always resized back
    to the original H×W so callers don't need to know which mode ran.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)

    original_h, original_w = img.shape[:2]

    if pre_downscale < 1.0:
        small_w = max(1, int(original_w * pre_downscale))
        small_h = max(1, int(original_h * pre_downscale))
        img = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)

    detector = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, sal_map = detector.computeSaliency(img)
    if not success:
        raise RuntimeError(f"Saliency computation failed for {image_path}")

    sal_map = sal_map.astype(np.float32)

    if pre_downscale < 1.0:
        sal_map = cv2.resize(sal_map, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

    return sal_map

"""Fast saliency via OpenCV spectral residual. Compare against a slower/accurate
method (e.g. U2-Net) per plan §3.5 ablation once the greedy baseline is validated.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def saliency_map(image_path: str | Path) -> np.ndarray:
    """Returns a float32 saliency map in [0, 1], same H×W as the input."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)

    detector = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, sal_map = detector.computeSaliency(img)
    if not success:
        raise RuntimeError(f"Saliency computation failed for {image_path}")

    return sal_map.astype(np.float32)

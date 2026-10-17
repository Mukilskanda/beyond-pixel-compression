"""Saliency methods for pixelpath. Two OpenCV built-ins compared per plan §3.5:

- "spectral_residual" (default): fast, FFT-based, the v1 baseline.
- "fine_grained": OpenCV's second static-saliency algorithm. NOT the
  heavyweight deep-learning method (e.g. U2-Net) originally envisioned in
  plan §3.5 — that needs pretrained weights downloaded over the network,
  which has been unreliable in this environment, plus a torch dependency
  this project doesn't otherwise need. fine_grained is a real second
  algorithm (different underlying method, not a parameter tweak on the
  same one) that ships in opencv-contrib with zero extra downloads, so it's
  a legitimate fast-vs-fast-but-different comparator now, with the U2-Net
  upgrade still open for whenever network/dependency cost is worth paying.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_DETECTORS = {
    "spectral_residual": cv2.saliency.StaticSaliencySpectralResidual_create,
    "fine_grained": cv2.saliency.StaticSaliencyFineGrained_create,
}


def saliency_map(image_path: str | Path, pre_downscale: float = 1.0, method: str = "spectral_residual") -> np.ndarray:
    """Returns a float32 saliency map in [0, 1], same H×W as the input.

    `pre_downscale` (early-vs-late discard ablation, plan literature review
    §1/STAR): when < 1.0, the image is downsampled BEFORE saliency is
    computed — the saliency pass itself works from less information ("early
    discard"). Default 1.0 computes saliency at full resolution and only
    discards information at the later crop/budget-selection step ("late
    discard" — the STAR finding: later-stage discard preserves more useful
    information than committing early). The output is always resized back
    to the original H×W so callers don't need to know which mode ran.

    `method`: "spectral_residual" (default, fast) or "fine_grained" (second
    OpenCV algorithm — see module docstring for why this isn't U2-Net).
    """
    if method not in _DETECTORS:
        raise ValueError(f"unknown saliency method {method!r}, expected one of {list(_DETECTORS)}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)

    original_h, original_w = img.shape[:2]

    if pre_downscale < 1.0:
        small_w = max(1, int(original_w * pre_downscale))
        small_h = max(1, int(original_h * pre_downscale))
        img = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)

    detector = _DETECTORS[method]()
    success, sal_map = detector.computeSaliency(img)
    if not success:
        raise RuntimeError(f"Saliency computation failed for {image_path}")

    sal_map = sal_map.astype(np.float32)

    if pre_downscale < 1.0:
        sal_map = cv2.resize(sal_map, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

    return sal_map

"""Baseline 5 (plan §3.2): OCR-everything, applied indiscriminately regardless
of router decision — even to photos/charts/counting images where OCR is the
wrong tool. Cheap, and its failure pattern is informative: this is what you'd
get with no router at all, i.e. the ablation "text path only, no routing."
"""

from __future__ import annotations

from pathlib import Path

from textpath.ocr import OCRResult, image_to_markdown


def ocr_everything(image_path: str | Path) -> OCRResult:
    """No routing decision — always OCR, regardless of image content.
    Requires the Tesseract binary; see textpath/ocr.py and README setup.
    """
    return image_to_markdown(image_path)

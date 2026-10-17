"""Stage 3a text path: OCR + layout -> structured markdown, not a raw OCR dump.

Uses Tesseract as the baseline engine (simple, no model download). Swap in
PaddleOCR for better layout/table handling once the pipeline is validated —
see plan §Stage 3a and §3.5 (saliency/OCR engine ablation).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytesseract
from PIL import Image


@dataclass
class OCRResult:
    markdown: str
    word_confidences: list[float]  # per Tesseract; use to compute WER proxy

    @property
    def mean_confidence(self) -> float:
        return sum(self.word_confidences) / len(self.word_confidences) if self.word_confidences else 0.0


def image_to_markdown(image_path: str | Path, min_confidence: float = 0.0) -> OCRResult:
    """Emit reading-order text grouped into paragraphs by vertical gaps.

    This is the naive baseline: line-by-line text with paragraph breaks on
    large vertical whitespace. Table structure detection (PP-Structure or
    LayoutParser) is a week-3/4 upgrade — see plan §Stage 3a.

    `min_confidence` (Tesseract's 0-100 scale) drops low-confidence word
    guesses before they ever reach the prompt — the cheapest place to cut
    text tokens, since a misread word costs the same tokens as a correct one
    but only adds noise. Default 0.0 keeps the old behavior (every word with
    conf >= 0 is kept); see textpath/compress.py for the rest of the
    token-reduction pipeline (dedup, whitespace collapse) that runs after this.
    """
    img = Image.open(image_path)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    lines: dict[int, list[str]] = {}
    confidences: list[float] = []
    prev_bottom: dict[int, int] = {}
    paragraph_breaks: set[int] = set()

    n = len(data["text"])
    for i in range(n):
        word = data["text"][i].strip()
        conf = float(data["conf"][i])
        if not word or conf < max(0.0, min_confidence):
            continue
        line_id = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        top = data["top"][i]
        height = data["height"][i]

        lines.setdefault(hash(line_id), []).append(word)
        confidences.append(conf)

        block_key = data["block_num"][i]
        if block_key in prev_bottom and top - prev_bottom[block_key] > height * 1.5:
            paragraph_breaks.add(hash(line_id))
        prev_bottom[block_key] = top + height

    paragraphs = []
    current: list[str] = []
    for line_hash, words in lines.items():
        if line_hash in paragraph_breaks and current:
            paragraphs.append(" ".join(current))
            current = []
        current.append(" ".join(words))
    if current:
        paragraphs.append(" ".join(current))

    markdown = "\n\n".join(paragraphs)
    return OCRResult(markdown=markdown, word_confidences=confidences)

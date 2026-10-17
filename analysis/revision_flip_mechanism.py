"""INDICON 2026 revision: failure-mechanism label for every DocVQA forward
flip, using an objective rule instead of a free-form review.

For each flip, the OCR text the text route received is regenerated
(textpath.ocr.image_to_markdown + compress_markdown, as in
experiments/text_route_real_validation.py) and searched for each
ground-truth answer after lower-casing and removing whitespace and
punctuation:

  LAYOUT      a ground-truth answer is present in the OCR text, so the
              information was extracted but its position (row, column,
              label) was lost.
  EXTRACTION  no ground-truth answer is present: OCR missed, garbled or
              did not read the text (including logos and pictures).

Zero API calls.
Usage: python experiments/revision_flip_mechanism.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from textpath.compress import compress_markdown
from textpath.ocr import image_to_markdown

FLIPS = Path("results/docvqa_forward_flips.md")


def norm(s: str) -> str:
    return re.sub(r"[\W_]+", "", s.lower())


def main() -> None:
    blocks = FLIPS.read_text(encoding="utf-8").split("\n## ")[1:]
    counts = {"LAYOUT": 0, "EXTRACTION": 0}
    for b in blocks:
        head = b.splitlines()[0]
        img = head.split("|")[1].strip().replace("\\", "/")
        gts = [g.strip() for g in re.search(r"- GT: (.*)", b).group(1).split("|")]
        text = norm(compress_markdown(image_to_markdown(img).markdown))
        hit = [g for g in gts if norm(g) and norm(g) in text]
        label = "LAYOUT" if hit else "EXTRACTION"
        counts[label] += 1
        print(f"{head:<60} {label:<11} gt={gts} found={hit}")
    print(counts)


if __name__ == "__main__":
    main()

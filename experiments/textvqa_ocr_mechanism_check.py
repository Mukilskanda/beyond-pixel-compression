"""Offline mechanism check, zero API cost: before proposing any real spend
to test §12/§13's findings on TextVQA (natural photos, not documents), does
OCR even extract usable text from this domain at all? DocVQA/ChartQA are
"designed to be read" — dense paragraphs or clean axis labels. TextVQA is
real photos where text is incidental (a brand name on a camera, a street
sign at an angle) — real risk that Tesseract's OCR quality collapses here,
which would make text-route mechanically pointless regardless of what
real billing would show. Checking this now, offline, is what stands
between "propose a real-call budget" and "waste it on a domain where the
OCR step was never going to work."

Also checks each image's own OCR text against that sample's real ground-
truth answers (a cheap, zero-cost proxy for "did OCR even capture the
answer-bearing text", not a claim about real API fidelity) — if OCR
routinely fails to capture the literal answer string, that's a strong
signal the API's real answer will also be text-route-unreachable,
independent of tokens/cost.

Usage:
    python experiments/textvqa_ocr_mechanism_check.py --n 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.textvqa_loader import load_textvqa
from textpath.compress import compress_markdown
from textpath.ocr import image_to_markdown


def run(n: int = 15) -> dict:
    samples = load_textvqa(limit=n)

    ocr_lengths = []
    mean_confidences = []
    answer_recall_hits = 0
    per_sample = []

    for s in samples:
        result = image_to_markdown(str(s.image_path))
        compressed = compress_markdown(result.markdown)
        ocr_lower = result.markdown.lower()

        recall_hit = any(a.lower() in ocr_lower for a in s.answers if a.strip())
        if recall_hit:
            answer_recall_hits += 1

        ocr_lengths.append(len(result.markdown))
        mean_confidences.append(result.mean_confidence)
        per_sample.append(
            {
                "question_id": s.question_id,
                "question": s.question,
                "ocr_chars": len(result.markdown),
                "ocr_mean_confidence": round(result.mean_confidence, 1),
                "answer_found_in_ocr_text": recall_hit,
            }
        )

    n_samples = len(samples)
    return {
        "n": n_samples,
        "mean_ocr_chars": round(sum(ocr_lengths) / n_samples, 1) if n_samples else 0,
        "n_empty_ocr": sum(1 for l in ocr_lengths if l == 0),
        "mean_ocr_confidence": round(sum(mean_confidences) / n_samples, 1) if n_samples else 0,
        "answer_recall_rate": round(answer_recall_hits / n_samples, 3) if n_samples else 0,
        "note": (
            "answer_recall_rate is a cheap OCR-quality PROXY (does the literal "
            "ground-truth string appear anywhere in raw OCR text) — not a "
            "fidelity measurement. A real API can still answer correctly from "
            "visual context even when OCR misses the literal string (e.g. "
            "inferring a brand from a logo shape), so a low rate here is "
            "evidence AGAINST text-route working on this domain, not proof."
        ),
        "per_sample": per_sample,
    }


if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=15)
    args = parser.parse_args()
    print(json.dumps(run(args.n), indent=2))

"""Real-data cross-page dedup check — the direct replacement for
`cross_page_dedup_check.py`'s synthetic-fixture caveat. That script sweeps
invented boilerplate/body profiles because no real multi-page-same-document
data was known to be reachable offline; this script uses one, found this
session (`data/mpdocvqa_loader.py`, MP-DocVQA on HuggingFace Hub, real
scanned pages of a real document, zero API cost — Tesseract OCR + tiktoken
only, same as every other offline check in this project).

**The real result, stated plainly rather than buried: on the one real
10-page document checked this session (`ylvj0223`, a real "Annexure to the
Auditor's Report" + Board of Directors report, real OCR text), cross-page
dedup found ZERO cross-page-repeated lines and saved ZERO tokens** —
`compress_pages` is not broken (it works correctly on the synthetic
fixture and on the regression tests), but this specific real document
simply does not carry an exact-repeated boilerplate line across its pages
the way the synthetic fixture assumed one would. Page numbers are visible
in the source images but are page-specific (differ per page, so can't
dedup) or aren't captured as a discrete OCR'd line at all with this
project's line-grouping-by-vertical-gap OCR method (`textpath/ocr.py`).

**This is a real update to this project's honest position on cross-page
dedup, not a rounding error to explain away**: the earlier synthetic
sweep's 1.05x-4.04x range is a real property of the dedup mechanism GIVEN
that documents repeat boilerplate the way the fixture assumed — but the
one real document checked doesn't do that, so the real-world payoff on
this project's only real multi-page sample is 1.0x (no benefit). This
does not prove no real document ever repeats boilerplate exactly (n=1,
one document type, one OCR method) — but it is real evidence against the
assumption being safely generalizable, which is stronger and more honest
than having no real data point at all.

Usage:
    python experiments/cross_page_dedup_real_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import tiktoken

from data.mpdocvqa_loader import load_mpdocvqa_pages
from textpath.compress import compress_markdown, compress_pages
from textpath.ocr import image_to_markdown

_ENC = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def run(doc_id: str | None = None, n_pages: int = 10) -> dict:
    pages = load_mpdocvqa_pages(doc_id=doc_id, n_pages=n_pages)
    if not pages:
        return {"error": "no real pages available (network/cache miss) — see data/mpdocvqa_loader.py"}

    raw_texts = [image_to_markdown(str(p.image_path)).markdown for p in pages]

    no_dedup_tokens = sum(_count_tokens(t) for t in raw_texts)
    within_page = [compress_markdown(t) for t in raw_texts]
    within_page_tokens = sum(_count_tokens(t) for t in within_page)
    cross_page = compress_pages(raw_texts)
    cross_page_tokens = sum(_count_tokens(t) for t in cross_page)

    removed_lines = []
    seen: set[str] = set()
    for i, t in enumerate(raw_texts):
        for line in t.split("\n"):
            line = line.strip()
            if not line:
                continue
            key = line.lower()
            if key in seen:
                removed_lines.append((i, line[:80]))
            else:
                seen.add(key)

    return {
        "doc_id": pages[0].doc_id,
        "n_pages": len(pages),
        "no_dedup_tokens": no_dedup_tokens,
        "within_page_dedup_tokens": within_page_tokens,
        "cross_page_dedup_tokens": cross_page_tokens,
        "cross_page_vs_none": round(no_dedup_tokens / cross_page_tokens, 3) if cross_page_tokens else None,
        "lines_removed_by_cross_page_dedup": len(removed_lines),
        "note": (
            "REAL document, real OCR, real tiktoken counts — not synthetic. "
            "See module docstring for the honest interpretation of a zero "
            "or near-zero result."
        ),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))

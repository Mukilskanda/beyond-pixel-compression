"""PDF ingestion — this project had zero PDF handling before (grep -rli pdf
across the repo returned nothing); everything assumed a raster image.

The token-reduction case for PDFs is stronger than for photos: a native-text
PDF page already has its text embedded losslessly by whatever produced the
PDF. Rasterizing that page to an image and running it through Tesseract
(the old-only path, via textpath/ocr.py) pays the flat image-token cost
(paper/draft.md §5, §9.4) AND introduces OCR noise/errors, when the exact
embedded text could be pulled out directly for a fraction of the tokens and
zero misreads. OCR is only needed as a fallback for scanned/image-only PDF
pages, which have no embedded text layer at all.

Uses PyMuPDF (`pymupdf`, imported as `fitz` for API compat) — already a
project dependency for this reason, not newly justified per-call.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz


@dataclass
class PDFPageResult:
    page_number: int  # 0-indexed
    text: str
    used_ocr: bool  # True if this page had no usable embedded text and was rasterized+OCR'd


def _has_usable_text(text: str, min_chars: int = 10) -> bool:
    """A page with a real embedded text layer will have more than a
    trivial amount of extractable text; scanned/image-only pages return
    empty or near-empty strings from PyMuPDF's text extraction."""
    return len(text.strip()) >= min_chars


def extract_pdf_pages(
    pdf_path: str | Path,
    ocr_min_confidence: float = 0.0,
    rasterize_dpi: int = 200,
) -> list[PDFPageResult]:
    """Extract text per page, preferring the embedded text layer and only
    falling back to rasterize+OCR (textpath/ocr.py) for pages that don't
    have one.

    `ocr_min_confidence` is forwarded to the OCR fallback (see
    textpath/ocr.py's min_confidence parameter, the noise-word filter).
    `rasterize_dpi` controls the fallback render resolution — irrelevant to
    billed cost per this project's own flat-image-cost finding, but affects
    OCR accuracy, so it stays a real parameter rather than hardcoded.
    """
    doc = fitz.open(str(pdf_path))
    results: list[PDFPageResult] = []

    for page_number, page in enumerate(doc):
        embedded_text = page.get_text()
        if _has_usable_text(embedded_text):
            results.append(PDFPageResult(page_number=page_number, text=embedded_text.strip(), used_ocr=False))
            continue

        # Scanned/image-only page: no embedded text layer, fall back to OCR.
        import tempfile

        from textpath.ocr import image_to_markdown

        pix = page.get_pixmap(dpi=rasterize_dpi)
        fd, tmp_path = tempfile.mkstemp(suffix=".png")
        import os

        os.close(fd)  # release the handle before pymupdf/tesseract open the same path (Windows can't share it)
        try:
            pix.save(tmp_path)
            ocr_result = image_to_markdown(tmp_path, min_confidence=ocr_min_confidence)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        results.append(PDFPageResult(page_number=page_number, text=ocr_result.markdown, used_ocr=True))

    doc.close()
    return results

"""Post-OCR text compression — the actual token-reduction lever for this
project (see paper/draft.md's new §13): unlike image tokens, which prior
sessions proved flat/invariant to pixel-space tricks on every provider
tested (§5, §9.4), OCR text tokens scale directly with content, and until
this module existed nothing ever shortened `OCRResult.markdown` before it
reached the prompt (harness/pipeline.py's TEXT/HYBRID branches passed it
through unmodified).

Two independent, composable reductions:
  1. Low-confidence word filtering — handled upstream in
     `textpath.ocr.image_to_markdown(min_confidence=...)`, not here.
  2. Structural compression — this module: collapse redundant blank
     paragraphs and drop duplicate lines (repeated headers/footers/
     watermarks are common on multi-page scans and contribute nothing new
     per repetition).
"""
from __future__ import annotations


def compress_markdown(markdown: str, dedup_lines: bool = True) -> str:
    """Collapse excess blank paragraphs and (optionally) drop lines that are
    exact duplicates of an earlier line in the same document, keeping the
    first occurrence and its position in reading order.

    Line-level, not paragraph-level: two paragraphs can share a duplicate
    header line inside otherwise-distinct content, so dedup is applied to
    the individual lines that make up each paragraph, and empty paragraphs
    left behind by removed lines are dropped.
    """
    seen: set[str] = set()
    kept_paragraphs: list[str] = []

    for paragraph in markdown.split("\n\n"):
        lines = [line.strip() for line in paragraph.split("\n") if line.strip()]
        if dedup_lines:
            kept_lines = []
            for line in lines:
                key = line.lower()
                if key in seen:
                    continue
                seen.add(key)
                kept_lines.append(line)
            lines = kept_lines
        if lines:
            kept_paragraphs.append("\n".join(lines))

    return "\n\n".join(kept_paragraphs)

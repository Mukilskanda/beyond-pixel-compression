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


def _compress_with_seen(markdown: str, seen: set[str], dedup_lines: bool = True) -> str:
    """Shared implementation: dedups lines against `seen`, mutating it in
    place so a caller can carry the set across multiple documents/pages."""
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


def compress_markdown(markdown: str, dedup_lines: bool = True) -> str:
    """Collapse excess blank paragraphs and (optionally) drop lines that are
    exact duplicates of an earlier line in the same document, keeping the
    first occurrence and its position in reading order.

    Line-level, not paragraph-level: two paragraphs can share a duplicate
    header line inside otherwise-distinct content, so dedup is applied to
    the individual lines that make up each paragraph, and empty paragraphs
    left behind by removed lines are dropped.
    """
    return _compress_with_seen(markdown, set(), dedup_lines)


def compress_pages(pages: list[str], dedup_lines: bool = True) -> list[str]:
    """Cross-page variant: dedups a line against every line seen on any
    EARLIER page in the same call, not just earlier in the same page.
    §12.4's real-billing run found `compress_markdown` never saved a single
    token on real DocVQA samples, because DocVQA's pages are independent
    single-page documents with no internal repetition for it to find. This
    function targets the case that check couldn't reach: a genuine
    multi-page document, where a repeated header/footer/watermark line
    would previously survive on every page because each page was compressed
    in isolation (a fresh `seen` set per call). Order matters: page 1 is
    compressed against nothing, page 2 against everything kept from page 1,
    and so on — later pages are the ones that benefit, not earlier ones.

    Carries the same over-deletion risk `compress_markdown` already has, at
    document instead of page scope: any line that's an exact duplicate
    gets dropped from every later page it recurs on, whether it's genuinely
    redundant boilerplate or a coincidentally-identical but meaningful
    value (e.g. a repeated total, a repeated "N/A"). This function does not
    distinguish the two cases — dedup is exact-string, not semantic.
    """
    seen: set[str] = set()
    return [_compress_with_seen(page, seen, dedup_lines) for page in pages]

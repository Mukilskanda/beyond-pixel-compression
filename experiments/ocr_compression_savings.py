"""Offline, zero-API-cost measurement of textpath/compress.py's token
reduction on real OCR output (paper/draft.md's new §13).

Runs OCR (local Tesseract, no network/API) over the 40 real DocVQA samples
already materialized under data/docvqa_cache/ by the router-threshold
calibration step (router/rules.py), then compares raw vs. compressed
markdown using an offline tokenizer (tiktoken cl100k_base — a reasonable
generic proxy, not any specific provider's exact tokenizer, since none of
Gemini/Groq/Anthropic expose one locally). No provider is called. $0 spent.

Usage: python experiments/ocr_compression_savings.py
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tiktoken

from textpath.compress import compress_markdown
from textpath.ocr import image_to_markdown

CACHE_DIR = Path("data/docvqa_cache")
ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))


def main() -> None:
    image_paths = sorted(CACHE_DIR.glob("*.png"))
    if not image_paths:
        raise SystemExit(
            f"No cached samples found under {CACHE_DIR}/ — run "
            "data.docvqa_loader.load_docvqa() first (network, no API cost)."
        )

    raw_counts: list[int] = []
    compressed_counts: list[int] = []

    print(f"{'sample':<12}{'raw_tokens':>12}{'compressed_tokens':>20}{'reduction':>12}")
    for path in image_paths:
        ocr_result = image_to_markdown(path)
        raw = ocr_result.markdown
        compressed = compress_markdown(raw)

        raw_tok = count_tokens(raw)
        comp_tok = count_tokens(compressed)
        raw_counts.append(raw_tok)
        compressed_counts.append(comp_tok)

        reduction = f"{(1 - comp_tok / raw_tok) * 100:.1f}%" if raw_tok else "n/a"
        print(f"{path.stem:<12}{raw_tok:>12}{comp_tok:>20}{reduction:>12}")

    total_raw = sum(raw_counts)
    total_compressed = sum(compressed_counts)
    print()
    print(f"n = {len(image_paths)}")
    print(f"total raw tokens:        {total_raw}")
    print(f"total compressed tokens (dedup only): {total_compressed}")
    if total_raw:
        print(f"aggregate reduction (dedup only): {(1 - total_compressed / total_raw) * 100:.2f}%")
    print(f"median per-sample raw tokens:        {statistics.median(raw_counts)}")
    print(f"median per-sample compressed tokens: {statistics.median(compressed_counts)}")
    print(
        "\nNote: dedup_lines finds ~0% reduction on this corpus because these "
        "are single-page samples with no repeated header/footer lines to "
        "collapse — it's a real lever for multi-page documents, not this one."
    )

    print("\n--- min_confidence word filtering (the lever that actually helps on single pages) ---")
    print(f"{'min_confidence':>15}{'filtered_tokens':>18}{'reduction':>12}")
    for min_conf in (0, 30, 50, 60, 70):
        filtered_total = 0
        for path in image_paths:
            filtered = image_to_markdown(path, min_confidence=min_conf)
            filtered_total += count_tokens(filtered.markdown)
        reduction = f"{(1 - filtered_total / total_raw) * 100:.2f}%" if total_raw else "n/a"
        print(f"{min_conf:>15}{filtered_total:>18}{reduction:>12}")
    print(
        "\nCaveat: this reduction is real and measured offline, but its effect "
        "on answer fidelity is NOT measured here (no real API calls made) — "
        "some filtered words may have been correct despite low OCR confidence. "
        "See plan step 5 for the capped real-call validation that checks this."
    )


if __name__ == "__main__":
    main()

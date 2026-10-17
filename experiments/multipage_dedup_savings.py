"""Multi-page dedup measurement — the case §12.2 identified but could not
test with single-page samples. Uses the 6 real sequential pages of the same
financial report (question_ids 57374-57388) already identified in §11.2 as
sharing an identical scan template/header — concatenates their OCR output
into one multi-page document (the shape textpath/compress.py's dedup_lines
is actually meant for) and measures the real reduction. Local OCR only,
tiktoken for offline counting. $0 spent.

Usage: python experiments/multipage_dedup_savings.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import tiktoken

from textpath.compress import compress_markdown
from textpath.ocr import image_to_markdown

SEQUENTIAL_PAGE_IDS = ["57374", "57375", "57376", "57386", "57387", "57388"]
CACHE_DIR = Path("data/docvqa_cache")
ENCODING = tiktoken.get_encoding("cl100k_base")


def main() -> None:
    pages = []
    for qid in SEQUENTIAL_PAGE_IDS:
        path = CACHE_DIR / f"{qid}.png"
        if not path.exists():
            raise SystemExit(f"{path} not cached — run data.docvqa_loader.load_docvqa(limit=40) first.")
        pages.append(image_to_markdown(path).markdown)

    multipage_raw = "\n\n".join(pages)
    multipage_compressed = compress_markdown(multipage_raw)

    raw_tokens = len(ENCODING.encode(multipage_raw))
    compressed_tokens = len(ENCODING.encode(multipage_compressed))

    print(f"Pages concatenated: {len(pages)} (question_ids {SEQUENTIAL_PAGE_IDS})")
    print(f"Raw multi-page tokens:        {raw_tokens}")
    print(f"Compressed multi-page tokens: {compressed_tokens}")
    print(f"Reduction: {(1 - compressed_tokens / raw_tokens) * 100:.2f}%")
    print()
    print("--- Compressed output (first 600 chars) ---")
    print(multipage_compressed[:600])


if __name__ == "__main__":
    main()

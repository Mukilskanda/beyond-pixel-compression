"""Offline mechanism check for a combination never tested before: §12's
text-route (OCR text instead of the image, 75-95% real reduction on Groq)
has only ever been measured per-page, one call per page. §13.8's batching
(all N questions in ONE call) has only ever been measured for the *image*
route (a tiled composite). Nobody has combined them: N pages, each OCR'd to
text instead of tiled as an image, all N page-texts and N questions sent in
ONE text-only call.

The case for this being stronger than either half alone: §12.4 found the
image line item removable (text route). §13.8 found paying the per-call
image cost N times removable (batching). A per-page-text route already has
no image cost to begin with, so batching it removes a second cost that
doesn't apply here — the N-times *request-overhead* cost (repeated system
prompt/instructions, repeated JSON-formatting boilerplate) is what
batching removes for text, not an image re-payment. Whether that's still
worth it, and by how much, is exactly what this script measures — zero API
spend, offline only.

Method: real DocVQA cache images stand in for PDF pages (OCR is identical
either way — Tesseract doesn't know or care if the source was a raster PDF
page or a photo; textpath/pdf.py's own OCR fallback path calls the exact
same textpath/ocr.py function this script does). Real Tesseract OCR (zero
API cost) + real textpath/compress.py compression, on N real cached
DocVQA images. Token counts are via tiktoken's cl100k_base encoding as an
**offline proxy**, not the real Gemini/Groq tokenizer (neither exposes a
local tokenizer) — stated explicitly as an approximation, not a real-billed
number. Real per-image and real per-call costs from prior sessions
(§12.4, §13.8) are pulled in as fixed reference constants, cited, not
re-measured here.

Usage:
    python experiments/pdf_batched_text_route_check.py --n 4
    python experiments/pdf_batched_text_route_check.py --n 4 --dataset chartqa
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tiktoken

from data.chartqa_loader import load_chartqa
from data.docvqa_loader import load_docvqa
from textpath.compress import compress_markdown
from textpath.ocr import image_to_markdown

DATASETS = {"docvqa": load_docvqa, "chartqa": load_chartqa}

# Real, billed reference constants from prior sessions — not measured by
# this script. Cited so a reader can trace where each number comes from.
REAL_REFERENCE = {
    # §12.4: real per-page image-route cost (unreduced, one image per call).
    "groq_image_per_page": 1980,  # DocVQA, n=24, §13.8's baseline table
    "gemini_image_per_page": 380,  # DocVQA, n=8, §13.8's baseline table
    # §13.8: real batched-TILED-IMAGE route, one call, N=4 questions.
    "groq_tiled_batched_n4_docvqa": 3317,
    "gemini_tiled_batched_n4_docvqa": 284,
    # §12.4: real per-page TEXT route (no image), one call per page,
    # compressed OCR text — range observed across real samples.
    "groq_text_per_page_range": (29, 467),
    "gemini_text_per_page_range": (17, 216),
}

_ENC = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def build_batched_text_prompt(pages: list[str], questions: list[str]) -> str:
    """Mirrors experiments/batched_tile_qa_check.py's build_prompt() shape
    (numbered items, explicit JSON-only instruction) but carries page TEXT
    instead of a tile-image reference, since there is no image in this
    route at all."""
    n = len(pages)
    assert n == len(questions)
    numbered_pages = "\n\n".join(f"--- Page {k} ---\n{pages[k - 1]}" for k in range(1, n + 1))
    numbered_questions = "\n".join(f"Page {k}: {q}" for k, q in enumerate(questions, start=1))
    instruction = (
        f"Below is the extracted text of {n} document pages, followed by {n} "
        f"questions, one per page. Answer EACH question using ONLY the content "
        f"of that specific page."
    )
    return (
        f"{instruction}\n\n{numbered_pages}\n\nQuestions:\n{numbered_questions}\n\n"
        f"Respond with ONLY a JSON object mapping each page number (as a string) "
        f"to its answer, e.g. {{\"1\": \"...\", \"2\": \"...\"}}. No other text "
        f"outside the JSON."
    )


def run(dataset_name: str, n: int, offset: int = 0) -> dict:
    print(f"Loading {offset + n} real {dataset_name} samples (using [{offset}:{offset + n}])...")
    samples = DATASETS[dataset_name](limit=offset + n)[offset:]

    print("Running real Tesseract OCR + real compression on each page (zero API cost)...")
    pages_raw = []
    pages_compressed = []
    for s in samples:
        ocr_result = image_to_markdown(s.image_path)
        pages_raw.append(ocr_result.markdown)
        pages_compressed.append(compress_markdown(ocr_result.markdown))
    questions = [s.question for s in samples]

    # Route A: N separate text-only calls (today's actual §12.4 shape) —
    # one prompt per page, summed. No real per-sample number to reuse
    # directly (§12.4's range is per-sample, not this dataset's exact
    # pages), so this is an offline proxy count for OUR pages, built the
    # same way §12's real per-page text prompts were built (question +
    # compressed OCR text, no batching wrapper).
    per_page_prompts = [f"Question: {q}\n\nDocument text:\n{p}" for q, p in zip(questions, pages_compressed)]
    route_a_total = sum(_count_tokens(p) for p in per_page_prompts)

    # Route B (novel, this script): ONE batched text-only call, N pages,
    # N questions, structured JSON answer — the untested combination.
    batched_prompt = build_batched_text_prompt(pages_compressed, questions)
    route_b_total = _count_tokens(batched_prompt)

    # Reference-only comparisons against real, previously-measured numbers
    # (§12.4, §13.8) — not computed by this script, just displayed for scale.
    ref = REAL_REFERENCE
    result = {
        "dataset": dataset_name,
        "n": n,
        "offset": offset,
        "route_a_n_separate_text_calls_total_tokens": route_a_total,
        "route_b_batched_text_call_total_tokens": route_b_total,
        "route_a_vs_b_reduction": round(route_a_total / route_b_total, 2) if route_b_total else None,
        "per_page_compressed_tokens": [_count_tokens(p) for p in pages_compressed],
        "per_page_raw_tokens": [_count_tokens(p) for p in pages_raw],
        "compression_saved_any_tokens": any(
            _count_tokens(r) > _count_tokens(c) for r, c in zip(pages_raw, pages_compressed)
        ),
        "reference_real_groq_image_per_page_n_pages": ref["groq_image_per_page"] * n,
        "reference_real_gemini_image_per_page_n_pages": ref["gemini_image_per_page"] * n,
        "note": (
            "route_a/route_b tokens are tiktoken cl100k_base counts on OUR "
            "prompts (offline proxy, not the real Gemini/Groq tokenizer). "
            "reference_real_* fields are real billed numbers from §12.4/§13.8 "
            "for a DIFFERENT sample set, shown for scale only, not a "
            "same-sample comparison."
        ),
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS), default="docvqa")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    result = run(args.dataset, args.n, args.offset)
    print(json.dumps(result, indent=2))

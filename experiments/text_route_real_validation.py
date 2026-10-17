"""Plan step 5: the real-billing question this project has never actually
answered. §9.4 measured `unreduced` (image only) and `ocr_everything` (OCR
text ADDED on top of the image) — never a pure TEXT route that replaces the
image outright. `harness/providers.py`'s new `call_text_only()` methods
(added this session, since `harness/runner.py` never had a text-only call
path at all) make that call possible for the first time.

Three conditions per sample, per provider:
  1. unreduced          - image only, baseline (existing baselines.unreduced)
  2. text_raw           - call_text_only() with raw OCR markdown, no image
  3. text_compressed    - call_text_only() with compress_markdown()'d OCR, no image

Real DocVQA samples, real billed tokens, cached via ResponseCache so a
re-run costs nothing. Respects each provider's free-tier daily quota
(Gemini: 20 req/day/model: `harness/providers.py`. Groq: far more generous,
1000 req/day). $ / request cap must be sized to those limits, not assumed.

Usage:
    python experiments/text_route_real_validation.py --provider groq --n 10
    python experiments/text_route_real_validation.py --provider gemini --n 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.docvqa_loader import load_docvqa
from harness.cache import ResponseCache
from harness.providers import GeminiProvider, GroqProvider, OpenRouterProvider
from harness.runner import ResultDB
from textpath.compress import compress_markdown
from textpath.ocr import image_to_markdown

PROVIDERS = {"gemini": GeminiProvider, "groq": GroqProvider, "openrouter": OpenRouterProvider}
# Text-only calls need real image content already extracted, but
# OpenRouterProvider has no call_text_only() yet - openrouter runs of this
# script are image-route (unreduced) only until that's added.


def next_offset(provider_name: str, dataset: str = "docvqa") -> int:
    """Auto-resume support: count how many distinct DocVQA images this
    provider has already been run against for `unreduced` in results.sqlite,
    so a daily re-run of this script naturally continues where the last one
    left off instead of re-spending quota on already-measured images.
    Matched on the provider's own real model name (from CallResult.model,
    what the API actually reports), not a guessed substring - "groq" never
    appears in qwen/qwen3.6-27b's model string."""
    model_name = PROVIDERS[provider_name]().model
    db = ResultDB("results/results.sqlite")
    cur = db._conn.execute(
        "SELECT COUNT(DISTINCT image_path) FROM runs WHERE dataset = ? AND method = 'unreduced' AND model = ?",
        (dataset, model_name),
    )
    count = cur.fetchone()[0]
    db.close()
    return count


def already_used(provider_name: str, dataset: str = "docvqa") -> set[str]:
    """Real image basenames already spent on this provider - HuggingFace's
    streaming DocVQA loader does not guarantee stable ordering across
    process runs (a network retry mid-stream has been observed to shift
    it), so a position-based --offset skip alone can silently re-spend
    real quota on an image already measured. Filtering by name, not
    position, is what actually prevents that."""
    model_name = PROVIDERS[provider_name]().model
    db = ResultDB("results/results.sqlite")
    cur = db._conn.execute(
        "SELECT DISTINCT image_path FROM runs WHERE dataset = ? AND method = 'unreduced' AND model = ?",
        (dataset, model_name),
    )
    used = {Path(row[0]).name for row in cur.fetchall()}
    db.close()
    return used


def run(provider_name: str, n: int, max_tokens: int = 1024, offset: int | None = None) -> None:
    if offset is None:
        offset = next_offset(provider_name)
    skip = already_used(provider_name)
    print(f"Loading real DocVQA samples starting near [{offset}], skipping {len(skip)} already-used images...")
    pool = load_docvqa(limit=offset + n + len(skip) + 20)[offset:]
    samples = [s for s in pool if s.image_path.name not in skip][:n]

    cache = ResponseCache("results/cache.sqlite")
    provider = PROVIDERS[provider_name](cache=cache)
    db = ResultDB("results/results.sqlite")

    rows = []
    for sample in samples:
        ocr_result = image_to_markdown(sample.image_path)
        raw_text = ocr_result.markdown
        compressed_text = compress_markdown(raw_text)

        # max_tokens is a Groq/OpenAI-shaped kwarg; GeminiProvider's config
        # object has no such field (it's max_output_tokens there), so only
        # forward it for Groq to avoid breaking a future Gemini re-run.
        extra = {"max_tokens": max_tokens} if provider_name in ("groq", "openrouter") else {}
        conditions = [
            ("unreduced", lambda: provider.call(sample.image_path, sample.question, **extra)),
        ]
        if hasattr(provider, "call_text_only"):
            conditions += [
                ("text_raw", lambda: provider.call_text_only(raw_text, sample.question, **extra)),
                ("text_compressed", lambda: provider.call_text_only(compressed_text, sample.question, **extra)),
            ]

        for method_name, call_fn in conditions:
            try:
                result = call_fn()
            except Exception as exc:  # real API errors (quota, auth, etc.) surfaced, not swallowed
                print(f"  [{sample.question_id}] {method_name}: ERROR {exc}")
                continue

            db.log(
                dataset="docvqa",
                image_path=str(sample.image_path),
                method=method_name,
                result=result,
                ground_truths=sample.answers,
                prompt=sample.question,
            )
            rows.append((sample.question_id, method_name, result.input_tokens))
            print(f"  [{sample.question_id}] {method_name}: input_tokens={result.input_tokens}")
            time.sleep(3)  # stay well under any provider's per-minute cap

    print(f"\n{'question_id':<14}{'unreduced':>12}{'text_raw':>12}{'text_compressed':>18}")
    by_qid: dict[str, dict[str, int | None]] = {}
    for qid, method, tokens in rows:
        by_qid.setdefault(qid, {})[method] = tokens
    for qid, methods in by_qid.items():
        print(
            f"{qid:<14}{methods.get('unreduced', 'n/a'):>12}{methods.get('text_raw', 'n/a'):>12}{methods.get('text_compressed', 'n/a'):>18}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=list(PROVIDERS), required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="raise past qwen3.6-27b's <think> preamble (paper/draft.md §9.4, §12.4)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=None,
        help="skip the first N DocVQA samples; omit to auto-resume from how many this provider has already done (see next_offset())",
    )
    args = parser.parse_args()
    run(args.provider, args.n, max_tokens=args.max_tokens, offset=args.offset)

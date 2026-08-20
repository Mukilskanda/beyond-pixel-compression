"""§12's text-route validation, replayed on ChartQA instead of DocVQA — the
same three real-billing conditions (unreduced/text_raw/text_compressed),
same providers, same judge-scoring pipeline. Tests whether the DocVQA
finding (75-95% real token reduction, ~fidelity-neutral per judge scoring)
generalizes to a structurally different visual domain: chart images, where
OCR only captures axis labels/legend text, not the chart's actual semantic
content (bar heights, line trends, relative comparisons) - the single most
likely reviewer objection to "shown on one document-QA dataset."

Usage:
    python experiments/chartqa_text_route_validation.py --provider groq --n 15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.chartqa_loader import load_chartqa
from harness.cache import ResponseCache
from harness.providers import GeminiProvider, GroqProvider
from harness.runner import ResultDB
from textpath.compress import compress_markdown
from textpath.ocr import image_to_markdown

PROVIDERS = {"gemini": GeminiProvider, "groq": GroqProvider}


def run(provider_name: str, n: int, max_tokens: int = 2000) -> None:
    print(f"Loading {n} real ChartQA samples...")
    samples = load_chartqa(limit=n)

    cache = ResponseCache("results/cache.sqlite")
    provider = PROVIDERS[provider_name](cache=cache)
    db = ResultDB("results/results.sqlite")

    extra = {"max_tokens": max_tokens} if provider_name == "groq" else {}

    rows = []
    for sample in samples:
        ocr_result = image_to_markdown(sample.image_path)
        raw_text = ocr_result.markdown
        compressed_text = compress_markdown(raw_text)

        conditions = [
            ("unreduced", lambda: provider.call(sample.image_path, sample.question, **extra)),
            ("text_raw", lambda: provider.call_text_only(raw_text, sample.question, **extra)),
            ("text_compressed", lambda: provider.call_text_only(compressed_text, sample.question, **extra)),
        ]

        for method_name, call_fn in conditions:
            try:
                result = call_fn()
            except Exception as exc:
                print(f"  [{sample.question_id}] {method_name}: ERROR {exc}")
                continue

            db.log(
                dataset="chartqa",
                image_path=str(sample.image_path),
                method=method_name,
                result=result,
                ground_truths=sample.answers,
                prompt=sample.question,
            )
            rows.append((sample.question_id, method_name, result.input_tokens))
            print(f"  [{sample.question_id}] {method_name}: input_tokens={result.input_tokens} (ocr_chars={len(raw_text)})")

    print(f"\n{'question_id':<20}{'unreduced':>12}{'text_raw':>12}{'text_compressed':>18}")
    by_qid: dict[str, dict[str, int | None]] = {}
    for qid, method, tokens in rows:
        by_qid.setdefault(qid, {})[method] = tokens
    for qid, methods in by_qid.items():
        print(f"{qid:<20}{methods.get('unreduced', 'n/a'):>12}{methods.get('text_raw', 'n/a'):>12}{methods.get('text_compressed', 'n/a'):>18}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=list(PROVIDERS), required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=2000)
    args = parser.parse_args()
    run(args.provider, args.n, max_tokens=args.max_tokens)

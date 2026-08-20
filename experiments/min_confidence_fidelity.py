"""Real-billing fidelity check for the min_confidence OCR word filter
(§12.2's open caveat: the 6-15% token reduction it measures offline was
never checked against real answer quality). Two conditions per sample,
TEXT route (no image, via call_text_only): raw OCR (min_confidence=0) vs
filtered OCR (min_confidence=60, the middle of the offline sweep with a
real ~12% reduction). Same 12 real DocVQA samples already used in
experiments/text_route_real_validation.py, so results are directly
comparable. A short sleep between calls avoids the Groq TPM (8000/min)
rate limit that cut a previous run short.

Usage: python experiments/min_confidence_fidelity.py --n 12
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.docvqa_loader import load_docvqa
from harness.cache import ResponseCache
from harness.providers import GroqProvider
from harness.runner import ResultDB
from textpath.ocr import image_to_markdown

MIN_CONFIDENCE_FILTERED = 60


def run(n: int, max_tokens: int = 1500, sleep_s: float = 3.0, offset: int = 0) -> None:
    print(f"Loading {offset + n} real DocVQA samples (using [{offset}:{offset + n}])...")
    samples = load_docvqa(limit=offset + n)[offset:]

    cache = ResponseCache("results/cache.sqlite")
    provider = GroqProvider(cache=cache)
    db = ResultDB("results/results.sqlite")

    rows = []
    for sample in samples:
        raw_text = image_to_markdown(sample.image_path, min_confidence=0).markdown
        filtered_text = image_to_markdown(sample.image_path, min_confidence=MIN_CONFIDENCE_FILTERED).markdown

        conditions = [
            ("min_conf_0", raw_text),
            (f"min_conf_{MIN_CONFIDENCE_FILTERED}", filtered_text),
        ]

        for method_name, text in conditions:
            try:
                result = provider.call_text_only(text, sample.question, max_tokens=max_tokens)
            except Exception as exc:
                print(f"  [{sample.question_id}] {method_name}: ERROR {exc}")
                time.sleep(sleep_s)
                continue

            db.log(
                dataset="docvqa",
                image_path=str(sample.image_path),
                method=method_name,
                result=result,
                ground_truths=sample.answers,
                prompt=sample.question,
            )
            rows.append((sample.question_id, method_name, result.input_tokens, result.finish_reason))
            print(f"  [{sample.question_id}] {method_name}: input_tokens={result.input_tokens} finish_reason={result.finish_reason}")
            time.sleep(sleep_s)  # stay under Groq's 8000 TPM

    print(f"\n{'question_id':<14}{'min_conf_0':>14}{f'min_conf_{MIN_CONFIDENCE_FILTERED}':>16}")
    by_qid: dict[str, dict[str, int | None]] = {}
    for qid, method, tokens, _ in rows:
        by_qid.setdefault(qid, {})[method] = tokens
    for qid, methods in by_qid.items():
        print(f"{qid:<14}{methods.get('min_conf_0', 'n/a'):>14}{methods.get(f'min_conf_{MIN_CONFIDENCE_FILTERED}', 'n/a'):>16}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=1500)
    parser.add_argument("--sleep-s", type=float, default=3.0)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    run(args.n, max_tokens=args.max_tokens, sleep_s=args.sleep_s, offset=args.offset)

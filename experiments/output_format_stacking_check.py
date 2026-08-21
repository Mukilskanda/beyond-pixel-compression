"""Priority 2 (deferred from an earlier session, quota-blocked before it
could start): does constraining OUTPUT format stack with the existing
tiling + mode-selection compounding (§13.4, 14.2x input-token reduction)?
Everything found so far in this project's tiling work (§13) measured
INPUT tokens only - output tokens were never a lever tested for
reduction. If a model reliably follows a short-answer instruction, output
token cost drops too, on top of the input-side savings, for a combined
multiplier - but only if it doesn't hurt fidelity (a short-answer
constraint that can't correctly express a tile-position answer would be
a real, reportable failure mode, not assumed away).

Mechanism check first, small scale, same discipline as every other real
finding in this project: for the SAME tiled + low-resolution composite
image (the exact input-side setup behind the 14.2x number), two real
conditions per tile - free-form answer vs. an explicit length constraint
("Answer in 5 words or fewer.") - judge-scored for fidelity, with real
output_tokens compared directly.

Usage:
    python experiments/output_format_stacking_check.py --n 4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baselines.baselines import GEMINI_LOW_DETAIL_SETTINGS
from data.docvqa_loader import load_docvqa
from harness.cache import ResponseCache
from harness.judge import judge_correctness
from harness.providers import GeminiProvider, GroqProvider
from harness.runner import ResultDB
from pixelpath.tiling import tile_images

JUDGE_MODEL = "openai/gpt-oss-120b"
CONSTRAINT_INSTRUCTION = " Answer in 5 words or fewer, no explanation."
PROVIDERS = {"gemini": GeminiProvider, "groq": GroqProvider}


def run(n: int, provider_name: str = "gemini", max_tokens: int = 2500, sleep_s: float = 1.0) -> None:
    print(f"Loading {n} real DocVQA samples...")
    samples = load_docvqa(limit=n)

    cache = ResponseCache("results/cache.sqlite")
    provider = PROVIDERS[provider_name](cache=cache)
    call_extra = GEMINI_LOW_DETAIL_SETTINGS if provider_name == "gemini" else {"max_tokens": max_tokens}
    judge = GroqProvider(model=JUDGE_MODEL, cache=cache)
    db = ResultDB("results/results.sqlite")

    grid = tile_images([s.image_path for s in samples], tile_size=(600, 800))
    tmp_path = Path("data/scratch") / f"output_format_tile_n{n}_{provider_name}.png"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(tmp_path)

    rows = []
    for k, sample in enumerate(samples, start=1):
        base_prompt = (
            f"This image is a {n}-tile grid, tiles labeled 'Tile 1' through 'Tile {n}'. "
            f"Referring ONLY to Tile {k}, answer this question about that specific sub-image: {sample.question}"
        )
        conditions = [
            ("free_form", base_prompt),
            ("constrained", base_prompt + CONSTRAINT_INSTRUCTION),
        ]
        for method_name, prompt in conditions:
            try:
                result = provider.call(tmp_path, prompt, **call_extra)
            except Exception as exc:
                print(f"  tile={k} {method_name}: ERROR {exc}")
                time.sleep(sleep_s)
                continue

            db.log(
                dataset=f"docvqa_tiled_output_format_{provider_name}",
                image_path=str(tmp_path),
                method=method_name,
                result=result,
                ground_truths=sample.answers,
                prompt=prompt,
            )
            verdict = judge_correctness(sample.question, result.text or "", sample.answers, judge)
            rows.append((k, method_name, verdict, result.input_tokens, result.output_tokens))
            print(f"  tile={k}/{n} {method_name}: verdict={verdict} input={result.input_tokens} output={result.output_tokens}")
            time.sleep(sleep_s)

    print(f"\n{'tile':>6}{'method':<14}{'input_tok':>12}{'output_tok':>12}{'verdict':>10}")
    for k, method_name, verdict, inp, out in rows:
        print(f"{k:>6}{method_name:<14}{inp:>12}{out:>12}{str(verdict):>10}")

    free = [(inp, out, v) for k, m, v, inp, out in rows if m == "free_form"]
    constrained = [(inp, out, v) for k, m, v, inp, out in rows if m == "constrained"]
    if free and constrained:
        avg_out_free = sum(o for _, o, _ in free) / len(free)
        avg_out_constrained = sum(o for _, o, _ in constrained) / len(constrained)
        correct_free = sum(1 for _, _, v in free if v is True)
        correct_constrained = sum(1 for _, _, v in constrained if v is True)
        print(f"\nAvg output tokens: free_form={avg_out_free:.1f}, constrained={avg_out_constrained:.1f}, ratio={avg_out_constrained/avg_out_free:.2f}")
        print(f"Correct (judge=True): free_form={correct_free}/{len(free)}, constrained={correct_constrained}/{len(constrained)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--provider", choices=list(PROVIDERS), default="gemini")
    parser.add_argument("--max-tokens", type=int, default=2500)
    parser.add_argument("--sleep-s", type=float, default=1.0)
    args = parser.parse_args()
    run(args.n, provider_name=args.provider, max_tokens=args.max_tokens, sleep_s=args.sleep_s)

"""Real-billing fidelity check for image tiling (§ tiling mechanism check
confirmed real, large cost savings — this tests whether the model can
still answer correctly about a SPECIFIC tile once N real DocVQA pages are
composited into one grid image, or whether cross-tile confusion /
degrading accuracy sets in as N grows).

For each tile position k in a size-N grid, sends ONE real call: the tiled
image + a question that names the tile explicitly and asks that tile's
real DocVQA question ("Referring to Tile k in the grid image, answer this
question about that specific sub-image: <question>"). Judge-scores the
answer against that tile's real ground truth (harness/judge.py, blind to
tile position/N). This directly tests both content correctness AND
cross-tile confusion (a wrong-but-real-answer-for-a-different-tile would
still fail the ground-truth judge check).

Usage:
    python experiments/tiling_fidelity_check.py --provider groq --ns 2,4,6,9
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.chartqa_loader import load_chartqa
from data.docvqa_loader import load_docvqa
from harness.cache import ResponseCache
from harness.judge import judge_correctness
from harness.providers import GeminiProvider, GroqProvider
from harness.runner import ResultDB
from pixelpath.tiling import tile_images

PROVIDERS = {"gemini": GeminiProvider, "groq": GroqProvider}
DATASETS = {"docvqa": load_docvqa, "chartqa": load_chartqa}
JUDGE_MODEL = "openai/gpt-oss-120b"


def run(provider_name: str, ns: list[int], dataset_name: str = "docvqa", max_tokens: int = 500, sleep_s: float = 2.0) -> None:
    max_n = max(ns)
    print(f"Loading {max_n} real {dataset_name} samples...")
    samples = DATASETS[dataset_name](limit=max_n)

    cache = ResponseCache("results/cache.sqlite")
    extra = {"max_tokens": max_tokens} if provider_name == "groq" else {}
    provider = PROVIDERS[provider_name](cache=cache)
    judge = GroqProvider(model=JUDGE_MODEL, cache=cache)
    db = ResultDB("results/results.sqlite")

    results = []
    for n in ns:
        subset = samples[:n]
        grid = tile_images([s.image_path for s in subset], tile_size=(600, 800))
        tmp_path = Path("data/scratch") / f"fidelity_tile_{dataset_name}_n{n}_{provider_name}.png"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        grid.save(tmp_path)

        for k, sample in enumerate(subset, start=1):
            prompt = (
                f"This image is a {n}-tile grid, tiles labeled 'Tile 1' through 'Tile {n}'. "
                f"Referring ONLY to Tile {k}, answer this question about that specific sub-image: {sample.question}"
            )
            try:
                result = provider.call(tmp_path, prompt, **extra)
            except Exception as exc:
                print(f"  N={n} tile={k}: ERROR {exc}")
                time.sleep(sleep_s)
                continue

            db.log(
                dataset=f"{dataset_name}_tiled",
                image_path=str(tmp_path),
                method=f"tile_n{n}",
                result=result,
                ground_truths=sample.answers,
                prompt=prompt,
            )

            verdict = judge_correctness(sample.question, result.text or "", sample.answers, judge)
            results.append((n, k, verdict, result.input_tokens))
            print(f"  N={n} tile={k}/{n} ({sample.question_id}): verdict={verdict} input_tokens={result.input_tokens}")
            time.sleep(sleep_s)

    print(f"\n{'N':>4}{'correct':>10}{'incorrect':>12}{'unknown':>10}{'accuracy':>12}")
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0, 0])
    for n, k, verdict, _ in results:
        if verdict is True:
            agg[n][0] += 1
        elif verdict is False:
            agg[n][1] += 1
        else:
            agg[n][2] += 1
    for n in sorted(agg):
        correct, incorrect, unknown = agg[n]
        total_scored = correct + incorrect
        acc = f"{correct / total_scored * 100:.1f}%" if total_scored else "n/a"
        print(f"{n:>4}{correct:>10}{incorrect:>12}{unknown:>10}{acc:>12}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=list(PROVIDERS), required=True)
    parser.add_argument("--dataset", choices=list(DATASETS), default="docvqa")
    parser.add_argument("--ns", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--sleep-s", type=float, default=2.0)
    args = parser.parse_args()
    run(args.provider, [int(x) for x in args.ns.split(",")], dataset_name=args.dataset, max_tokens=args.max_tokens, sleep_s=args.sleep_s)

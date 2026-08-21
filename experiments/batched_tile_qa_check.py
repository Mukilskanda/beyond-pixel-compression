"""Novel candidate, derived directly from §13.7's diagnosis: tiling loses
on Groq for the N-distinct-questions workload because each of the N
required calls re-pays the image cost AND re-does the grid-navigation
reasoning from scratch. If instead ONE call carries all N questions and
returns a structured (JSON) answer keyed by tile, the image cost and the
grid-navigation reasoning are each paid ONCE, not N times - a direct fix
for the exact mechanism that made Groq's tiling net-negative.

Mechanism + fidelity check together (one real call already gives both
cost and per-tile answers to judge-score) - same discipline as every
other real finding in this project: small N first, real billing, judge
each parsed answer against its own real ground truth.

Usage:
    python experiments/batched_tile_qa_check.py --provider groq --n 4
    python experiments/batched_tile_qa_check.py --provider gemini --dataset chartqa --n 4
    python experiments/batched_tile_qa_check.py --provider groq --n 4 --offset 4 --efficient
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baselines.baselines import GEMINI_LOW_DETAIL_SETTINGS
from data.chartqa_loader import load_chartqa
from data.docvqa_loader import load_docvqa
from harness.cache import ResponseCache
from harness.judge import judge_correctness
from harness.metrics import _strip_think_block
from harness.providers import GeminiProvider, GroqProvider
from harness.runner import ResultDB
from pixelpath.tiling import tile_images

JUDGE_MODEL = "openai/gpt-oss-120b"
PROVIDERS = {"gemini": GeminiProvider, "groq": GroqProvider}
DATASETS = {"docvqa": load_docvqa, "chartqa": load_chartqa}

EFFICIENCY_INSTRUCTION = (
    " Work through the tiles in order, left to right, top to bottom. Locate each "
    "tile's position ONCE and do not re-verify or second-guess the grid layout "
    "repeatedly - identify a tile's position in one pass and move on to answering "
    "its question. Keep your reasoning brief and direct."
)


def build_prompt(n: int, questions: list[str], efficient: bool = False) -> str:
    numbered = "\n".join(f"Tile {k}: {q}" for k, q in enumerate(questions, start=1))
    instruction = (
        f"This image is a {n}-tile grid, tiles labeled 'Tile 1' through 'Tile {n}'. "
        f"Answer EACH of the following {n} questions, one per tile, using ONLY the "
        f"content of that specific tile:"
    )
    if efficient:
        instruction += EFFICIENCY_INSTRUCTION
    return (
        f"{instruction}\n{numbered}\n\n"
        f"Respond with ONLY a JSON object mapping each tile number (as a string) to "
        f"its answer, e.g. {{\"1\": \"...\", \"2\": \"...\"}}. No other text outside the JSON."
    )


def parse_batched_answers(text: str, n: int) -> dict[str, str]:
    cleaned = _strip_think_block(text or "")
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return {str(k): str(v) for k, v in parsed.items()}


def run(
    provider_name: str,
    n: int,
    dataset_name: str = "docvqa",
    offset: int = 0,
    efficient: bool = False,
    max_tokens: int = 3000,
) -> dict:
    print(f"Loading {offset + n} real {dataset_name} samples (using [{offset}:{offset + n}])...")
    samples = DATASETS[dataset_name](limit=offset + n)[offset:]

    cache = ResponseCache("results/cache.sqlite")
    provider = PROVIDERS[provider_name](cache=cache)
    judge = GroqProvider(model=JUDGE_MODEL, cache=cache)
    db = ResultDB("results/results.sqlite")

    tag = f"{dataset_name}_o{offset}_{'eff' if efficient else 'base'}"
    grid = tile_images([s.image_path for s in samples], tile_size=(600, 800))
    tmp_path = Path("data/scratch") / f"batched_tile_n{n}_{provider_name}_{tag}.png"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(tmp_path)

    prompt = build_prompt(n, [s.question for s in samples], efficient=efficient)
    extra = GEMINI_LOW_DETAIL_SETTINGS if provider_name == "gemini" else {"max_tokens": max_tokens}

    result = provider.call(tmp_path, prompt, **extra)
    print(f"Real call: input_tokens={result.input_tokens} output_tokens={result.output_tokens} finish_reason={result.finish_reason}")

    db.log(
        dataset=f"{dataset_name}_batched_tiled_{provider_name}_{tag}",
        image_path=str(tmp_path),
        method=f"batched_n{n}",
        result=result,
        ground_truths=["n/a - scored per-tile below"],
        prompt=prompt,
    )

    answers = parse_batched_answers(result.text, n)
    print(f"Parsed {len(answers)}/{n} answers from JSON response.")

    correct = incorrect = unknown = missing = 0
    for k, sample in enumerate(samples, start=1):
        answer = answers.get(str(k))
        if answer is None:
            print(f"  tile={k} ({sample.question_id}): MISSING from parsed response")
            missing += 1
            continue
        verdict = judge_correctness(sample.question, answer, sample.answers, judge)
        print(f"  tile={k} ({sample.question_id}): verdict={verdict} answer={answer!r}")
        if verdict is True:
            correct += 1
        elif verdict is False:
            incorrect += 1
        else:
            unknown += 1

    total_scored = correct + incorrect
    acc = f"{correct / total_scored * 100:.1f}%" if total_scored else "n/a"
    total_tokens = result.input_tokens + (result.output_tokens or 0)
    print(f"\nSummary: correct={correct} incorrect={incorrect} unknown={unknown} missing={missing} accuracy={acc}")
    print(f"Real cost: 1 call, input_tokens={result.input_tokens}, output_tokens={result.output_tokens}, total={total_tokens}, finish_reason={result.finish_reason}")

    return {
        "correct": correct, "incorrect": incorrect, "unknown": unknown, "missing": missing,
        "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
        "total_tokens": total_tokens, "finish_reason": result.finish_reason,
        "truncated": result.finish_reason in ("length", "max_tokens"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=list(PROVIDERS), required=True)
    parser.add_argument("--dataset", choices=list(DATASETS), default="docvqa")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--efficient", action="store_true", help="add the brevity/no-re-verification instruction")
    parser.add_argument("--max-tokens", type=int, default=3000)
    args = parser.parse_args()
    run(args.provider, args.n, dataset_name=args.dataset, offset=args.offset, efficient=args.efficient, max_tokens=args.max_tokens)

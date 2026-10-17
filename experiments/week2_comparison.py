"""Week-2 go/no-go checkpoint (plan §6, §7 timeline): the first real
cost-fidelity comparison. Scoped tight to fit today's remaining Gemini
free-tier quota (20 requests/day/model, discovered mid-session — see
paper/draft.md §7 for the full quota story) rather than the original
20-30 sample plan.

Core question: does uniform downscaling already capture most of the
achievable gain? If yes, the honest thing to do is report that plainly, not
reshape the framing to hide it (explicit instruction, see git history).

Conditions compared (kept to 3 to fit quota — not the full 6-baseline sweep):
  1. unreduced       — the fidelity/cost ceiling
  2. uniform_downscale (50%) — the "trivial baseline that often works
     embarrassingly well" per the plan
  3. full pipeline (harness/pipeline.py:build_payload) — this project's
     router + adaptive budget + region selection

Stops cleanly on a 429 rather than crashing uninformatively, and logs
exactly how many real calls succeeded before stopping.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from baselines.baselines import uniform_downscale, unreduced
from data.docvqa_loader import load_docvqa
from harness.cache import ResponseCache
from harness.pipeline import build_payload
from harness.providers import GeminiProvider
from harness.runner import ResultDB

N_SAMPLES = 5  # 3 conditions x 5 samples = 15 calls, fits today's remaining quota


def run() -> dict:
    print(f"Loading {N_SAMPLES} real DocVQA samples...")
    samples = load_docvqa(limit=N_SAMPLES)

    cache = ResponseCache("results/cache.sqlite")
    provider = GeminiProvider(cache=cache)
    db = ResultDB("results/results.sqlite")

    call_count = 0
    stopped_early = False

    for sample in samples:
        # Condition 1: unreduced
        try:
            image_bytes = unreduced(sample.image_path)
            tmp_path = Path("data/scratch") / f"unreduced_{sample.image_path.name}"
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_bytes(image_bytes)
            result = provider.call(str(tmp_path), sample.question)
            db.log("docvqa_week2", str(sample.image_path), "unreduced", result, sample.answers, sample.question)
            call_count += 1
            print(f"  [unreduced] {sample.image_path.name}: input_tokens={result.input_tokens}")
            time.sleep(4)
        except Exception as e:
            print(f"  STOPPED at unreduced/{sample.image_path.name}: {type(e).__name__}: {str(e)[:150]}")
            stopped_early = True
            break

        # Condition 2: uniform_downscale
        try:
            image_bytes = uniform_downscale(sample.image_path, scale=0.5)
            tmp_path = Path("data/scratch") / f"downscale_{sample.image_path.name}"
            tmp_path.write_bytes(image_bytes)
            result = provider.call(str(tmp_path), sample.question)
            db.log(
                "docvqa_week2", str(sample.image_path), "uniform_downscale_50", result, sample.answers, sample.question
            )
            call_count += 1
            print(f"  [downscale] {sample.image_path.name}: input_tokens={result.input_tokens}")
            time.sleep(4)
        except Exception as e:
            print(f"  STOPPED at downscale/{sample.image_path.name}: {type(e).__name__}: {str(e)[:150]}")
            stopped_early = True
            break

        # Condition 3: full pipeline
        try:
            payload = build_payload(sample.image_path, sample.question)
            if payload.image_bytes is not None:
                tmp_path = Path("data/scratch") / f"pipeline_{sample.image_path.name}"
                tmp_path.write_bytes(payload.image_bytes)
                prompt = sample.question
                if payload.text_context:
                    prompt = f"Extracted text:\n{payload.text_context}\n\nQuestion: {sample.question}"
                result = provider.call(str(tmp_path), prompt)
            else:
                # Pure TEXT route: no image at all. GeminiProvider.call()
                # requires an image_path argument (it's an image-vision
                # provider wrapper) — a text-only call path isn't
                # implemented. Real DocVQA samples haven't been observed to
                # route TEXT anyway (see router/rules.py's documented known
                # gap), so this is expected to never trigger in practice;
                # skip cleanly rather than crash if it ever does.
                print(
                    f"  [pipeline] {sample.image_path.name}: TEXT route (no image) — GeminiProvider has no text-only path, skipping"
                )
                continue
            db.log(
                "docvqa_week2",
                str(sample.image_path),
                f"full_pipeline_{payload.route.value}",
                result,
                sample.answers,
                sample.question,
            )
            call_count += 1
            print(
                f"  [pipeline/{payload.route.value}] {sample.image_path.name}: input_tokens={result.input_tokens} budget={payload.pixel_budget}"
            )
            time.sleep(4)
        except Exception as e:
            print(f"  STOPPED at pipeline/{sample.image_path.name}: {type(e).__name__}: {str(e)[:150]}")
            stopped_early = True
            break

    db.close()
    cache.close()

    print(f"\nTotal real API calls made: {call_count}")
    print(f"Stopped early due to rate limit: {stopped_early}")
    return {"call_count": call_count, "stopped_early": stopped_early}


if __name__ == "__main__":
    run()

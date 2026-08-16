"""Cross-provider go/no-go comparison on Groq (qwen/qwen3.6-27b), mirroring
experiments/week2_comparison.py's Gemini run for direct comparability.
Real DocVQA samples, same discipline: real numbers, no reshaping.

Free tier: 30 RPM / 1000 RPD for qwen/qwen3.6-27b (per console.groq.com/docs/rate-limits,
checked before this run — far more generous than Gemini's 20/day).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baselines.baselines import (
    fixed_ratio_crop,
    jpeg_quality_reduction,
    uniform_downscale,
    unreduced,
)
from baselines.ocr_everything import ocr_everything
from data.docvqa_loader import load_docvqa
from harness.cache import ResponseCache
from harness.pipeline import build_payload
from harness.providers import GroqProvider
from harness.runner import ResultDB

N_SAMPLES = 3


def run() -> dict:
    print(f"Loading {N_SAMPLES} real DocVQA samples...")
    samples = load_docvqa(limit=N_SAMPLES)

    cache = ResponseCache("results/cache.sqlite")
    provider = GroqProvider(cache=cache)
    db = ResultDB("results/results.sqlite")
    Path("data/scratch").mkdir(parents=True, exist_ok=True)

    call_count = 0
    stopped_early = False

    for sample in samples:
        conditions = [
            ("unreduced", lambda p: unreduced(p), {}),
            ("uniform_downscale_50", lambda p: uniform_downscale(p, 0.5), {}),
            ("jpeg_quality_50", lambda p: jpeg_quality_reduction(p, 50), {}),
            ("fixed_ratio_crop_60", lambda p: fixed_ratio_crop(p, 0.6), {}),
            # Groq's equivalent of Gemini's media_resolution is the OpenAI-shaped
            # `detail` param — already informally tested and found to have ZERO
            # effect (1827 tokens regardless of detail=low/high/unset). Included
            # here for a like-for-like row in the comparison table.
            ("low_detail", lambda p: unreduced(p), {"detail": "low"}),
        ]

        for method_name, transform, extra_settings in conditions:
            try:
                image_bytes = transform(sample.image_path)
                tmp_path = Path("data/scratch") / f"{method_name}_{sample.image_path.name}"
                tmp_path.write_bytes(image_bytes)
                result = provider.call(str(tmp_path), sample.question, max_tokens=300, **extra_settings)
                db.log("docvqa_week2_groq", str(sample.image_path), method_name, result, sample.answers, sample.question)
                call_count += 1
                print(f"  [{method_name}] {sample.image_path.name}: input_tokens={result.input_tokens}")
                time.sleep(2)
            except Exception as e:
                print(f"  STOPPED at {method_name}/{sample.image_path.name}: {type(e).__name__}: {str(e)[:150]}")
                stopped_early = True
                break

        # OCR-everything (text baseline, still costs a real call since we send extracted text)
        try:
            ocr_result = ocr_everything(sample.image_path)
            # This baseline is text-only in principle, but GroqProvider requires
            # an image; send the original image with the OCR text prepended to
            # the prompt, consistent with how HYBRID route treats OCR text.
            prompt = f"Extracted text:\n{ocr_result.markdown}\n\nQuestion: {sample.question}"
            result = provider.call(str(sample.image_path), prompt, max_tokens=300)
            db.log("docvqa_week2_groq", str(sample.image_path), "ocr_everything", result, sample.answers, sample.question)
            call_count += 1
            print(f"  [ocr_everything] {sample.image_path.name}: input_tokens={result.input_tokens}")
            time.sleep(2)
        except Exception as e:
            print(f"  STOPPED at ocr_everything/{sample.image_path.name}: {type(e).__name__}: {str(e)[:150]}")
            stopped_early = True

        # full pipeline
        try:
            payload = build_payload(sample.image_path, sample.question)
            if payload.image_bytes is not None:
                tmp_path = Path("data/scratch") / f"pipeline_{sample.image_path.name}"
                tmp_path.write_bytes(payload.image_bytes)
                prompt = sample.question
                if payload.text_context:
                    prompt = f"Extracted text:\n{payload.text_context}\n\nQuestion: {sample.question}"
                result = provider.call(str(tmp_path), prompt, max_tokens=300)
                db.log("docvqa_week2_groq", str(sample.image_path), f"full_pipeline_{payload.route.value}", result, sample.answers, sample.question)
                call_count += 1
                print(f"  [pipeline/{payload.route.value}] {sample.image_path.name}: input_tokens={result.input_tokens} budget={payload.pixel_budget}")
                time.sleep(2)
            else:
                print(f"  [pipeline] {sample.image_path.name}: TEXT route (no image) — GroqProvider has no text-only path, skipping")
        except Exception as e:
            print(f"  STOPPED at pipeline/{sample.image_path.name}: {type(e).__name__}: {str(e)[:150]}")
            stopped_early = True

        if stopped_early:
            break

    db.close()
    cache.close()

    print(f"\nTotal real API calls made: {call_count}")
    print(f"Stopped early: {stopped_early}")
    return {"call_count": call_count, "stopped_early": stopped_early}


if __name__ == "__main__":
    run()

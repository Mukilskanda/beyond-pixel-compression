"""INDICON 2026 revision, Reviewer 1 point 3: direct measurement of every
Table III column instead of the modeled 4 x mean-baseline column.

Per window of N=4 samples (dataset offset o), all on the same provider
settings so the comparison is like-for-like:

  sep      4 separate single-image calls, one question each
  multi    1 call carrying the 4 images UNTILED + all 4 questions (JSON out)
  tiled    1 call on the 2x2 tiled composite + all 4 questions (JSON out),
           identical prompt/settings to experiments/batched_tile_qa_check.py
           (so existing trials are cache hits, not new calls)
  pertile  4 calls on the tiled composite, one tile question each
           (Groq only; Reviewer 1 asked for Groq per-tile fidelity at N=4)

Input AND output tokens are recorded for every call; every answer is
judge-scored (openai/gpt-oss-120b, harness/judge.py), so fidelity is on the
same metric as Section IV-C.

Gemini runs use MEDIA_RESOLUTION_LOW in every condition (matching the
original batched call); Groq uses max_tokens=3000 for sep/multi/tiled and
the original per-tile budget (500 + 250N) for pertile.

Output: one JSON line per (window, condition) appended to
results/revision_table3.jsonl; every call also logged to results.sqlite
with dataset tag rev_t3_<provider>_<dataset>_o<offset>_<condition>.

Usage:
    python experiments/revision_table3.py --provider groq --dataset docvqa --offsets 0,4,8
    python experiments/revision_table3.py --provider gemini --dataset docvqa --offsets 0,4,8 --conditions sep,multi,tiled
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from baselines.baselines import GEMINI_LOW_DETAIL_SETTINGS
from data.chartqa_loader import load_chartqa
from data.docvqa_loader import load_docvqa
from experiments.batched_tile_qa_check import build_prompt, parse_batched_answers
from harness.cache import ResponseCache
from harness.judge import judge_correctness
from harness.providers import CallResult, GeminiProvider, GroqProvider
from harness.runner import ResultDB
from pixelpath.tiling import tile_images

N = 4
GROQ_MODEL = "qwen/qwen3.8-27b"  # qwen/qwen3.6-27b returned 404 model_not_found on 2026-10-16
JUDGE_MODEL = "openai/gpt-oss-120b"
DATASETS = {"docvqa": load_docvqa, "chartqa": load_chartqa}
OUT = Path("results/revision_table3.jsonl")


class DailyQuotaExhausted(RuntimeError):
    pass


def with_retry(fn, label: str, tries: int = 4):
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:  # provider SDKs raise different 429 types
            msg = str(exc)
            if "per day" in msg.lower() or "tokens per day" in msg.lower() or "TPD" in msg or "PerDay" in msg:
                raise DailyQuotaExhausted(msg[:300]) from exc
            if "429" in msg or "rate" in msg.lower() or "RESOURCE_EXHAUSTED" in msg:
                wait = 65
                m = re.search(r"try again in ([\d.]+)s", msg)
                if m:
                    wait = float(m.group(1)) + 3
                print(f"    [{label}] rate-limited, sleeping {wait:.0f}s (attempt {attempt + 1})")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"{label}: still rate-limited after {tries} attempts")


def multi_image_call(provider, image_paths: list[Path], prompt: str, cache: ResponseCache, **settings) -> CallResult:
    """One request carrying several separate (untiled) images. Cached on the
    concatenated image bytes + prompt + settings, like the provider wrappers."""
    blob = b"".join(Path(p).read_bytes() for p in image_paths)
    key = {"prompt": prompt, "multi_image": len(image_paths), **settings}
    hit = cache.get(blob, provider.model, key)
    if hit is not None:
        return CallResult(**hit)

    start = time.monotonic()
    if isinstance(provider, GeminiProvider):
        from google.genai import types
        from PIL import Image

        contents = []
        for k, p in enumerate(image_paths, start=1):
            contents += [f"Image {k}:", Image.open(p)]
        contents.append(prompt)
        config = types.GenerateContentConfig(**settings) if settings else None
        resp = provider.client.models.generate_content(model=provider.model, contents=contents, config=config)
        usage = resp.usage_metadata
        fr = resp.candidates[0].finish_reason if getattr(resp, "candidates", None) else None
        result = CallResult(
            text=resp.text or "",
            input_tokens=usage.prompt_token_count,
            output_tokens=usage.candidates_token_count,
            latency_s=time.monotonic() - start,
            model=provider.model,
            raw={},
            finish_reason=str(fr).lower() if fr else None,
        )
    else:
        settings = dict(settings)
        max_tokens = settings.pop("max_tokens", 1024)
        content = []
        for k, p in enumerate(image_paths, start=1):
            b64 = base64.standard_b64encode(Path(p).read_bytes()).decode()
            content.append({"type": "text", "text": f"Image {k}:"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        content.append({"type": "text", "text": prompt})
        resp = provider.client.chat.completions.create(
            model=provider.model, max_tokens=max_tokens, messages=[{"role": "user", "content": content}], **settings
        )
        result = CallResult(
            text=resp.choices[0].message.content or "",
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            latency_s=time.monotonic() - start,
            model=resp.model,
            raw={},
            finish_reason=str(resp.choices[0].finish_reason).lower(),
        )
    cache.put(blob, provider.model, key, result.__dict__)
    return result


def multi_prompt(questions: list[str]) -> str:
    numbered = "\n".join(f"Image {k}: {q}" for k, q in enumerate(questions, start=1))
    return (
        f"You are given {len(questions)} separate images, labeled 'Image 1' through 'Image {len(questions)}'. "
        f"Answer EACH of the following {len(questions)} questions, one per image, using ONLY the content of that "
        f"specific image:\n{numbered}\n\n"
        f"Respond with ONLY a JSON object mapping each image number (as a string) to its answer, "
        f'e.g. {{"1": "...", "2": "..."}}. No other text outside the JSON.'
    )


def truncated(r: CallResult) -> bool:
    return (r.finish_reason or "") in ("length", "max_tokens", "finishreason.max_tokens")


def run_window(provider_name, dataset, offset, conditions, cache, db, judge, sleep_s):
    samples = DATASETS[dataset](limit=offset + N)[offset : offset + N]
    provider = GeminiProvider(cache=cache) if provider_name == "gemini" else GroqProvider(model=GROQ_MODEL, cache=cache)
    base = dict(GEMINI_LOW_DETAIL_SETTINGS) if provider_name == "gemini" else {"max_tokens": 3000}
    tag = f"rev_t3_{provider_name}_{dataset}_o{offset}"

    def judge_one(q, ans, gts):
        return with_retry(lambda: judge_correctness(q, ans or "", gts, judge), "judge")

    def record(cond, calls, verdicts, extra=None):
        row = {
            "provider": provider_name,
            "dataset": dataset,
            "offset": offset,
            "condition": cond,
            "n_calls": len(calls),
            "input_tokens": sum(c.input_tokens or 0 for c in calls),
            "output_tokens": sum(c.output_tokens or 0 for c in calls),
            "per_call_input": [c.input_tokens for c in calls],
            "per_call_output": [c.output_tokens for c in calls],
            "any_truncated": any(truncated(c) for c in calls),
            "correct": sum(1 for v in verdicts if v is True),
            "incorrect": sum(1 for v in verdicts if v is False),
            "unscored": sum(1 for v in verdicts if v is None),
            "verdicts": verdicts,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if extra:
            row.update(extra)
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"  {cond:<8} in={row['input_tokens']:>6} out={row['output_tokens']:>6} "
            f"correct={row['correct']}/{N} trunc={row['any_truncated']}"
        )
        return row

    grid = tile_images([s.image_path for s in samples], tile_size=(600, 800))
    grid_path = Path("data/scratch") / f"batched_tile_n{N}_{provider_name}_{dataset}_o{offset}_base.png"
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    if not grid_path.exists():
        grid.save(grid_path)

    if "sep" in conditions:
        calls, verdicts = [], []
        for s in samples:
            r = with_retry(lambda s=s: provider.call(s.image_path, s.question, **dict(base)), "sep")
            db.log(
                dataset=f"{tag}_sep",
                image_path=str(s.image_path),
                method="rev_sep",
                result=r,
                ground_truths=s.answers,
                prompt=s.question,
            )
            calls.append(r)
            verdicts.append(None if truncated(r) else judge_one(s.question, r.text, s.answers))
            time.sleep(sleep_s)
        record("sep", calls, verdicts)

    if "multi" in conditions:
        prompt = multi_prompt([s.question for s in samples])
        try:
            r = with_retry(
                lambda: multi_image_call(provider, [s.image_path for s in samples], prompt, cache, **dict(base)),
                "multi",
            )
        except DailyQuotaExhausted:
            raise
        except Exception as exc:
            print(f"  multi   REQUEST FAILED: {str(exc)[:300]}")
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "provider": provider_name,
                            "dataset": dataset,
                            "offset": offset,
                            "condition": "multi",
                            "error": str(exc)[:500],
                        }
                    )
                    + "\n"
                )
        else:
            db.log(
                dataset=f"{tag}_multi",
                image_path=";".join(str(s.image_path) for s in samples),
                method="rev_multi_n4",
                result=r,
                ground_truths=["n/a"],
                prompt=prompt,
            )
            answers = parse_batched_answers(r.text, N)
            verdicts = [
                judge_one(s.question, answers[str(k)], s.answers) if str(k) in answers else None
                for k, s in enumerate(samples, start=1)
            ]
            record("multi", [r], verdicts, {"parsed": len(answers)})
        time.sleep(sleep_s)

    if "tiled" in conditions:
        prompt = build_prompt(N, [s.question for s in samples])
        r = with_retry(lambda: provider.call(grid_path, prompt, **dict(base)), "tiled")
        db.log(
            dataset=f"{tag}_tiled",
            image_path=str(grid_path),
            method="rev_tiled_n4",
            result=r,
            ground_truths=["n/a"],
            prompt=prompt,
        )
        answers = parse_batched_answers(r.text, N)
        verdicts = [
            judge_one(s.question, answers[str(k)], s.answers) if str(k) in answers else None
            for k, s in enumerate(samples, start=1)
        ]
        record("tiled", [r], verdicts, {"parsed": len(answers)})
        time.sleep(sleep_s)

    if "pertile" in conditions:
        calls, verdicts = [], []
        pt = dict(base)
        if provider_name == "groq":
            pt["max_tokens"] = 500 + 250 * N
        for k, s in enumerate(samples, start=1):
            prompt = (
                f"This image is a {N}-tile grid, tiles labeled 'Tile 1' through 'Tile {N}'. "
                f"Referring ONLY to Tile {k}, answer this question about that specific sub-image: {s.question}"
            )
            r = with_retry(lambda prompt=prompt: provider.call(grid_path, prompt, **dict(pt)), "pertile")
            db.log(
                dataset=f"{tag}_pertile",
                image_path=str(grid_path),
                method="rev_pertile_n4",
                result=r,
                ground_truths=s.answers,
                prompt=prompt,
            )
            calls.append(r)
            verdicts.append(None if truncated(r) else judge_one(s.question, r.text, s.answers))
            time.sleep(sleep_s)
        record("pertile", calls, verdicts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["groq", "gemini"], required=True)
    ap.add_argument("--dataset", choices=list(DATASETS), default="docvqa")
    ap.add_argument("--offsets", default="0,4,8")
    ap.add_argument("--conditions", default="sep,multi,tiled,pertile")
    ap.add_argument("--sleep", type=float, default=8.0)
    a = ap.parse_args()
    cache = ResponseCache("results/cache.sqlite")
    db = ResultDB("results/results.sqlite")
    judge = GroqProvider(model=JUDGE_MODEL, cache=cache)
    conditions = a.conditions.split(",")
    for off in [int(x) for x in a.offsets.split(",")]:
        print(f"\n=== {a.provider}/{a.dataset} offset {off} ===")
        try:
            run_window(a.provider, a.dataset, off, conditions, cache, db, judge, a.sleep)
        except DailyQuotaExhausted as exc:
            print(f"DAILY QUOTA EXHAUSTED, stopping: {exc}")
            break


if __name__ == "__main__":
    main()

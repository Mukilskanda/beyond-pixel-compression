"""Revision-2 step A: pixel-reduction sweep on Groq Qwen3.8-27B.

Replicates Table II's test (does client-side pixel reduction change the
reported image cost?) on the successor model, which the paper currently
shows only for the withdrawn Qwen3.6-27B. Reported as a separate
replication; never pooled with Qwen3.6 data.

Images: 20 of the 172 natural images in results/groq_image_billing.csv,
chosen with seed 42 and stratified by page shape: all 9 tall pages
(h/w >= 1.8), 5 of the 9 near-square pages (0.9 <= h/w <= 1.2) and 6 of the
remaining images. The chosen IDs are written to
results/qwen38_pixel_sweep_ids.json before any call.

Conditions per image (same question text for all, the dataset question):
unreduced, downscale 50%, downscale 10%, JPEG quality 50, centre crop 60%,
and the OpenAI-style detail=low request parameter on the unreduced image.

Every call goes through harness.quota.metered (stops cleanly at the Groq
daily cap) and is logged to results/results.sqlite (dataset tag
rev2_qwen38_pixel_sweep) and to results/qwen38_pixel_sweep.jsonl. Resumable.

Usage (from the repository root): python experiments/qwen38_pixel_sweep.py
"""

from __future__ import annotations

import csv
import json
import random
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from baselines.baselines import fixed_ratio_crop, jpeg_quality_reduction, uniform_downscale, unreduced
from harness import quota
from harness.cache import ResponseCache
from harness.providers import GroqProvider
from harness.runner import ResultDB

MODEL = "qwen/qwen3.8-27b"
MAX_TOKENS = 300
OUT = Path("results/qwen38_pixel_sweep.jsonl")
IDS = Path("results/qwen38_pixel_sweep_ids.json")
SCRATCH = Path("data/scratch/qwen38_pixel_sweep")

CONDITIONS = {
    "unreduced": (lambda p: unreduced(p), "image/png", {}),
    "downscale_50": (lambda p: uniform_downscale(p, 0.5), "image/png", {}),
    "downscale_10": (lambda p: uniform_downscale(p, 0.1), "image/png", {}),
    "jpeg_q50": (lambda p: jpeg_quality_reduction(p, 50), "image/jpeg", {}),
    "crop_60": (lambda p: fixed_ratio_crop(p, 0.6), "image/png", {}),
    "detail_low": (lambda p: unreduced(p), "image/png", {"detail": "low"}),
}


def choose_images() -> list[dict]:
    if IDS.exists():
        return json.loads(IDS.read_text(encoding="utf-8"))
    with open("results/groq_image_billing.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["hw"] = int(r["height_px"]) / int(r["width_px"])
    tall = [r for r in rows if r["hw"] >= 1.8]
    square = [r for r in rows if 0.9 <= r["hw"] <= 1.2]
    other = [r for r in rows if r not in tall and r not in square]
    rng = random.Random(42)
    chosen = tall + rng.sample(square, 5) + rng.sample(other, 6)
    picked = [{"dataset": r["dataset"], "image": r["image"], "width_px": int(r["width_px"]),
               "height_px": int(r["height_px"]), "qwen36_tokens": int(r["billed_input_tokens"]),
               "shape": "tall" if r in tall else ("near-square" if r in square else "other")} for r in chosen]
    IDS.write_text(json.dumps(picked, indent=1), encoding="utf-8")
    return picked


def question_for(dataset: str, image: str) -> str:
    folder = "docvqa_cache" if dataset == "docvqa" else "chartqa_cache"
    con = sqlite3.connect("results/results.sqlite")
    row = con.execute(
        "SELECT prompt FROM runs WHERE method='unreduced' AND model='qwen/qwen3.6-27b' AND image_path LIKE ? LIMIT 1",
        (f"%{folder}%{image}",),
    ).fetchone()
    return row[0]


def main() -> None:
    images = choose_images()
    cache = ResponseCache("results/cache.sqlite")
    groq = GroqProvider(model=MODEL, cache=cache)
    db = ResultDB("results/results.sqlite")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        done = {(json.loads(x)["image"], json.loads(x)["condition"]) for x in OUT.read_text(encoding="utf-8").splitlines()}
    for img in images:
        folder = "docvqa_cache" if img["dataset"] == "docvqa" else "chartqa_cache"
        src = Path("data") / folder / img["image"]
        q = question_for(img["dataset"], img["image"])
        for cond, (transform, media, extra) in CONDITIONS.items():
            if (img["image"], cond) in done:
                continue
            ext = "jpg" if media == "image/jpeg" else "png"
            path = SCRATCH / f"{Path(img['image']).stem}_{cond}.{ext}"
            if not path.exists():
                path.write_bytes(transform(src))
            try:
                r = quota.metered("groq", MODEL, lambda: groq.call(path, q, media_type=media, max_tokens=MAX_TOKENS, **extra),
                                  est_tokens=2_400, note=f"pixel_sweep {img['image']} {cond}")
            except quota.BudgetExhausted as exc:
                print(f"STOP: {exc}")
                return
            except Exception as exc:  # provider rejected this request (e.g. image below 32 px): record it, go on
                msg = str(exc)
                if "400" not in msg:
                    raise
                row = {"image": img["image"], "dataset": img["dataset"], "shape": img["shape"], "condition": cond,
                       "model": MODEL, "prompt_tokens": None, "rejected": True, "error": msg[:400],
                       "sent_size": list(__import__("PIL.Image", fromlist=["Image"]).open(path).size),
                       "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
                with open(OUT, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"{img['image']:<24} {cond:<13} REJECTED by provider: {msg[:120]}")
                continue
            db.log(dataset="rev2_qwen38_pixel_sweep", image_path=str(path), method=cond, result=r, prompt=q)
            row = {"image": img["image"], "dataset": img["dataset"], "shape": img["shape"], "condition": cond,
                   "model": r.model, "prompt_tokens": r.input_tokens, "completion_tokens": r.output_tokens,
                   "finish_reason": r.finish_reason, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{img['image']:<24} {cond:<13} prompt_tokens={r.input_tokens}")
            time.sleep(2)
    print("pixel sweep complete;", quota.status())


if __name__ == "__main__":
    main()

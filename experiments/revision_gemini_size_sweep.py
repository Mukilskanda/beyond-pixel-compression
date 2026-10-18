"""Revision-2 step B: logged re-run of the Gemini 2.5 Flash image-size sweep (Fig. 1).

The original 12-call sweep (Aug. 2026) was run interactively and not logged.
This re-runs the same design with every call logged:
  - 12 points from 10x10 to 8000x8000 px, one distinct image per point,
    blank-white images plus one dense random-noise image (seed 0) at
    1000x1000 and one 3000x4000 page-shaped image;
  - default resolution, fixed short prompt;
  - full usage_metadata, the request config and timestamps written to
    results/gemini_size_sweep.jsonl (and each call to results.sqlite,
    dataset tag rev2_gemini_size_sweep).
Calls go through harness.quota.metered, so the run stops cleanly at the
Gemini daily request cap and resumes on the next run.

Also writes analysis/fig1_data.csv: for every plotted side length, the
documented-rule tokens (<=384 px in both dimensions: 258; otherwise
ceil(w/768) x ceil(h/768) tiles x 258) and, for the swept points, the
measured image tokens.

Usage (from the repository root): python experiments/revision_gemini_size_sweep.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from PIL import Image

from harness import quota
from harness.cache import ResponseCache
from harness.providers import GeminiProvider
from harness.runner import ResultDB

MODEL = "gemini-2.5-flash"
PROMPT = "Describe this image in one word."
OUT = Path("results/gemini_size_sweep.jsonl")
FIG1 = Path("analysis/fig1_data.csv")
SCRATCH = Path("data/scratch/size_sweep")

# (label, width, height, kind)
POINTS = [
    ("10x10", 10, 10, "blank"),
    ("32x32", 32, 32, "blank"),
    ("100x100", 100, 100, "blank"),
    ("384x384", 384, 384, "blank"),
    ("512x512", 512, 512, "blank"),
    ("768x768", 768, 768, "blank"),
    ("1000x1000", 1000, 1000, "blank"),
    ("1000x1000_noise", 1000, 1000, "noise"),
    ("2000x2000", 2000, 2000, "blank"),
    ("3000x4000", 3000, 4000, "blank"),
    ("4000x4000", 4000, 4000, "blank"),
    ("8000x8000", 8000, 8000, "blank"),
]


def documented_tokens(w: int, h: int) -> int:
    if w <= 384 and h <= 384:
        return 258
    return math.ceil(w / 768) * math.ceil(h / 768) * 258


def make_image(label: str, w: int, h: int, kind: str) -> Path:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    path = SCRATCH / f"{label}.png"
    if not path.exists():
        if kind == "noise":
            arr = np.random.default_rng(0).integers(0, 256, size=(h, w, 3), dtype=np.uint8)
            Image.fromarray(arr).save(path)
        else:
            Image.new("RGB", (w, h), "white").save(path)
    return path


def image_tokens(usage: dict) -> int | None:
    for d in usage.get("prompt_tokens_details") or []:
        if "IMAGE" in str(d.get("modality")):
            return d.get("token_count")
    return None


def write_fig1_data() -> None:
    measured = {}
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["width"] == r["height"] and r["kind"] == "blank":
                measured[r["width"]] = r["image_tokens"]
    sides = sorted({int(round(x)) for x in np.logspace(1, math.log10(8000), 400)} | set(measured))
    FIG1.parent.mkdir(exist_ok=True)
    with open(FIG1, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["side_px", "documented_rule_tokens", "measured_image_tokens"])
        for s in sides:
            w.writerow([s, documented_tokens(s, s), measured.get(s, "")])
    print(f"wrote {FIG1}: documented rule at 8000x8000 = {documented_tokens(8000, 8000)}")


def main() -> None:
    cache = ResponseCache("results/cache.sqlite")
    gem = GeminiProvider(cache=cache)
    db = ResultDB("results/results.sqlite")
    done = set()
    if OUT.exists():
        done = {json.loads(line)["label"] for line in OUT.read_text(encoding="utf-8").splitlines()}
    for label, w, h, kind in POINTS:
        if label in done:
            continue
        path = make_image(label, w, h, kind)
        try:
            r = quota.metered("gemini", MODEL, lambda: gem.call(path, PROMPT), note=f"size_sweep {label}")
        except quota.BudgetExhausted as exc:
            print(f"STOP: {exc}")
            break
        usage = (r.raw or {}).get("usage_metadata") or {}
        db.log(dataset="rev2_gemini_size_sweep", image_path=str(path), method=f"sweep_{kind}", result=r, prompt=PROMPT)
        row = {"label": label, "width": w, "height": h, "kind": kind, "model": MODEL,
               "request_config": {"prompt": PROMPT, "media_resolution": "default"},
               "prompt_tokens": r.input_tokens, "image_tokens": image_tokens(usage),
               "documented_rule_tokens": documented_tokens(w, h), "usage_metadata": usage,
               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{label:<17} image_tokens={row['image_tokens']} prompt_tokens={r.input_tokens} documented={row['documented_rule_tokens']}")
        time.sleep(4)
    write_fig1_data()
    print(quota.status())


if __name__ == "__main__":
    main()

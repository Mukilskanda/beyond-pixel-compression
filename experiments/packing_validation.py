"""Validate pixelpath/packing.py's local token-cost estimate against real
provider numbers, using Gemini 2.5 Flash's free tier (zero cost).

Isolates image-only token cost from prompt-text overhead: a fixed short
prompt is measured alone first (text-only call), then subtracted from each
image+prompt call's total. This is necessary because packing.py models
IMAGE tiling cost specifically, not a full multi-modal payload — sending the
router's actual OCR text alongside the image (as harness/pipeline.py does
for HYBRID) would confound the measurement with un-modeled text tokens.

IMPORTANT — free tier caveat: this validates against Gemini 2.5 Flash's
FREE TIER specifically. Free-tier and paid-tier requests may be tokenized
identically (image tiling is a property of the model, not the billing tier)
but this has NOT been independently confirmed here — flag this explicitly
in any methodology section that cites these numbers, don't generalize
silently to "the provider's token cost" without that caveat.

Usage:
    python experiments/packing_validation.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from baselines.baselines import unreduced, uniform_downscale
from data.docvqa_loader import load_docvqa
from pixelpath.packing import tile_cost
from PIL import Image

FIXED_PROMPT = "Describe this image in one word."

# Hand-picked from a real 20-sample DocVQA pull, spread across the route
# types that actually occur in real data. NOTE: no TEXT-routed sample is
# included because none exist in the real 20 — this is the same documented
# known gap from router/rules.py (real docs currently route HYBRID or
# PIXEL, never TEXT under v1 thresholds), not an oversight in this probe.
SELECTED = {
    # (filename, route) — route determines which processed-image transform
    # this validation sends, matching what harness/pipeline.py actually sends.
    "49153.png": "hybrid",   # chart
    "24580.png": "hybrid",   # handwritten note
    "57349.png": "hybrid",   # company letter
    "39079.png": "hybrid",   # ad
    "24426.png": "hybrid",   # budget/foundation doc
    "49168.png": "hybrid",   # schedule
    "57366.png": "hybrid",   # fashion ad
    "57368.png": "pixel",    # two-column table report
    "57370.png": "pixel",    # two-column table report
    "57373.png": "pixel",    # two-column table report
}


def processed_image_bytes(image_path: Path, route: str) -> bytes:
    """Match what harness/pipeline.py actually sends for each route's image component."""
    if route == "pixel":
        return unreduced(image_path)
    if route == "hybrid":
        return uniform_downscale(image_path, scale=0.5)
    raise ValueError(f"no image sent for route={route!r}")


def run_validation() -> list[dict]:
    from google import genai

    client = genai.Client()

    print("Measuring fixed-prompt overhead (text-only call)...")
    baseline_response = client.models.generate_content(model="gemini-2.5-flash", contents=[FIXED_PROMPT])
    prompt_overhead_tokens = baseline_response.usage_metadata.prompt_token_count
    print(f"  fixed prompt overhead: {prompt_overhead_tokens} tokens\n")

    all_samples = {s.image_path.name: s for s in load_docvqa(limit=20)}

    results = []
    for i, (filename, route) in enumerate(SELECTED.items()):
        if filename not in all_samples:
            print(f"WARNING: {filename} not found in the loaded 20 samples — skipping")
            continue
        sample = all_samples[filename]

        import io

        image_bytes = processed_image_bytes(sample.image_path, route)
        pil_img = Image.open(io.BytesIO(image_bytes))  # dimensions of the ACTUAL processed bytes sent, not the original file
        w, h = pil_img.size

        local_estimate = tile_cost(w, h)

        # Space calls out to stay well under 15 RPM even with the baseline call.
        if i > 0:
            time.sleep(5)

        response = client.models.generate_content(model="gemini-2.5-flash", contents=[pil_img, FIXED_PROMPT])
        real_total = response.usage_metadata.prompt_token_count
        real_image_tokens = real_total - prompt_overhead_tokens

        diff = real_image_tokens - local_estimate
        pct_diff = (diff / real_image_tokens * 100) if real_image_tokens else float("nan")

        row = {
            "sample": filename,
            "route": route,
            "dimensions": f"{w}x{h}",
            "local_estimate": local_estimate,
            "real_tokens": real_image_tokens,
            "diff": diff,
            "pct_diff": pct_diff,
        }
        results.append(row)
        print(f"{filename:12} route={route:7} dims={w}x{h:<6} local={local_estimate:5} real={real_image_tokens:5} "
              f"diff={diff:+6} ({pct_diff:+.1f}%)")

    return results


if __name__ == "__main__":
    results = run_validation()

    print("\n=== Summary ===")
    diffs = [r["pct_diff"] for r in results]
    print(f"Mean %% diff: {sum(diffs)/len(diffs):.1f}%%")
    print(f"Min/Max %% diff: {min(diffs):.1f}%% / {max(diffs):.1f}%%")

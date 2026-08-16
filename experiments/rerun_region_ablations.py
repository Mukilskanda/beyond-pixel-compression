"""Re-run the three region-selection ablations (paper/draft.md §3.4, §6,
experiments/ablations.yaml) at larger real-data scale than their original
verification (5-10 samples), using the 40 real DocVQA samples now available.
Zero API calls - these ablations operate on local saliency/region-selection
code only, never touch a provider. Confirms whether the original findings
(stated as real, small-N results) hold up at 4-8x the sample size, and
reports plainly if any change.
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from data.docvqa_loader import load_docvqa
from pixelpath.regions import Region, dp_select, greedy_select, propose_regions
from pixelpath.saliency import saliency_map

N_SAMPLES = 40


def ablation_dp_vs_greedy(samples) -> dict:
    """Original finding (9749527, 30 real combos): identical on flat-cost
    regions. Re-run at 40 samples x 3 budgets = 120 combos."""
    budgets = [400, 700, 1000]
    differences = 0
    total = 0
    for s in samples:
        try:
            regions = propose_regions(s.image_path, grid=4)
        except Exception as e:
            print(f"  [dp_vs_greedy] skip {s.image_path}: {type(e).__name__}: {e}")
            continue
        for b in budgets:
            total += 1
            greedy = {(r.x, r.y) for r in greedy_select(regions, b)}
            dp = {(r.x, r.y) for r in dp_select(regions, b)}
            if greedy != dp:
                differences += 1
    return {"total_combos": total, "differences": differences, "matches_original_finding": differences == 0}


def ablation_early_vs_late_discard(samples) -> dict:
    """Original finding (a239a89, 5 samples): pre_downscale=0.25 changed
    top-4 regions on 3/5 samples. Re-run at 40 samples."""
    changed = 0
    total = 0
    overlaps = []
    for s in samples:
        try:
            regions_late = propose_regions(s.image_path, grid=4, pre_downscale=1.0)
            regions_early = propose_regions(s.image_path, grid=4, pre_downscale=0.25)
        except Exception as e:
            print(f"  [early_vs_late] skip {s.image_path}: {type(e).__name__}: {e}")
            continue
        total += 1
        top4_late = {(r.x, r.y) for r in sorted(regions_late, key=lambda r: -r.value)[:4]}
        top4_early = {(r.x, r.y) for r in sorted(regions_early, key=lambda r: -r.value)[:4]}
        overlap = len(top4_late & top4_early) / 4
        overlaps.append(overlap)
        if top4_late != top4_early:
            changed += 1
    return {
        "total_samples": total,
        "changed": changed,
        "changed_fraction": changed / total if total else None,
        "mean_top4_overlap_when_changed": (
            float(np.mean([o for o in overlaps if o < 1.0])) if any(o < 1.0 for o in overlaps) else None
        ),
    }


def ablation_saliency_method(samples) -> dict:
    """Original finding (d896267, 1 sample): fine_grained shows stronger
    dense-vs-blank discrimination (1.42x vs 1.24x). Re-run at 40 samples,
    averaging the ratio across the whole corpus, not one sample."""
    ratios_spectral = []
    ratios_fine = []
    for s in samples:
        try:
            sal_spectral = saliency_map(s.image_path, method="spectral_residual")
            sal_fine = saliency_map(s.image_path, method="fine_grained")
        except Exception as e:
            print(f"  [saliency_method] skip {s.image_path}: {type(e).__name__}: {e}")
            continue
        h, w = sal_spectral.shape
        # Approximate "dense text" region as center 60%, "blank margin" as outer 20% border -
        # a coarse proxy (documents aren't guaranteed to have this layout), stated as such.
        margin_h, margin_w = int(h * 0.1), int(w * 0.1)
        border_mask = np.ones((h, w), dtype=bool)
        border_mask[margin_h:h - margin_h, margin_w:w - margin_w] = False
        center_mask = ~border_mask

        for sal, bucket in [(sal_spectral, ratios_spectral), (sal_fine, ratios_fine)]:
            center_mean = float(sal[center_mask].mean())
            border_mean = float(sal[border_mask].mean())
            if border_mean > 1e-9:
                bucket.append(center_mean / border_mean)

    return {
        "n_samples_used": len(ratios_spectral),
        "mean_ratio_spectral_residual": float(np.mean(ratios_spectral)) if ratios_spectral else None,
        "mean_ratio_fine_grained": float(np.mean(ratios_fine)) if ratios_fine else None,
        "fine_grained_higher_on_average": (
            float(np.mean(ratios_fine)) > float(np.mean(ratios_spectral))
            if ratios_spectral and ratios_fine else None
        ),
    }


def run() -> dict:
    print(f"Loading {N_SAMPLES} real, already-downloaded DocVQA samples (no API calls)...")
    samples = load_docvqa(limit=N_SAMPLES)
    print(f"Loaded {len(samples)} real samples.\n")

    print("=== Ablation 1: DP vs greedy region selection ===")
    r1 = ablation_dp_vs_greedy(samples)
    print(r1)

    print("\n=== Ablation 2: early vs late discard ===")
    r2 = ablation_early_vs_late_discard(samples)
    print(r2)

    print("\n=== Ablation 3: saliency method (spectral_residual vs fine_grained) ===")
    r3 = ablation_saliency_method(samples)
    print(r3)

    return {
        "_meta": {"n_samples": len(samples), "cost_usd": 0, "purpose": "Re-verify region-selection ablations at larger real-data scale (was 5-10 samples, now 40)"},
        "dp_vs_greedy": r1,
        "early_vs_late_discard": r2,
        "saliency_method": r3,
    }


if __name__ == "__main__":
    import json
    result = run()
    out = Path("results/region_ablations_rerun.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out}")

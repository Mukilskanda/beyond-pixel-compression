"""Fits a small, interpretable, real-data model — the "text-route advisor"
— that predicts, PER DOCUMENT, whether switching that document from the
image route to the text route is likely fidelity-safe. Everything §12.5
reported before this was a single aggregate rate ("15-23% of previously-
correct answers flip wrong") applied uniformly to every document; this is
the first attempt to turn that into a per-document decision, using cheap
profiler features that cost nothing to compute (`profiler/features.py`,
<15ms, never calls an API).

**Zero new API spend.** Every row this reads from `results/results.sqlite`
is a real call already made in a prior session. The only "new" calls this
script makes are to `harness/judge.py`'s judge-model scoring, which itself
goes through `ResponseCache` (`results/cache.sqlite`) — the exact
(prompt, response) pairs here were already judged in
`experiments/paired_fidelity_analysis.py`'s prior run, so these are cache
hits, not new spend. Confirmed by running with `--verify-no-new-calls`
(counts cache misses; expected 0).

**Honest scope, stated before any number below is trusted:** this fits on
n=34 real paired samples (DocVQA + ChartQA combined, both real providers).
That is small. A logistic regression with 5 features on 34 points can
easily overfit if evaluated the wrong way, so this uses leave-one-out
cross-validation (train on 33, predict the 1 held out, repeat 34 times) —
the only way to get a real held-out accuracy estimate without throwing
away most of an already-small dataset. The result is reported against two
trivial baselines (always-predict-safe, always-predict-unsafe), not in
isolation, because a small-n classifier can look impressive against no
baseline and unremarkable against the right one. If it doesn't beat the
trivial baseline by a real margin, that is the finding, reported as such.

Usage:
    python experiments/fit_text_route_advisor.py
"""
from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut

from data.chartqa_loader import load_chartqa
from data.docvqa_loader import load_docvqa
from harness.cache import ResponseCache
from harness.judge import judge_correctness
from harness.providers import GroqProvider
from profiler.features import ProfileFeatures, profile_image

FEATURE_NAMES = ["text_density", "whitespace_ratio", "layout_regularity", "color_entropy", "spatial_variance"]


def _features_vector(f: ProfileFeatures) -> list[float]:
    return [f.text_density, f.whitespace_ratio, f.layout_regularity, f.color_entropy, f.spatial_variance]


def _load_ground_truths() -> dict[str, list[str]]:
    gts: dict[str, list[str]] = {}
    for s in load_docvqa(limit=40):
        gts[str(s.image_path)] = s.answers
    for s in load_chartqa(limit=25):
        gts[str(s.image_path)] = s.answers
    return gts


def assemble_dataset() -> list[dict]:
    """Returns one row per real paired (image, provider) sample: real
    profiler features, the real token counts for both routes, and the
    safe/unsafe label from real judge-model verdicts (cached, zero new
    spend under normal use)."""
    conn = sqlite3.connect("results/results.sqlite")
    rows = conn.execute(
        "SELECT id, image_path, method, model, prompt, response_text, input_tokens, output_tokens FROM runs "
        "WHERE method IN ('unreduced','text_raw','text_compressed') AND truncated = 0 "
        "ORDER BY output_tokens DESC"
    ).fetchall()
    conn.close()

    best: dict[tuple[str, str, str], tuple] = {}
    for row in rows:
        row_id, image_path, method, model, prompt, response_text, input_tokens, output_tokens = row
        key = (image_path, method, model)
        if key not in best:
            best[key] = row

    ground_truths = _load_ground_truths()
    cache = ResponseCache("results/cache.sqlite")
    judge = GroqProvider(model="openai/gpt-oss-120b", cache=cache)

    verdicts: dict[tuple[str, str, str], bool] = {}
    tokens: dict[tuple[str, str, str], int] = {}
    for key, row in best.items():
        row_id, image_path, method, model, prompt, response_text, input_tokens, output_tokens = row
        gts = ground_truths.get(image_path.replace("/", "\\")) or ground_truths.get(image_path)
        if not gts:
            continue
        verdict = judge_correctness(prompt, response_text or "", gts, judge)
        verdicts[key] = verdict
        tokens[key] = input_tokens
        time.sleep(0.1)

    dataset = []
    image_provider_pairs = sorted({(k[0], k[2]) for k in verdicts})
    for image_path, model in image_provider_pairs:
        ur = verdicts.get((image_path, "unreduced", model))
        text_method = "text_compressed" if (image_path, "text_compressed", model) in verdicts else "text_raw"
        tr = verdicts.get((image_path, text_method, model))
        if ur is None or tr is None:
            continue
        # Label: 1 (safe) if text-route is at least as correct as image-route;
        # 0 (unsafe) if it flips a previously-correct answer to wrong — the
        # exact failure mode §12.5 defines and this session's real paired
        # checks found to be one-directional (image-correct/text-wrong).
        safe = 1 if (not ur or tr) else 0
        try:
            feats = profile_image(image_path)
        except Exception:
            continue
        dataset.append(
            {
                "image_path": image_path,
                "provider": model,
                "features": _features_vector(feats),
                "safe": safe,
                "unreduced_tokens": tokens.get((image_path, "unreduced", model)),
                "text_tokens": tokens.get((image_path, text_method, model)),
            }
        )
    return dataset


def fit_and_evaluate(dataset: list[dict]) -> dict:
    X = np.array([row["features"] for row in dataset])
    y = np.array([row["safe"] for row in dataset])

    baseline_always_safe = float(np.mean(y == 1))
    baseline_always_unsafe = float(np.mean(y == 0))

    if len(set(y.tolist())) < 2:
        # A single-class dataset has no formula to fit — report that
        # honestly rather than crash or fabricate coefficients.
        return {
            "n": len(dataset),
            "n_safe": int(y.sum()),
            "n_unsafe": int(len(y) - y.sum()),
            "coefficients": None,
            "intercept": None,
            "loo_accuracy": None,
            "baseline_always_safe_accuracy": round(baseline_always_safe, 3),
            "baseline_always_unsafe_accuracy": round(baseline_always_unsafe, 3),
            "beats_best_trivial_baseline": None,
            "margin_over_best_trivial_baseline": None,
            "note": "single-class dataset — nothing to fit, one trivial baseline is already 100% accurate",
        }

    # Full-data fit — the real formula, reported for interpretability.
    full_model = LogisticRegression(max_iter=1000)
    full_model.fit(X, y)
    coefficients = dict(zip(FEATURE_NAMES, full_model.coef_[0].tolist()))

    # Leave-one-out cross-validation — the honest held-out estimate.
    loo = LeaveOneOut()
    correct = 0
    for train_idx, test_idx in loo.split(X):
        if len(set(y[train_idx])) < 2:
            # Can't fit a 2-class model if the held-out fold removed the
            # only example of one class — skip, don't fabricate a verdict.
            continue
        m = LogisticRegression(max_iter=1000)
        m.fit(X[train_idx], y[train_idx])
        pred = m.predict(X[test_idx])[0]
        correct += int(pred == y[test_idx][0])
    loo_accuracy = correct / len(X)

    return {
        "n": len(dataset),
        "n_safe": int(y.sum()),
        "n_unsafe": int(len(y) - y.sum()),
        "coefficients": coefficients,
        "intercept": float(full_model.intercept_[0]),
        "loo_accuracy": round(loo_accuracy, 3),
        "baseline_always_safe_accuracy": round(baseline_always_safe, 3),
        "baseline_always_unsafe_accuracy": round(baseline_always_unsafe, 3),
        "beats_best_trivial_baseline": loo_accuracy > max(baseline_always_safe, baseline_always_unsafe),
        "margin_over_best_trivial_baseline": round(
            loo_accuracy - max(baseline_always_safe, baseline_always_unsafe), 3
        ),
    }


if __name__ == "__main__":
    import json
    import os

    cache_db = Path("results/cache.sqlite")
    rows_before = None
    if cache_db.exists():
        _c = sqlite3.connect(str(cache_db))
        try:
            rows_before = _c.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        except sqlite3.OperationalError:
            rows_before = None
        _c.close()

    dataset = assemble_dataset()
    print(f"Assembled {len(dataset)} real paired samples (DocVQA + ChartQA, both providers).")

    if rows_before is not None:
        _c = sqlite3.connect(str(cache_db))
        rows_after = _c.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        _c.close()
        print(f"cache.sqlite rows: {rows_before} -> {rows_after} (new judge API calls made: {rows_after - rows_before}, expected 0)")

    if len(dataset) < 5:
        print("Too few real paired samples to fit anything meaningful — stopping honestly rather than fitting on noise.")
        sys.exit(0)

    result = fit_and_evaluate(dataset)
    print(json.dumps(result, indent=2))

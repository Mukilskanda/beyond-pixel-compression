"""Visual semantic cache — safe-vs-unsafe cache-hit analysis on real,
already-downloaded DocVQA data. Zero provider API calls (Gemini/Groq/
Anthropic): this only uses (1) real DocVQA images, downloaded free via
HuggingFace `datasets` (same mechanism data/docvqa_loader.py always uses),
and (2) the real, human-annotated ground-truth answers that ship with the
dataset itself.

IMPORTANT SCOPE NOTE, stated explicitly rather than glossed over: this
measures whether visual similarity predicts GROUND-TRUTH-answer agreement
across image pairs - i.e. "if a prior real API call had answered image A's
question, and a new query asks about visually-similar image B, would A's
(ground-truth-correct) answer also be correct for B's question?" It does
NOT yet measure whether a real MODEL's actual output would transfer
correctly (that needs real model answers as cache entries, and this
project currently has only ~13 real model answers total across both
providers - too few for a real threshold curve on its own). This is the
zero-cost half of the caching experiment; the model-answer-based
validation is separate future work gated on API quota (see draft.md §11
for the honest split).

Metric for "would the cached answer be correct for the new query": ANLS
(harness/metrics.anls) between the cached image's first ground-truth
answer and the new image's real ground-truth answer set - the same metric
this project already uses for real fidelity measurement, not a new one
invented for this analysis.
"""

from __future__ import annotations

import json
import sys
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from data.docvqa_loader import load_docvqa
from harness.metrics import anls
from pixelpath.cache_similarity import _saliency_signature

N_SAMPLES = 40
THRESHOLDS = [0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]


def run() -> dict:
    print(f"Loading {N_SAMPLES} real, already-cached DocVQA samples (no API calls)...")
    samples = load_docvqa(limit=N_SAMPLES)
    print(f"Loaded {len(samples)} real samples with real ground-truth answers.")

    print("Computing per-image saliency signatures once each (not per-pair)...")
    signatures = {s.question_id: _saliency_signature(s.image_path) for s in samples}
    print(f"Computed {len(signatures)} signatures.")

    print("Computing pairwise similarity + would-cache-be-correct labels...")
    pairs = []
    for cached, query in permutations(samples, 2):
        sim = float(np.dot(signatures[cached.question_id], signatures[query.question_id]))
        cached_answer = cached.answers[0]
        correctness = anls(cached_answer, query.answers)
        pairs.append(
            {
                "cached_qid": cached.question_id,
                "query_qid": query.question_id,
                "similarity": sim,
                "cached_answer_would_be_correct": correctness >= 0.5,
                "anls_if_reused": correctness,
            }
        )

    print(f"{len(pairs)} ordered (cached, query) pairs from {len(samples)} samples.")

    curve = []
    for t in THRESHOLDS:
        hits = [p for p in pairs if p["similarity"] >= t]
        hit_rate = len(hits) / len(pairs) if pairs else 0.0
        correctness_retained = (
            sum(1 for p in hits if p["cached_answer_would_be_correct"]) / len(hits)
            if hits
            else None  # no hits at this threshold - undefined, not zero
        )
        curve.append(
            {
                "threshold": t,
                "n_hits": len(hits),
                "hit_rate": hit_rate,
                "correctness_retained": correctness_retained,
            }
        )
        cr_str = f"{correctness_retained:.3f}" if correctness_retained is not None else "N/A (no hits)"
        print(f"  threshold={t}: hit_rate={hit_rate:.4f} ({len(hits)} hits), correctness_retained={cr_str}")

    result = {
        "_meta": {
            "purpose": "Safe-vs-unsafe visual semantic cache hit analysis, ground-truth-based (zero API spend)",
            "n_samples": len(samples),
            "n_pairs": len(pairs),
            "similarity_metric": "cosine similarity of 16x16 saliency-map signatures (pixelpath/cache_similarity.py)",
            "correctness_metric": "ANLS(cached_answer, query_ground_truths) >= 0.5 (harness/metrics.anls, same metric used for real fidelity measurement elsewhere in this project)",
            "cost_usd": 0,
            "scope_caveat": "Uses ground-truth answers as cache-entry proxies, NOT real model answers - see module docstring. Real-model-answer validation is separate, quota-gated future work.",
        },
        "curve": curve,
        "pairs": pairs,
    }

    out_path = Path("results/cache_safety_analysis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}")
    return result


if __name__ == "__main__":
    run()

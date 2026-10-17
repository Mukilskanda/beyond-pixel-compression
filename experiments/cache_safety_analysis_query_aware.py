"""Query-aware version of experiments/cache_safety_analysis.py (paper/draft.md
§11.4) - same 40 real DocVQA samples, same 1,560 ordered pairs, same
threshold sweep, so the two curves are directly comparable. The only change
is the similarity signature: combined_score = min(image_sim, question_sim)
instead of image_sim alone, requiring BOTH axes to clear the threshold for
a cache hit (§11.2/§11.3's image-only cache is left unmodified and
unre-run, so that negative result stays exactly reproducible).

Zero provider API calls - same DocVQA-only, ground-truth-answer-as-proxy
scope caveat as the original analysis (see that script's docstring).
Question similarity uses TF-IDF (scikit-learn, local, zero network) - see
pixelpath/cache_similarity.QuestionSimilarity docstring for why TF-IDF
specifically, not a downloaded embedding model.
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
from pixelpath.cache_similarity import QuestionSimilarity, _saliency_signature

N_SAMPLES = 40
# Extended down from the original [0.90, ..., 0.999] sweep (§11.2/§11.3):
# that range produced zero hits for the combined min(image_sim, question_sim)
# signature (max observed combined similarity in this corpus is ~0.80,
# found only after running the original sweep and seeing an all-zero
# curve - not assumed in advance). Lower thresholds included so the real
# shape of the curve is visible, not just "always zero."
THRESHOLDS = [0.50, 0.60, 0.65, 0.70, 0.75, 0.78, 0.80, 0.90, 0.95, 0.99]


def run() -> dict:
    print(f"Loading {N_SAMPLES} real, already-cached DocVQA samples (no API calls)...")
    samples = load_docvqa(limit=N_SAMPLES)
    print(f"Loaded {len(samples)} real samples with real ground-truth answers.")

    print("Computing per-image saliency signatures once each...")
    img_signatures = {s.question_id: _saliency_signature(s.image_path) for s in samples}

    print("Fitting TF-IDF question-similarity over the 40 real questions...")
    question_sim = QuestionSimilarity()
    question_sim.fit([s.question for s in samples])

    print("Computing combined (image AND question) similarity + would-cache-be-correct labels...")
    pairs = []
    for cached, query in permutations(samples, 2):
        img_sim = float(np.dot(img_signatures[cached.question_id], img_signatures[query.question_id]))
        q_sim = question_sim.similarity(cached.question, query.question)
        combined = min(img_sim, q_sim)
        cached_answer = cached.answers[0]
        correctness = anls(cached_answer, query.answers)
        pairs.append(
            {
                "cached_qid": cached.question_id,
                "query_qid": query.question_id,
                "image_similarity": img_sim,
                "question_similarity": q_sim,
                "combined_similarity": combined,
                "cached_answer_would_be_correct": correctness >= 0.5,
                "anls_if_reused": correctness,
            }
        )

    print(f"{len(pairs)} ordered (cached, query) pairs from {len(samples)} samples.")

    curve = []
    for t in THRESHOLDS:
        # a hit requires BOTH image_sim >= t AND question_sim >= t, i.e.
        # combined_similarity (= min of the two) >= t
        hits = [p for p in pairs if p["combined_similarity"] >= t]
        hit_rate = len(hits) / len(pairs) if pairs else 0.0
        correctness_retained = sum(1 for p in hits if p["cached_answer_would_be_correct"]) / len(hits) if hits else None
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
            "purpose": "Query-aware (image AND question) cache hit analysis, same 40 samples/1560 pairs as the image-only analysis (results/cache_safety_analysis.json), zero API spend",
            "n_samples": len(samples),
            "n_pairs": len(pairs),
            "similarity_metric": "min(image cosine similarity of 16x16 saliency signatures, question cosine similarity of TF-IDF vectors)",
            "correctness_metric": "ANLS(cached_answer, query_ground_truths) >= 0.5",
            "cost_usd": 0,
            "scope_caveat": "Uses ground-truth answers as cache-entry proxies, NOT real model answers - same scope as the image-only analysis.",
        },
        "curve": curve,
        "pairs": pairs,
    }

    out_path = Path("results/cache_safety_analysis_query_aware.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}")
    return result


if __name__ == "__main__":
    run()

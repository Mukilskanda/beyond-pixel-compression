"""Regression check against the query-aware cache-safety results
(tests/fixtures/cache_safety_analysis_query_aware_2026-08-16.json,
produced by experiments/cache_safety_analysis_query_aware.py) so §11.4's
partial-fix finding doesn't silently drift. Same 40 real DocVQA
samples/1560 pairs as the image-only fixture, zero API spend.
"""

import json
from pathlib import Path

import pytest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "cache_safety_analysis_query_aware_2026-08-16.json"

pytestmark = pytest.mark.skipif(
    not FIXTURE_PATH.exists(),
    reason="tests/fixtures/cache_safety_analysis_query_aware_2026-08-16.json not present - "
    "run experiments/cache_safety_analysis_query_aware.py to regenerate",
)


def load_fixture():
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_image_only_fixture():
    path = Path(__file__).parent / "fixtures" / "cache_safety_analysis_2026-08-16.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_fixture_is_zero_spend():
    data = load_fixture()
    assert data["_meta"]["cost_usd"] == 0


def test_same_sample_and_pair_count_as_image_only_analysis():
    """§11.4's curve must be directly comparable to §11.2's - same corpus,
    same pair count, or the comparison in draft.md is invalid."""
    query_aware = load_fixture()
    image_only = load_image_only_fixture()
    assert query_aware["_meta"]["n_samples"] == image_only["_meta"]["n_samples"]
    assert query_aware["_meta"]["n_pairs"] == image_only["_meta"]["n_pairs"]


def test_hit_rate_never_exceeds_image_only_hit_rate_at_shared_thresholds():
    """Requiring BOTH axes to clear a threshold can only be more
    restrictive than image similarity alone - hit rate at any shared
    threshold must be <= the image-only hit rate at that threshold."""
    query_aware = load_fixture()
    image_only = load_image_only_fixture()
    io_by_threshold = {r["threshold"]: r["hit_rate"] for r in image_only["curve"]}
    for row in query_aware["curve"]:
        if row["threshold"] in io_by_threshold:
            assert row["hit_rate"] <= io_by_threshold[row["threshold"]] + 1e-9


def test_paraphrase_collision_case_is_wrong_at_its_own_highest_similarity():
    """§11.4's specific negative sub-finding: the highest-combined-similarity
    pair in the whole dataset (same fill-in-template question differing by
    one contrastive word, e.g. 'current year' vs 'previous year') is STILL
    a wrong cache hit - query-awareness does not fix this particular case,
    it's the inverse failure mode of the template-collision problem."""
    data = load_fixture()
    top_pair = max(data["pairs"], key=lambda p: p["combined_similarity"])
    assert top_pair["combined_similarity"] > 0.7  # confirms this is a real near-match, not noise
    assert top_pair["cached_answer_would_be_correct"] is False


def test_zero_shared_vocabulary_pairs_are_correctly_excluded():
    """The majority-fix part of §11.4: pairs with completely unrelated
    questions (zero TF-IDF overlap) must never register as a potential
    cache hit, unlike the image-only signature which conflated them via
    template similarity alone."""
    data = load_fixture()
    zero_vocab_pairs = [p for p in data["pairs"] if p["question_similarity"] == 0.0]
    assert len(zero_vocab_pairs) > 0  # sanity: such pairs exist in this corpus
    assert all(p["combined_similarity"] == 0.0 for p in zero_vocab_pairs)

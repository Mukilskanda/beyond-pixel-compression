"""Regression check against the real cache-safety analysis results
(results/cache_safety_analysis.json, produced by
experiments/cache_safety_analysis.py on 40 real, zero-cost DocVQA samples)
so the negative finding in paper/draft.md §11 doesn't silently drift or
require re-running to re-verify. Does NOT call any real API - the analysis
itself is zero-spend (real DocVQA download + ground-truth answers only).
"""
import json
from pathlib import Path

import pytest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "cache_safety_analysis_2026-08-16.json"

pytestmark = pytest.mark.skipif(
    not FIXTURE_PATH.exists(),
    reason="tests/fixtures/cache_safety_analysis_2026-08-16.json not present - run experiments/cache_safety_analysis.py to regenerate",
)


def load_fixture():
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_fixture_is_zero_spend():
    data = load_fixture()
    assert data["_meta"]["cost_usd"] == 0


def test_fixture_uses_ground_truth_proxy_not_real_model_answers():
    """Locks in the honest scope caveat so it can't silently be dropped from
    the fixture without also being noticed here."""
    data = load_fixture()
    assert "ground-truth" in data["_meta"]["scope_caveat"].lower()


def test_correctness_retained_stays_low_even_at_high_similarity():
    """The headline negative finding (§11.2): even near-perfect similarity
    does not reliably predict a safe cache hit on real DocVQA data. If this
    ever flips to a high correctness-retained value, that's a real change
    worth investigating and reporting, not silently accepting."""
    data = load_fixture()
    highest_threshold_row = max(data["curve"], key=lambda r: r["threshold"])
    assert highest_threshold_row["correctness_retained"] is not None
    assert highest_threshold_row["correctness_retained"] < 0.5, (
        "correctness_retained at the highest tested threshold is no longer "
        "low - this contradicts §11.2's negative finding and must be "
        "investigated (and draft.md updated) before trusting this number"
    )


def test_hit_rate_decreases_monotonically_with_threshold():
    data = load_fixture()
    curve = sorted(data["curve"], key=lambda r: r["threshold"])
    hit_rates = [r["hit_rate"] for r in curve]
    assert all(a >= b for a, b in zip(hit_rates, hit_rates[1:]))

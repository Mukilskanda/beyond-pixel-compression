"""Regression check against the real region-ablation re-run results
(results/region_ablations_rerun.json is regenerable via
experiments/rerun_region_ablations.py; a copy is kept in tests/fixtures/
for a fixture that doesn't require re-downloading DocVQA to test against).
Zero API calls involved anywhere in this ablation - purely local
saliency/region-selection code."""
import json
from pathlib import Path

import pytest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "region_ablations_rerun_2026-08-16.json"

pytestmark = pytest.mark.skipif(
    not FIXTURE_PATH.exists(),
    reason="fixture not present - run experiments/rerun_region_ablations.py to regenerate",
)


def load_fixture():
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_zero_cost():
    data = load_fixture()
    assert data["_meta"]["cost_usd"] == 0


def test_dp_vs_greedy_still_identical_on_flat_cost_regions():
    data = load_fixture()
    assert data["dp_vs_greedy"]["matches_original_finding"] is True
    assert data["dp_vs_greedy"]["differences"] == 0


def test_early_vs_late_discard_still_shows_real_effect():
    """Original finding: a real, measurable effect (not noise). At 40
    samples this must still be a meaningfully nonzero fraction, not
    collapse to ~0% (which would contradict the original claim)."""
    data = load_fixture()
    assert data["early_vs_late_discard"]["changed_fraction"] > 0.3


def test_fine_grained_still_shows_stronger_discrimination_on_average():
    data = load_fixture()
    assert data["saliency_method"]["fine_grained_higher_on_average"] is True

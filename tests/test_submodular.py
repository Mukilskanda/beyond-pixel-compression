"""Regression tests for pixelpath/submodular.py, including an empirical
check that greedy achieves at least (1 - 1/e) of the brute-force optimum on
small synthetic instances with deliberately overlapping regions (where the
distinction from modular greedy_select actually matters)."""
import math

import numpy as np
import pytest

from pixelpath.regions import Region
from pixelpath.submodular import (
    brute_force_optimal,
    submodular_coverage_value,
    submodular_greedy_select,
)

GUARANTEE = 1 - 1 / math.e  # ~0.632


def _uniform_saliency(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=float)


def test_no_overlap_selects_all_within_budget():
    sal = _uniform_saliency(10, 10)
    regions = [
        Region(x=0, y=0, w=5, h=5, value=25, cost=10),
        Region(x=5, y=0, w=5, h=5, value=25, cost=10),
    ]
    selected = submodular_greedy_select(regions, budget=20, saliency_map=sal)
    assert len(selected) == 2


def test_fully_overlapping_region_contributes_nothing_marginally():
    sal = _uniform_saliency(10, 10)
    full = Region(x=0, y=0, w=10, h=10, value=100, cost=10)
    subset = Region(x=0, y=0, w=5, h=5, value=25, cost=10)
    selected = submodular_greedy_select([full, subset], budget=20, saliency_map=sal)
    # full covers everything the subset would; subset should not be picked
    # once full is selected (marginal value 0), even though budget allows it.
    assert full in selected
    assert subset not in selected


def test_budget_respected():
    sal = _uniform_saliency(20, 20)
    regions = [Region(x=i * 4, y=0, w=4, h=20, value=80, cost=30) for i in range(5)]
    selected = submodular_greedy_select(regions, budget=60, saliency_map=sal)
    assert sum(r.cost for r in selected) <= 60


def test_coverage_value_does_not_double_count_overlap():
    sal = _uniform_saliency(10, 10)
    a = Region(x=0, y=0, w=10, h=10, value=100, cost=10)
    b = Region(x=0, y=0, w=10, h=10, value=100, cost=10)  # identical region
    assert submodular_coverage_value([a, b], sal) == 100.0  # not 200


def test_negative_budget_raises():
    with pytest.raises(ValueError):
        submodular_greedy_select([], budget=-1, saliency_map=_uniform_saliency(4, 4))


@pytest.mark.parametrize("seed", range(5))
def test_greedy_meets_1_minus_1_over_e_bound_on_overlapping_instances(seed):
    """The actual claim this module makes: greedy total value >= (1-1/e) *
    true optimum, checked empirically against brute force on small random
    overlapping instances (n<=6 regions, tractable to brute force)."""
    rng = np.random.default_rng(seed)
    h, w = 12, 12
    sal = rng.random((h, w)) * 10

    regions = []
    for _ in range(6):
        rw, rh = int(rng.integers(3, 8)), int(rng.integers(3, 8))
        rx, ry = int(rng.integers(0, w - rw)), int(rng.integers(0, h - rh))
        cost = int(rng.integers(5, 15))
        regions.append(Region(x=rx, y=ry, w=rw, h=rh, value=0.0, cost=cost))

    budget = 25
    greedy_value = submodular_coverage_value(
        submodular_greedy_select(regions, budget, sal), sal
    )
    opt_value = brute_force_optimal(regions, budget, sal)

    if opt_value == 0:
        assert greedy_value == 0
    else:
        assert greedy_value >= GUARANTEE * opt_value - 1e-9, (
            f"seed={seed}: greedy={greedy_value:.3f} fell below "
            f"(1-1/e)*opt={GUARANTEE * opt_value:.3f} (opt={opt_value:.3f})"
        )

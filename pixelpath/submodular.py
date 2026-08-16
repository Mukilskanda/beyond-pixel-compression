"""Submodular region selection (Block D / Step 3 item 8).

pixelpath/regions.py's existing greedy_select/dp_select operate on a MODULAR
(additive, no overlap discount) value function - region.value is just the
sum of saliency within that region, independent of what else is selected.
Greedy-by-ratio is optimal for a modular value function under a knapsack
constraint only in the equal-cost case (see dp_select's docstring); it has
no formal approximation guarantee against a variable-cost knapsack for a
modular function in general.

This module implements a genuinely SUBMODULAR value function instead:
marginal value of adding a region is the saliency mass NOT ALREADY COVERED
by previously selected regions (diminishing returns as regions start to
overlap in what they capture) - the standard "coverage" submodular function.
For a monotone submodular function under a knapsack constraint, greedy-by-
marginal-ratio is proven (Sviridenko 2004 / the classic Nemhauser-Wolsey-
Fisher cardinality result, extended to knapsack) to achieve at least
(1 - 1/e) ≈ 0.632 of the optimal value. `submodular_greedy_select` below
implements exactly that: recompute marginal value after each pick, not just
sort once.

FRAMING (per the reframe, paper/draft.md §3.4/§9): like water_filling.py,
this is validated machinery for a hypothetical content-sensitive-pricing
provider. It is not shown to reduce real cost on either tested provider
(Gemini, Groq both show flat per-image cost regardless of region choice).
Built and tested on its own mathematical/algorithmic merits.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from pixelpath.regions import Region


def _region_mask(region: Region, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[region.y:region.y + region.h, region.x:region.x + region.w] = True
    return mask


def submodular_greedy_select(
    regions: list[Region],
    budget: int,
    saliency_map: np.ndarray,
) -> list[Region]:
    """Greedy submodular knapsack selection: repeatedly pick the region with
    the highest (marginal newly-covered saliency mass / cost) ratio, where
    "marginal" accounts for overlap with already-selected regions - a
    region that mostly overlaps an already-picked region contributes little
    even if its raw saliency sum is high.

    This is NOT the same algorithm as pixelpath.regions.greedy_select, which
    uses each region's static (non-marginal) value/cost ratio and never
    revisits it. That distinction is exactly what makes this submodular
    (diminishing marginal returns) rather than modular.
    """
    if budget < 0:
        raise ValueError(f"budget must be non-negative, got {budget}")

    covered = np.zeros(saliency_map.shape, dtype=bool)
    remaining = list(regions)
    selected: list[Region] = []
    spent = 0

    while remaining:
        best_region = None
        best_ratio = -1.0
        best_marginal = 0.0
        for r in remaining:
            if spent + r.cost > budget:
                continue
            mask = _region_mask(r, saliency_map.shape)
            new_mask = mask & ~covered
            marginal = float(saliency_map[new_mask].sum())
            ratio = marginal / r.cost if r.cost else 0.0
            if ratio > best_ratio:
                best_ratio = ratio
                best_region = r
                best_marginal = marginal

        if best_region is None or best_marginal <= 0:
            break  # no affordable region adds any new coverage

        selected.append(best_region)
        covered |= _region_mask(best_region, saliency_map.shape)
        spent += best_region.cost
        remaining.remove(best_region)

    return selected


def submodular_coverage_value(regions: list[Region], saliency_map: np.ndarray) -> float:
    """Total value of a region SET under the coverage submodular function:
    union of covered saliency mass, not sum of individual region values
    (which would double-count overlapping regions)."""
    covered = np.zeros(saliency_map.shape, dtype=bool)
    for r in regions:
        covered |= _region_mask(r, saliency_map.shape)
    return float(saliency_map[covered].sum())


def brute_force_optimal(regions: list[Region], budget: int, saliency_map: np.ndarray) -> float:
    """Exact optimum by exhaustive subset search - only tractable for small
    region counts (used in tests to empirically check the (1-1/e) bound,
    not for production use)."""
    best = 0.0
    n = len(regions)
    for k in range(n + 1):
        for combo in combinations(range(n), k):
            subset = [regions[i] for i in combo]
            cost = sum(r.cost for r in subset)
            if cost > budget:
                continue
            value = submodular_coverage_value(subset, saliency_map)
            best = max(best, value)
    return best

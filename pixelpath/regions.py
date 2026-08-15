"""Stage 3b pixel path: budgeted region selection as knapsack, per plan §Stage 3b.

v1 baseline: greedy by value/cost ratio, on a fixed grid of candidate regions.
DP knapsack and provider-tiling-aware region merging (packing.py) are week-6
upgrades — greedy is the thing to beat, not the final answer.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pixelpath.saliency import saliency_map


@dataclass
class Region:
    x: int
    y: int
    w: int
    h: int
    value: float  # sum of saliency within the region
    cost: int      # token cost estimate (area-proportional placeholder — see packing.py)

    @property
    def ratio(self) -> float:
        return self.value / self.cost if self.cost else 0.0


def propose_regions(image_path: str | Path, grid: int = 4, tokens_per_cell: int = 85) -> list[Region]:
    """Partition the image into a grid×grid set of candidate crop regions.

    `tokens_per_cell` is a placeholder linear cost model — real provider tiling
    is a step function (see plan's knapsack note); replace via packing.py.
    """
    sal = saliency_map(image_path)
    h, w = sal.shape
    gh, gw = h // grid, w // grid

    regions = []
    for gy in range(grid):
        for gx in range(grid):
            y0, x0 = gy * gh, gx * gw
            y1 = h if gy == grid - 1 else y0 + gh
            x1 = w if gx == grid - 1 else x0 + gw
            cell = sal[y0:y1, x0:x1]
            regions.append(Region(x=x0, y=y0, w=x1 - x0, h=y1 - y0, value=float(cell.sum()), cost=tokens_per_cell))

    return regions


def greedy_select(regions: list[Region], budget: int) -> list[Region]:
    """Baseline knapsack solver: sort by value/cost ratio, take until budget exhausted."""
    ordered = sorted(regions, key=lambda r: r.ratio, reverse=True)
    selected = []
    spent = 0
    for r in ordered:
        if spent + r.cost <= budget:
            selected.append(r)
            spent += r.cost
    return selected

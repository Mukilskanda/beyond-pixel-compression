"""Stage 3c: per-image budget allocation, predicted from Stage 1 features.

Plan §Stage 3c: "Prior work applies a fixed pruning ratio to all inputs; the
literature explicitly flags this as a limitation. Predict per-image difficulty
from Stage 1 features and set B accordingly." This is a direct answer to a
stated open problem in the literature — worth getting right, not just stubbing.

v1: hand-tuned heuristic (this file). A learned regressor trained on the §4
label-generation sweep is the natural v2 upgrade, same relationship as
router/rules.py -> router/learned.py.
"""

from __future__ import annotations

from profiler.features import ProfileFeatures

# Token budget bounds — calibrated against pixelpath/regions.py's default
# tokens_per_cell=85 on a 4x4 grid (16 cells => max useful budget ~1360).
MIN_BUDGET = 200
MAX_BUDGET = 1200


def _difficulty_score(features: ProfileFeatures) -> float:
    """Higher score = image needs more tokens to answer correctly.

    Heuristic: high spatial variance (busy, detailed image) and high color
    entropy (rich content) both suggest the model needs more visual detail to
    do the task; high whitespace_ratio suggests most of the image is
    uninformative and can be pruned hard regardless of the other signals.
    """
    # Normalize spatial_variance (unbounded, image-dependent) via a soft cap —
    # calibrate the divisor against real data in week 8; this is a starting point.
    variance_signal = min(1.0, features.spatial_variance / 2000.0)
    entropy_signal = min(1.0, features.color_entropy / 8.0)  # max entropy for 8x8x8 histogram is 9 bits
    sparsity_discount = features.whitespace_ratio  # 0 (dense) .. 1 (mostly blank)

    raw = 0.5 * variance_signal + 0.5 * entropy_signal
    return raw * (1.0 - 0.6 * sparsity_discount)  # blank images get budget cut hard, never to zero


def allocate_budget(features: ProfileFeatures) -> int:
    """Map Stage 1 features to a token budget in [MIN_BUDGET, MAX_BUDGET]."""
    difficulty = _difficulty_score(features)  # in [0, 1] by construction
    return int(MIN_BUDGET + difficulty * (MAX_BUDGET - MIN_BUDGET))

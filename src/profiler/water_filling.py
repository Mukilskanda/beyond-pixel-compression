"""Entropy-based water-filling budget allocation across a batch of images
sharing a fixed total token pool (Block D / Step 3 item 8).

Distinct from profiler/budget.py's allocate_budget(), which maps ONE image's
features to a budget independently, unbounded by any shared pool.
water_fill_allocate() answers a different question: given a fixed total
token budget shared across N images in a batch, how should it be split so
harder (higher-entropy) images get more, without any image exceeding a
per-image cap? This is the classical water-filling algorithm (as used for
power allocation across channels): raise a shared "water level" until the
total budget is exhausted, capping any image whose level would exceed its
max.

FRAMING (per the reframe, paper/draft.md §3.4/§9): this is validated
machinery for a scenario where a provider's real cost is genuinely
allocable/shared across a request batch or scales with tokens sent - it is
NOT validated to save money on either provider actually tested (Gemini,
Groq), since both showed flat per-image cost regardless of token/pixel
allocation (§9.5). Built and tested on its own mathematical merits, not
because it's known to reduce real billed cost.
"""
from __future__ import annotations


def _entropy_weight(entropy: float, max_entropy: float = 8.0) -> float:
    """Normalize color/feature entropy to [0, 1] - higher entropy = harder
    image = deserves more budget under water-filling."""
    if max_entropy <= 0:
        raise ValueError("max_entropy must be positive")
    return max(0.0, min(1.0, entropy / max_entropy))


def water_fill_allocate(
    entropies: list[float],
    total_budget: int,
    min_per_image: int,
    max_per_image: int,
    max_entropy: float = 8.0,
) -> list[int]:
    """Classical water-filling: each image i has a "base level" derived from
    its entropy weight; water is poured in until `total_budget` is spent,
    raising all levels together but never letting any image exceed
    max_per_image. Returns one allocation per input image, in order.

    Every image gets at least min_per_image if total_budget allows it
    (raises ValueError if the pool can't cover the floor for every image -
    silently under-allocating would be a worse failure than a loud error).
    """
    n = len(entropies)
    if n == 0:
        return []
    if min_per_image > max_per_image:
        raise ValueError("min_per_image must be <= max_per_image")
    floor_total = n * min_per_image
    if total_budget < floor_total:
        raise ValueError(
            f"total_budget ({total_budget}) cannot even cover the {n}-image floor "
            f"of {floor_total} (min_per_image={min_per_image}) - reduce min_per_image, "
            f"increase total_budget, or reduce batch size"
        )

    weights = [_entropy_weight(e, max_entropy) for e in entropies]
    span = max_per_image - min_per_image
    remaining_pool = total_budget - floor_total
    capacity_span_total = n * span

    if capacity_span_total == 0 or remaining_pool <= 0 or all(w == 0.0 for w in weights):
        # every image already at its floor, no headroom to distribute, or
        # every image has zero entropy weight (no extra earned regardless of pool size)
        return [min_per_image] * n

    # Binary search the water level (in extra-token units, not a [0,1]
    # fraction) such that sum(min(span, level * weight_i)) == remaining_pool,
    # or the pool saturates every image at `span` if it's generous enough.
    # `hi` must reach far enough that even the smallest positive weight
    # saturates: level >= span / min(positive weight).
    def allocated_extra(level: float) -> float:
        return sum(min(span, level * w) for w in weights)

    hi_saturate = span / min(w for w in weights if w > 0)
    if allocated_extra(hi_saturate) <= remaining_pool:
        # pool is generous enough to fill every weighted image to max_per_image
        extras = [span if w > 0 else 0.0 for w in weights]
    else:
        lo, hi = 0.0, hi_saturate
        for _ in range(60):  # more than enough for float precision at this budget scale
            mid = (lo + hi) / 2
            if allocated_extra(mid) < remaining_pool:
                lo = mid
            else:
                hi = mid
        level = lo
        extras = [min(span, level * w) for w in weights]

    # Integer allocation: floor each extra, then distribute leftover units
    # (from rounding) to the highest-weight images first so the pool is
    # fully spent, not silently lost to rounding.
    int_extras = [int(e) for e in extras]
    spent_extra = sum(int_extras)
    pool_is_binding = allocated_extra(hi_saturate) > remaining_pool
    leftover = int(remaining_pool - spent_extra) if pool_is_binding else 0
    order = sorted(range(n), key=lambda i: weights[i], reverse=True)
    i = 0
    while leftover > 0 and i < len(order):
        idx = order[i % n]
        if int_extras[idx] < span:
            int_extras[idx] += 1
            leftover -= 1
        i += 1
        if i > n * (span + 1) + 10:  # safety valve, should never trigger given the loop invariant
            break

    return [min_per_image + e for e in int_extras]

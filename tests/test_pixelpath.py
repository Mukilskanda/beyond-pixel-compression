import random

from pixelpath.regions import Region, dp_select, greedy_select, propose_regions
from pixelpath.saliency import saliency_map


def test_saliency_map_shape_matches_image(sample_image):
    from PIL import Image

    img = Image.open(sample_image)
    sal = saliency_map(sample_image)
    assert sal.shape == (img.height, img.width)


def test_saliency_values_in_unit_range(sample_image):
    sal = saliency_map(sample_image)
    assert sal.min() >= 0.0
    assert sal.max() <= 1.0


def test_propose_regions_covers_grid(sample_image):
    regions = propose_regions(sample_image, grid=4)
    assert len(regions) == 16


def test_pre_downscale_output_matches_original_resolution(sample_image):
    """Early-vs-late discard ablation: pre_downscale must not change the
    output shape — only the resolution the saliency computation itself runs at."""
    from PIL import Image

    img = Image.open(sample_image)
    sal_full = saliency_map(sample_image, pre_downscale=1.0)
    sal_early = saliency_map(sample_image, pre_downscale=0.25)
    assert sal_full.shape == sal_early.shape == (img.height, img.width)


def test_pre_downscale_changes_region_proposal_values(sample_image):
    """Computing saliency from less information (early discard) should
    generally produce a different, coarser signal than full-resolution
    (late discard) — verified as a real, non-trivial effect on real DocVQA
    scans (see experiments/ablations.yaml: early_vs_late_discard)."""
    late = propose_regions(sample_image, pre_downscale=1.0)
    early = propose_regions(sample_image, pre_downscale=0.25)
    late_values = [r.value for r in late]
    early_values = [r.value for r in early]
    assert late_values != early_values


def test_greedy_select_respects_budget():
    regions = [
        Region(x=0, y=0, w=10, h=10, value=10.0, cost=50),
        Region(x=10, y=0, w=10, h=10, value=8.0, cost=50),
        Region(x=20, y=0, w=10, h=10, value=1.0, cost=50),
    ]
    selected = greedy_select(regions, budget=100)
    assert sum(r.cost for r in selected) <= 100
    assert len(selected) == 2


def test_greedy_select_prefers_higher_value_ratio():
    high_value = Region(x=0, y=0, w=10, h=10, value=100.0, cost=10)
    low_value = Region(x=10, y=0, w=10, h=10, value=1.0, cost=10)
    selected = greedy_select([low_value, high_value], budget=10)
    assert selected == [high_value]


def test_dp_select_respects_budget():
    regions = [
        Region(x=0, y=0, w=10, h=10, value=10.0, cost=50),
        Region(x=10, y=0, w=10, h=10, value=8.0, cost=50),
        Region(x=20, y=0, w=10, h=10, value=1.0, cost=50),
    ]
    selected = dp_select(regions, budget=100)
    assert sum(r.cost for r in selected) <= 100


def test_dp_select_strictly_beats_greedy_on_stranded_budget_case():
    """The classic knapsack counterexample: greedy takes the best-ratio item
    first, which strands the remaining budget with nothing else fitting.
    DP finds the lower-ratio item that uses the whole budget and scores higher."""
    regions = [
        Region(x=0, y=0, w=1, h=1, value=6.0, cost=5),    # ratio 1.2 (best) — greedy grabs this first
        Region(x=1, y=0, w=1, h=1, value=10.0, cost=10),  # ratio 1.0 — uses the full budget alone
    ]
    budget = 10

    greedy_value = sum(r.value for r in greedy_select(regions, budget))
    dp_value = sum(r.value for r in dp_select(regions, budget))

    assert dp_value > greedy_value
    assert dp_value == 10.0  # DP must find the optimal (single item, full budget used)


def test_dp_select_never_worse_than_greedy_random_cases():
    """DP is provably optimal — across many random region sets, it must
    never score lower than greedy, though it may tie."""
    rng = random.Random(0)
    for _ in range(20):
        regions = [
            Region(x=i, y=0, w=1, h=1, value=rng.uniform(1, 100), cost=rng.choice([20, 40, 60, 85, 130]))
            for i in range(10)
        ]
        budget = rng.choice([100, 200, 300])
        greedy_value = sum(r.value for r in greedy_select(regions, budget))
        dp_value = sum(r.value for r in dp_select(regions, budget))
        assert dp_value >= greedy_value - 1e-9  # float tolerance


def test_dp_select_identical_to_greedy_when_costs_are_equal(sample_image):
    """Documented, verified finding: propose_regions() currently gives every
    region the same flat cost (tokens_per_cell), so ratio-order == value-order
    and greedy is already provably optimal in that special case. dp_select is
    correctly implemented (see the stranded-budget test above) but currently
    dormant in the live pipeline until packing.py replaces the flat cost
    model with real variable tiling costs. This test locks in that dormancy
    so it's an intentional, documented fact — not a silent regression if
    someone changes propose_regions() later without re-checking this."""
    regions = propose_regions(sample_image, grid=4)  # all same cost (flat tokens_per_cell)
    for budget in (200, 300, 400):
        greedy_value = sum(r.value for r in greedy_select(regions, budget))
        dp_value = sum(r.value for r in dp_select(regions, budget))
        assert dp_value == greedy_value

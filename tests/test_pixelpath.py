from pixelpath.regions import Region, greedy_select, propose_regions
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

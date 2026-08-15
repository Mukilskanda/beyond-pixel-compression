from profiler.budget import MAX_BUDGET, MIN_BUDGET, allocate_budget
from profiler.features import profile_image


def test_budget_within_bounds(sample_image, photo_like_image):
    for img in (sample_image, photo_like_image):
        feats = profile_image(img)
        budget = allocate_budget(feats)
        assert MIN_BUDGET <= budget <= MAX_BUDGET


def test_budget_is_deterministic(sample_image):
    feats = profile_image(sample_image)
    assert allocate_budget(feats) == allocate_budget(feats)


def test_blank_image_gets_reduced_budget(tmp_path):
    """A pure white image should score near the bottom of the budget range —
    the sparsity discount should meaningfully cut its allocation."""
    from PIL import Image

    blank_path = tmp_path / "blank.png"
    Image.new("RGB", (400, 400), "white").save(blank_path)

    feats = profile_image(blank_path)
    budget = allocate_budget(feats)
    midpoint = (MIN_BUDGET + MAX_BUDGET) / 2
    assert budget < midpoint

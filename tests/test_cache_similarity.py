from pixelpath.cache_similarity import VisualSemanticCache, image_similarity


def test_identical_image_has_similarity_near_one(sample_image):
    sim = image_similarity(sample_image, sample_image)
    assert sim > 0.999


def test_different_looking_images_score_lower_than_identical(sample_image, photo_like_image):
    self_sim = image_similarity(sample_image, sample_image)
    cross_sim = image_similarity(sample_image, photo_like_image)
    assert cross_sim < self_sim


def test_cache_empty_returns_none(sample_image):
    cache = VisualSemanticCache()
    entry, sim = cache.best_match(sample_image)
    assert entry is None
    assert sim == 0.0


def test_cache_hit_above_threshold(sample_image):
    cache = VisualSemanticCache()
    cache.add(sample_image, "total?", "$500.00")
    entry, sim = cache.lookup(sample_image, threshold=0.9)
    assert entry is not None
    assert entry.answer == "$500.00"
    assert sim > 0.9


def test_cache_miss_below_threshold(sample_image, photo_like_image):
    cache = VisualSemanticCache()
    cache.add(photo_like_image, "what is this?", "random noise")
    entry, sim = cache.lookup(sample_image, threshold=0.999)
    assert entry is None


def test_best_match_picks_most_similar_of_multiple_entries(sample_image, photo_like_image):
    cache = VisualSemanticCache()
    cache.add(photo_like_image, "q1", "a1")
    cache.add(sample_image, "q2", "a2")
    entry, sim = cache.best_match(sample_image)
    assert entry.answer == "a2"

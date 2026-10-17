from pixelpath.cache_similarity import (
    QuestionSimilarity,
    QueryAwareCache,
    VisualSemanticCache,
    image_similarity,
)


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


# --- Query-aware cache (§11.4) ---

CORPUS = [
    "What is the name of the company?",
    "Where is the company located?",
    "What is the total revenue?",
    "What is the average revenue?",
]


def _fitted_question_sim():
    qs = QuestionSimilarity()
    qs.fit(CORPUS)
    return qs


def test_identical_questions_score_similarity_near_one():
    qs = _fitted_question_sim()
    assert qs.similarity(CORPUS[0], CORPUS[0]) > 0.999


def test_unrelated_questions_score_lower_than_identical():
    qs = _fitted_question_sim()
    self_sim = qs.similarity(CORPUS[0], CORPUS[0])
    cross_sim = qs.similarity(CORPUS[0], CORPUS[2])
    assert cross_sim < self_sim


def test_question_similarity_requires_fit_before_use():
    import pytest

    qs = QuestionSimilarity()
    with pytest.raises(RuntimeError, match="must be fit"):
        qs.similarity("a", "b")


def test_query_aware_cache_hit_requires_both_image_and_question_match(sample_image):
    qs = _fitted_question_sim()
    cache = QueryAwareCache(question_similarity=qs)
    cache.add(sample_image, CORPUS[0], "cached answer")

    # same image, same question -> both axes should clear a high threshold
    entry, img_sim, q_sim = cache.lookup(sample_image, CORPUS[0], threshold=0.9)
    assert entry is not None
    assert img_sim > 0.9 and q_sim > 0.9


def test_query_aware_cache_miss_when_question_differs_even_with_identical_image(sample_image):
    """The case §11.2's image-only cache got wrong: same image, different
    question - a hit here would silently return the wrong answer."""
    qs = _fitted_question_sim()
    cache = QueryAwareCache(question_similarity=qs)
    cache.add(sample_image, CORPUS[0], "cached answer for company name")

    entry, img_sim, q_sim = cache.lookup(sample_image, CORPUS[2], threshold=0.9)
    assert entry is None  # image matches perfectly, but question does not
    assert img_sim > 0.9
    assert q_sim < 0.9


def test_query_aware_cache_empty_returns_none(sample_image):
    qs = _fitted_question_sim()
    cache = QueryAwareCache(question_similarity=qs)
    entry, img_sim, q_sim = cache.best_match(sample_image, CORPUS[0])
    assert entry is None
    assert img_sim == 0.0 and q_sim == 0.0

from harness.cache import ResponseCache


def test_cache_miss_then_hit(tmp_path):
    cache = ResponseCache(tmp_path / "cache.sqlite")
    image_bytes = b"fake-image-data"
    settings = {"prompt": "what is this?"}

    assert cache.get(image_bytes, "model-a", settings) is None

    cache.put(image_bytes, "model-a", settings, {"text": "an answer", "tokens": 42})
    result = cache.get(image_bytes, "model-a", settings)
    assert result == {"text": "an answer", "tokens": 42}


def test_cache_keyed_on_model_and_settings(tmp_path):
    cache = ResponseCache(tmp_path / "cache.sqlite")
    image_bytes = b"fake-image-data"

    cache.put(image_bytes, "model-a", {"prompt": "x"}, {"text": "A"})
    cache.put(image_bytes, "model-b", {"prompt": "x"}, {"text": "B"})
    cache.put(image_bytes, "model-a", {"prompt": "y"}, {"text": "C"})

    assert cache.get(image_bytes, "model-a", {"prompt": "x"}) == {"text": "A"}
    assert cache.get(image_bytes, "model-b", {"prompt": "x"}) == {"text": "B"}
    assert cache.get(image_bytes, "model-a", {"prompt": "y"}) == {"text": "C"}


def test_cache_different_image_bytes_miss(tmp_path):
    cache = ResponseCache(tmp_path / "cache.sqlite")
    cache.put(b"image-1", "model-a", {}, {"text": "A"})
    assert cache.get(b"image-2", "model-a", {}) is None

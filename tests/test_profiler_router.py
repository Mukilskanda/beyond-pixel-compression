from profiler.features import ProfileFeatures, profile_image
from router.rules import Route, route


def test_profile_image_returns_all_fields(sample_image):
    feats = profile_image(sample_image)
    assert isinstance(feats, ProfileFeatures)
    for value in (feats.text_density, feats.whitespace_ratio, feats.layout_regularity):
        assert 0.0 <= value <= 1.0


def test_profile_document_vs_photo(sample_image, photo_like_image):
    """A synthetic document (white bg, text) should show far more whitespace
    than a synthetic random-noise photo — the whole premise of the router."""
    doc_feats = profile_image(sample_image)
    photo_feats = profile_image(photo_like_image)
    assert doc_feats.whitespace_ratio > photo_feats.whitespace_ratio


def test_whitespace_ratio_not_fooled_by_realistic_scan_noise(tmp_path):
    """Regression test for a real bug: the original whitespace_ratio used a
    fixed cell.std() < 8.0 cutoff, which called every noisy-but-blank image
    'zero whitespace' — including a page that's obviously mostly blank to a
    human. Verified against real DocVQA scans where blank-margin cell std
    ranged 16-50 depending on scan quality. A page with visible Gaussian
    scan noise added to blank white must still register meaningful whitespace."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    # near-white background with realistic scan noise (std ~20, well above
    # the old fixed threshold of 8, but still visibly "blank paper" to a human)
    base = np.full((600, 800), 245, dtype=np.float64)
    noise = rng.normal(0, 20, base.shape)
    noisy_blank = np.clip(base + noise, 0, 255).astype("uint8")
    img = Image.fromarray(noisy_blank, mode="L").convert("RGB")
    path = tmp_path / "noisy_blank.png"
    img.save(path)

    feats = profile_image(str(path))
    assert feats.whitespace_ratio > 0.5, (
        f"whitespace_ratio={feats.whitespace_ratio} — a near-white noisy page "
        "should register as mostly blank, not near-zero"
    )


def test_route_spatial_query_never_goes_pure_text(sample_image):
    feats = profile_image(sample_image)
    r = route(feats, "how many blue rectangles are there?")
    assert r != Route.TEXT


def test_route_returns_valid_enum(sample_image, photo_like_image):
    for img in (sample_image, photo_like_image):
        feats = profile_image(img)
        r = route(feats, "what does this say?")
        assert r in (Route.TEXT, Route.PIXEL, Route.HYBRID)


def test_route_fires_text_on_real_sparse_docvqa_sample():
    """Regression test for the fixed TEXT-route gap (router/rules.py docstring):
    before the fix, whitespace_ratio > 0.85 never fired on any real scan.
    These feature values are real, measured (not synthetic) — question_id
    16424 from lmms-lab/DocVQA validation, profiled 2026-08-18
    (text_density=0.0087, whitespace_ratio=0.7734) — one of the sparser
    documents in a 40-sample real profiling run. It must now route TEXT."""
    feats = ProfileFeatures(
        text_density=0.0087,
        whitespace_ratio=0.7734,
        layout_regularity=0.0,
        color_entropy=0.0,
        spatial_variance=0.0,
    )
    assert route(feats, "what is the date on this document?") == Route.TEXT


def test_route_still_hybrid_on_real_dense_docvqa_sample():
    """Companion to the sparse-sample test above: a denser real sample
    (question_id 24580, text_density=0.2149, whitespace_ratio=0.1797) must
    stay HYBRID/PIXEL, not flip to TEXT — the fix narrows the gap, it
    doesn't make everything route TEXT."""
    feats = ProfileFeatures(
        text_density=0.2149,
        whitespace_ratio=0.1797,
        layout_regularity=0.0,
        color_entropy=0.0,
        spatial_variance=0.0,
    )
    assert route(feats, "what is the date on this document?") != Route.TEXT


def test_query_conditioning_ablation_ignores_spatial_intent(sample_image):
    """With query_conditioning=False, a spatial-intent query ('how many...')
    must not trigger the spatial override — routing depends on image features
    alone. This is the ablation switch from experiments/ablations.yaml."""
    feats = profile_image(sample_image)

    with_conditioning = route(feats, "how many blue rectangles are there?", query_conditioning=True)
    without_conditioning = route(feats, "how many blue rectangles are there?", query_conditioning=False)

    # With conditioning on, the spatial keyword forces PIXEL/HYBRID.
    assert with_conditioning != Route.TEXT
    # With conditioning off, the same query must resolve identically to a
    # query-free classification — i.e. match what the non-spatial rules alone decide.
    assert without_conditioning == route(feats, "irrelevant text with no keywords", query_conditioning=False)

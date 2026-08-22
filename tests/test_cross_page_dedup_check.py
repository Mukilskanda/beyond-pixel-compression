from experiments.cross_page_dedup_check import BODY_PROFILES, BOILERPLATE_PROFILES, run, sweep


def test_synthetic_fixture_extracts_expected_page_count():
    result = run(n_pages=4)
    assert result["n_pages"] == 4


def test_within_page_dedup_finds_nothing_on_this_fixture():
    """Boilerplate appears once per page (header only), so within-page
    dedup — which only sees one page at a time — has nothing to remove.
    This is the exact gap cross-page dedup exists to close, pinned so a
    future change to the fixture that accidentally duplicates a line
    WITHIN a page doesn't silently start passing for the wrong reason."""
    result = run(n_pages=4)
    assert result["within_page_dedup_tokens"] == result["no_dedup_tokens"]


def test_cross_page_dedup_beats_within_page_dedup_and_grows_with_n():
    small = run(n_pages=2)
    large = run(n_pages=8)

    assert small["cross_page_vs_within_page"] > 1.0
    assert large["cross_page_vs_within_page"] > small["cross_page_vs_within_page"]


def test_cross_page_dedup_never_removes_the_varying_body_content():
    """Mechanism correctness check on real PyMuPDF-extracted text (not a
    mock): the per-page body content (which varies, unlike the fixture's
    header) must never hit zero tokens even as more of the page gets
    deduped away."""
    result = run(n_pages=8)
    assert result["cross_page_dedup_tokens"] > 0


def test_heavier_boilerplate_profile_yields_more_savings_than_lighter():
    """The sweep's whole point: savings should scale with how much
    boilerplate repeats, not be a single fixed ratio — this is a real,
    dataset-independent property of the dedup mechanism, not something
    that needs a real corpus to demonstrate."""
    light = run(n_pages=8, profile="light", body="short_body")
    heavy = run(n_pages=8, profile="heavy", body="short_body")
    assert heavy["cross_page_vs_none"] > light["cross_page_vs_none"]


def test_longer_body_yields_less_relative_savings_than_shorter_body():
    """More genuine (non-duplicated) content per page dilutes the fixed
    boilerplate's share of the total — savings ratio should shrink as body
    length grows, for the same boilerplate profile."""
    short = run(n_pages=8, profile="medium", body="short_body")
    long_ = run(n_pages=8, profile="medium", body="long_body")
    assert short["cross_page_vs_none"] > long_["cross_page_vs_none"]


def test_sweep_covers_full_grid():
    rows = sweep()
    assert len(rows) == len(BOILERPLATE_PROFILES) * len(BODY_PROFILES) * 4  # 4 page counts
    profiles_seen = {r["profile"] for r in rows}
    bodies_seen = {r["body"] for r in rows}
    assert profiles_seen == set(BOILERPLATE_PROFILES)
    assert bodies_seen == set(BODY_PROFILES)

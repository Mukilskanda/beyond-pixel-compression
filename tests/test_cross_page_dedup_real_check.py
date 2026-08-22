import shutil
from pathlib import Path

import pytest

from experiments.cross_page_dedup_real_check import run

requires_tesseract = pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="tesseract binary not installed (see README setup)"
)
requires_real_cache = pytest.mark.skipif(
    not any(Path("data/mpdocvqa_cache").glob("*_p*.png")) if Path("data/mpdocvqa_cache").exists() else True,
    reason="no cached real MP-DocVQA pages on disk (network-fetched, gitignored — "
    "run experiments/cross_page_dedup_real_check.py once to populate)",
)

pytestmark = [requires_tesseract, requires_real_cache]


@pytest.fixture(scope="module")
def real_result():
    """Real Tesseract OCR on real page images is slow (~7s/page) — run
    once per test module and share across assertions, not once per test."""
    return run()


def test_real_check_runs_on_cached_pages_without_network(real_result):
    assert "error" not in real_result
    assert real_result["n_pages"] > 0


def test_real_check_reports_no_negative_or_impossible_savings(real_result):
    """Cross-page dedup can only remove tokens, never add them — this
    should hold on real data exactly as it does in the synthetic sweep's
    tests, regardless of whether real savings turn out to be zero."""
    assert real_result["cross_page_dedup_tokens"] <= real_result["no_dedup_tokens"]
    assert real_result["cross_page_vs_none"] >= 1.0


def test_real_check_line_removal_count_matches_token_reduction_direction(real_result):
    if real_result["lines_removed_by_cross_page_dedup"] == 0:
        assert real_result["cross_page_dedup_tokens"] == real_result["no_dedup_tokens"]
    else:
        assert real_result["cross_page_dedup_tokens"] < real_result["no_dedup_tokens"]

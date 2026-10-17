import sqlite3

import pytest

from experiments.evidence_strength_report import _bootstrap_ci, batched_tiling_reduction_ci


def test_bootstrap_ci_on_constant_ratios_has_zero_width():
    result = _bootstrap_ci([2.0, 2.0, 2.0, 2.0])
    assert result["mean"] == 2.0
    assert result["ci_low"] == result["ci_high"] == 2.0


def test_bootstrap_ci_handles_single_sample_without_crashing():
    result = _bootstrap_ci([3.5])
    assert result["n"] == 1
    assert result["ci_low"] is None


def test_bootstrap_ci_handles_empty_list_without_crashing():
    result = _bootstrap_ci([])
    assert result["n"] == 0
    assert result["mean"] is None


@pytest.fixture
def mixed_provider_db(tmp_path, monkeypatch):
    """Regression fixture for the real bug found this session: an
    unreduced-token baseline must never be pooled across providers with
    very different flat per-image costs (Groq ~1800 tokens, Gemini ~270)
    — doing so silently inflated Gemini's batched-tiling reduction ratio
    to 24.86x against a real reported 5.35x. This pins the fix by
    constructing a DB where naive dataset-only matching would reproduce
    the bug, and asserting the provider-aware version doesn't."""
    db_path = tmp_path / "results.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE runs (dataset TEXT, method TEXT, model TEXT, input_tokens INTEGER, truncated INTEGER)")
    # Groq: expensive flat image cost, mixed into the same 'docvqa' dataset tag.
    for tok in (1800, 1810, 1820, 1790):
        conn.execute("INSERT INTO runs VALUES ('docvqa', 'unreduced', 'qwen/qwen3.6-27b', ?, 0)", (tok,))
    # Gemini: cheap flat image cost, same dataset tag, different model.
    for tok in (270, 265, 280):
        conn.execute("INSERT INTO runs VALUES ('docvqa', 'unreduced', 'gemini-2.5-flash', ?, 0)", (tok,))
    # A real Gemini batched trial — tiny token count, as batching produces.
    conn.execute("INSERT INTO runs VALUES ('docvqa_batched_tiled_gemini', 'batched_n4', 'gemini-2.5-flash', 217, 0)")
    conn.commit()
    conn.close()

    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    db_path.rename(tmp_path / "results" / "results.sqlite")
    return tmp_path


def test_batched_tiling_ci_does_not_mix_providers_in_baseline(mixed_provider_db):
    result = batched_tiling_reduction_ci()
    gemini_result = result["docvqa/gemini"]
    # Real Gemini-only baseline (~272 avg) * 4 / 217 ~= 5.0, NOT ~25 (what
    # pooling in the ~1800-token Groq rows would produce).
    assert gemini_result["mean"] < 10.0
    assert gemini_result["mean"] > 3.0

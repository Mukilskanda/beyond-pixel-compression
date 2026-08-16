"""Regression tests for pixelpath/mode_select.py's mode-selection optimizer.

Uses the two real, provider-verified Gemini media_resolution token costs
(74 for "low", 258 for "unset" - see paper/draft.md §3.5/§7/§9.3 and
tests/fixtures/gemini_token_cost_2026-08-16.json). Fidelity numbers fed into
the optimizer in these tests are synthetic inputs to exercise the selection
LOGIC - real per-mode fidelity measurement (does low-resolution mode hurt
answer quality) is future work, not yet run, and is not claimed here.
"""
import pytest

from pixelpath.mode_select import (
    GEMINI_MEDIA_RESOLUTION_TOKENS,
    GROQ_DETAIL_TOKENS,
    ModeOptimizer,
)


def test_known_gemini_tiers_match_real_measurements():
    """Locks in the two real numbers so a future edit can't silently drift
    from what was actually measured."""
    assert GEMINI_MEDIA_RESOLUTION_TOKENS["low"] == 74
    assert GEMINI_MEDIA_RESOLUTION_TOKENS["unset"] == 258


def test_groq_has_only_one_real_mode():
    """Groq's detail parameter has zero measured effect (§9.3) - there should
    be no fabricated second tier."""
    assert len(GROQ_DETAIL_TOKENS) == 1


def test_picks_cheapest_mode_that_clears_floor():
    opt = ModeOptimizer(GEMINI_MEDIA_RESOLUTION_TOKENS)
    selection = opt.select(fidelity_by_mode={"low": 0.9, "unset": 0.95}, fidelity_floor=0.8)
    assert selection.chosen.mode == "low"
    assert selection.chosen.token_cost == 74


def test_rejects_cheap_mode_below_floor():
    opt = ModeOptimizer(GEMINI_MEDIA_RESOLUTION_TOKENS)
    selection = opt.select(fidelity_by_mode={"low": 0.4, "unset": 0.95}, fidelity_floor=0.8)
    assert selection.chosen.mode == "unset"
    assert selection.chosen.token_cost == 258


def test_falls_back_to_highest_fidelity_when_nothing_clears_floor():
    opt = ModeOptimizer(GEMINI_MEDIA_RESOLUTION_TOKENS)
    selection = opt.select(fidelity_by_mode={"low": 0.4, "unset": 0.6}, fidelity_floor=0.9)
    assert selection.chosen.mode == "unset"
    assert "no mode cleared" in selection.reason


def test_unknown_mode_raises_instead_of_guessing_cost():
    opt = ModeOptimizer(GEMINI_MEDIA_RESOLUTION_TOKENS)
    with pytest.raises(ValueError, match="no known token cost"):
        opt.select(fidelity_by_mode={"medium": 0.9}, fidelity_floor=0.8)


def test_single_mode_provider_always_returns_that_mode():
    opt = ModeOptimizer(GROQ_DETAIL_TOKENS)
    selection = opt.select(fidelity_by_mode={"unset": 0.5}, fidelity_floor=0.99)
    assert selection.chosen.mode == "unset"


def test_empty_token_costs_rejected():
    with pytest.raises(ValueError):
        ModeOptimizer({})

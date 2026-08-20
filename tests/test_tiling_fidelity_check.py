from experiments.tiling_fidelity_check import scale_max_tokens


def test_scale_max_tokens_increases_with_n():
    """Regression test for a real bug: a fixed max_tokens=500 truncated
    6/8 real Groq calls at N=8 (grid-navigation reasoning grows with N),
    and 3/8 still truncated even at max_tokens=2500. The budget must
    scale with N, not stay constant."""
    assert scale_max_tokens(500, 2) < scale_max_tokens(500, 8)
    assert scale_max_tokens(500, 8) < scale_max_tokens(500, 12)


def test_scale_max_tokens_matches_base_at_n_zero():
    assert scale_max_tokens(500, 0) == 500


def test_scale_max_tokens_real_n8_value_exceeds_original_bug_report():
    """N=8 with the original 500 base truncated most calls; the scaled
    value must clear the 2500 ceiling that STILL wasn't always enough,
    documented honestly rather than assumed fixed by this scaling alone."""
    assert scale_max_tokens(500, 8) >= 2500

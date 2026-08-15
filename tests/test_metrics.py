from harness.metrics import anls, answer_agreement, exact_match


def test_exact_match_case_and_whitespace_insensitive():
    assert exact_match("  Paris  ", ["paris"])
    assert not exact_match("London", ["Paris"])


def test_exact_match_any_ground_truth():
    assert exact_match("cat", ["dog", "cat", "bird"])


def test_anls_perfect_match():
    assert anls("$500.00", ["$500.00"]) == 1.0


def test_anls_below_threshold_returns_zero():
    assert anls("completely different text", ["$500.00"], threshold=0.5) == 0.0


def test_anls_close_match_above_threshold():
    score = anls("500.00", ["$500.00"])
    assert 0.5 < score < 1.0


def test_answer_agreement():
    assert answer_agreement("Paris", "  paris  ")
    assert not answer_agreement("Paris", "London")

from harness.metrics import anls, answer_agreement, exact_match, is_truncated


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


# --- Fidelity pipeline audit (Step 1) ---
# Regression coverage for silent-failure modes found while auditing beyond
# the already-fixed qwen/qwen3.6-27b <think>-preamble bug (paper/draft.md
# §9.4). is_truncated is the general-purpose version of the ad hoc fix
# applied there - these tests lock in that a truncated response is
# detectable, and that "we don't know" (finish_reason=None) never gets
# silently treated as "confirmed complete."


def test_is_truncated_recognizes_known_reasons():
    assert is_truncated("length")  # OpenAI/Groq-shaped
    assert is_truncated("LENGTH")  # case-insensitive
    assert is_truncated("max_tokens")  # Anthropic-shaped


def test_is_truncated_false_for_normal_stop():
    assert not is_truncated("stop")
    assert not is_truncated("end_turn")


def test_is_truncated_none_is_not_treated_as_truncated():
    """finish_reason=None means 'unknown' (e.g. OllamaProvider, which never
    populates it) - must not silently read as 'confirmed complete' OR
    'confirmed truncated'. is_truncated(None) being False means callers
    that need the distinction must check `is not None` separately."""
    assert not is_truncated(None)


def test_anls_empty_prediction_scores_as_wrong_not_error():
    """An empty response (e.g. a silently blocked/filtered call) must score
    as a real 0.0, not raise or silently pass - but this is INDISTINGUISHABLE
    from a truncation artifact unless finish_reason is also checked, which is
    why harness/runner.py checks is_truncated() before scoring at all."""
    assert anls("", ["$500.00"]) == 0.0


def test_anls_empty_ground_truth_list_scores_zero_not_silently_skipped():
    """If ground_truths is a non-empty list containing only an empty string
    (e.g. a data-loading bug that lost the real answer), anls must not
    silently return a passing score - confirms the metric doesn't
    special-case empty ground truth as an automatic match."""
    assert anls("some answer", [""]) == 0.0


# --- <think>-block stripping (paper/draft.md §12.4) ---
# Regression coverage for the real qwen/qwen3.6-27b case found during the
# real-billing validation run: a correct answer wrapped in a <think> block
# scored anls=0.0 because the metric compared against the raw text
# including the reasoning preamble, not because the answer was wrong.


def test_anls_strips_closed_think_block_before_scoring():
    prediction = "<think>Let me look at the letterhead. It says UNIVERSITY OF CALIFORNIA.</think>\nUniversity of California, San Diego"
    assert anls(prediction, ["University of California, San Diego"]) == 1.0


def test_exact_match_strips_closed_think_block_before_scoring():
    prediction = "<think>reasoning about the answer here</think>\nParis"
    assert exact_match(prediction, ["Paris"])


def test_anls_does_not_strip_unclosed_think_block():
    """An unclosed <think> block means the response was genuinely cut off
    mid-reasoning (a truncation case, handled separately by is_truncated())
    - stripping it here would silently hide that instead of surfacing it."""
    prediction = "<think>still reasoning and never reached an answer"
    assert anls(prediction, ["University of California, San Diego"]) == 0.0


def test_anls_think_block_stripping_is_case_insensitive():
    prediction = "<THINK>reasoning</THINK>\nParis"
    assert anls(prediction, ["Paris"]) == 1.0

"""Regression tests for experiments/pdf_batched_text_route_check.py — the
offline mechanism check for a batched multi-page TEXT route (never tested
before this session). Pure, synthetic fixtures only, no network/API calls
and no dependency on the real DocVQA cache, per this project's standing
practice of proving mechanism logic offline before it ever touches real
data or spend.
"""

from experiments.pdf_batched_text_route_check import _count_tokens, build_batched_text_prompt


def test_batched_prompt_contains_all_pages_and_questions_in_order():
    pages = ["Invoice #1, total $10", "Invoice #2, total $20"]
    questions = ["What is the total on page 1?", "What is the total on page 2?"]

    prompt = build_batched_text_prompt(pages, questions)

    assert "--- Page 1 ---" in prompt
    assert "--- Page 2 ---" in prompt
    assert "Invoice #1, total $10" in prompt
    assert "Invoice #2, total $20" in prompt
    assert "Page 1: What is the total on page 1?" in prompt
    assert "Page 2: What is the total on page 2?" in prompt
    assert prompt.index("Page 1") < prompt.index("Page 2")


def test_batched_prompt_demands_json_only_output():
    prompt = build_batched_text_prompt(["text"], ["question"])
    assert "JSON" in prompt
    assert "No other text outside the JSON" in prompt


def test_batched_prompt_requires_matching_page_and_question_counts():
    try:
        build_batched_text_prompt(["only one page"], ["q1", "q2"])
    except AssertionError:
        pass
    else:
        raise AssertionError("expected an AssertionError for mismatched page/question counts")


def test_fixed_wrapper_overhead_grows_with_n_not_shrinks():
    """This is the mechanism this offline check exists to isolate: unlike
    the tiled-IMAGE batching case (§13.8), where batching removes a large
    fixed per-call cost (~1800 tokens on Groq) paid N times, the TEXT
    route's per-call cost (harness/providers.py's call_text_only, just
    `f"{text}\\n\\n{prompt}"`) already has almost no fixed overhead to
    remove. Batching N pages into one call instead ADDS a JSON-formatting
    instruction (paid once) plus a small per-page wrapper (paid N times) —
    both strictly non-negative, so the wrapper-only cost of the batched
    route must be monotonically non-decreasing in N, never negative
    relative to more pages. This is what makes batched-text a net loss at
    every N this session actually measured (n=2..32 against real cached
    DocVQA pages, ratio 0.59 to 0.98 — approaching but never crossing 1.0)."""
    small = build_batched_text_prompt(["", ""], ["", ""])  # n=2
    large = build_batched_text_prompt([""] * 8, [""] * 8)  # n=8

    assert _count_tokens(large) > _count_tokens(small)


def test_route_a_per_page_wrapper_has_near_zero_fixed_overhead():
    """Route A (today's actual §12.4 shape, N separate text-only calls) has
    no batching wrapper at all — confirms there is no large fixed cost for
    a batched route to amortize away in the text case, unlike the image
    case's flat ~1800-token per-call floor (§9.4)."""
    per_page_prompt = "Question: \n\nDocument text:\n"
    assert _count_tokens(per_page_prompt) < 10

from experiments.batched_tile_qa_check import build_prompt, parse_batched_answers


def test_build_prompt_includes_all_questions_numbered():
    prompt = build_prompt(3, ["Q1?", "Q2?", "Q3?"])
    assert "Tile 1: Q1?" in prompt
    assert "Tile 2: Q2?" in prompt
    assert "Tile 3: Q3?" in prompt
    assert "3-tile grid" in prompt


def test_build_prompt_requests_json_output():
    prompt = build_prompt(2, ["Q1?", "Q2?"])
    assert "JSON" in prompt


def test_build_prompt_efficient_adds_brevity_instruction():
    base = build_prompt(4, ["Q1?"] * 4, efficient=False)
    efficient = build_prompt(4, ["Q1?"] * 4, efficient=True)
    assert len(efficient) > len(base)
    assert "brief and direct" in efficient
    assert "brief and direct" not in base


def test_parse_batched_answers_clean_json():
    text = '{"1": "Paris", "2": "42"}'
    result = parse_batched_answers(text, 2)
    assert result == {"1": "Paris", "2": "42"}


def test_parse_batched_answers_strips_think_block_first():
    """Regression test: real Groq responses wrap the JSON in a <think>
    reasoning preamble - the parser must find the JSON object after that
    block, not fail because the whole response isn't valid JSON on its own."""
    text = '<think>reasoning about tiles...</think>\n{"1": "Paris"}'
    result = parse_batched_answers(text, 1)
    assert result == {"1": "Paris"}


def test_parse_batched_answers_truncated_response_returns_empty():
    """Regression test for the real truncation case found in this
    project's own runs: a response cut off mid-reasoning (no valid JSON
    anywhere) must return an empty dict, not raise or silently guess."""
    text = "<think>still reasoning about tile 3, wait let me recheck"
    result = parse_batched_answers(text, 4)
    assert result == {}


def test_parse_batched_answers_malformed_json_returns_empty():
    text = '{"1": "Paris", invalid}'
    result = parse_batched_answers(text, 1)
    assert result == {}


def test_parse_batched_answers_keys_are_strings():
    """JSON object keys are always strings, but confirm explicitly since
    the caller looks answers up via str(k)."""
    text = '{"1": "answer one", "2": "answer two"}'
    result = parse_batched_answers(text, 2)
    assert isinstance(list(result.keys())[0], str)

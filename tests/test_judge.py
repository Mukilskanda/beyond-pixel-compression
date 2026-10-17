from harness.providers import CallResult
from harness.judge import judge_correctness


class StubJudgeProvider:
    """Zero-cost stand-in for a real judge provider - returns a fixed
    verdict text regardless of the prompt, so judge_correctness()'s
    parsing/plumbing can be tested without spending real API quota."""

    def __init__(self, verdict_text: str):
        self.verdict_text = verdict_text
        self.last_prompt: str | None = None

    def call_text_only(self, text, prompt, **settings):
        self.last_prompt = prompt
        return CallResult(
            text=self.verdict_text, input_tokens=10, output_tokens=1, latency_s=0.001, model="stub-judge", raw={}
        )


def test_judge_correctness_parses_yes():
    provider = StubJudgeProvider("YES")
    assert judge_correctness("What is the total?", "$500.00", ["$500.00"], provider) is True


def test_judge_correctness_parses_no():
    provider = StubJudgeProvider("NO")
    assert judge_correctness("What is the total?", "$100.00", ["$500.00"], provider) is False


def test_judge_correctness_returns_none_for_unparseable_verdict():
    provider = StubJudgeProvider("I'm not sure, could go either way")
    assert judge_correctness("What is the total?", "$500.00", ["$500.00"], provider) is None


def test_judge_correctness_strips_think_block_before_prompting():
    provider = StubJudgeProvider("YES")
    prediction = "<think>let me reason about this...</think>\nThe total is $500.00"
    judge_correctness("What is the total?", prediction, ["$500.00"], provider)
    assert "<think>" not in provider.last_prompt
    assert "let me reason" not in provider.last_prompt
    assert "$500.00" in provider.last_prompt


def test_judge_correctness_includes_all_ground_truths_in_prompt():
    provider = StubJudgeProvider("YES")
    judge_correctness("q", "answer", ["gt1", "gt2"], provider)
    assert "gt1" in provider.last_prompt
    assert "gt2" in provider.last_prompt


def test_judge_correctness_verdict_case_insensitive():
    provider = StubJudgeProvider("yes")
    assert judge_correctness("q", "a", ["a"], provider) is True


class RecordingMaxTokensProvider:
    """Regression test for a real bug: an earlier version called
    judge_provider.call_text_only(..., max_tokens=10), which silently
    failed against reasoning judge models (openai/gpt-oss-120b included)
    that burn hidden reasoning tokens before the visible YES/NO - a 60/60
    unparseable batch is what surfaced it. Confirms the fix (max_tokens=100)
    is what's actually sent, not just documented."""

    def __init__(self):
        self.last_max_tokens = None

    def call_text_only(self, text, prompt, **settings):
        self.last_max_tokens = settings.get("max_tokens")
        return CallResult(text="YES", input_tokens=10, output_tokens=1, latency_s=0.001, model="stub", raw={})


def test_judge_correctness_requests_enough_output_tokens_for_reasoning_models():
    provider = RecordingMaxTokensProvider()
    judge_correctness("q", "a", ["a"], provider)
    assert provider.last_max_tokens >= 100


class RecordingSettingsProvider:
    """Regression test for a second, related bug: max_tokens=100 alone
    still left ~18/60 real judge calls unparseable (openai/gpt-oss-120b's
    hidden reasoning budget is variable-length, not fixed) - the real fix
    was reasoning_effort="low", which suppresses most of the hidden
    reasoning outright. Confirms it's actually sent, not just documented."""

    def __init__(self):
        self.last_settings = None

    def call_text_only(self, text, prompt, **settings):
        self.last_settings = settings
        return CallResult(text="YES", input_tokens=10, output_tokens=1, latency_s=0.001, model="stub", raw={})


def test_judge_correctness_requests_low_reasoning_effort():
    provider = RecordingSettingsProvider()
    judge_correctness("q", "a", ["a"], provider)
    assert provider.last_settings.get("reasoning_effort") == "low"

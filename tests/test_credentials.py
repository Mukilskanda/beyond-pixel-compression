"""Regression coverage for harness/providers.credentials_status() - the
real-check function every sweep script must call before spending quota,
per the standing rule that a sweep confirms credentials load before
proceeding to spend-dependent work (draft.md §4.3/§8: every prior sweep
silently never ran because keys were session-scoped shell exports, not
persisted via .env)."""

from harness.providers import credentials_status


def test_returns_false_for_all_when_unset(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    status = credentials_status()
    assert status == {
        "GEMINI_API_KEY": False,
        "GROQ_API_KEY": False,
        "ANTHROPIC_API_KEY": False,
        "OPENROUTER_API_KEY": False,
    }


def test_returns_true_only_for_set_keys(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-value-for-test-only")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    status = credentials_status()
    assert status["GEMINI_API_KEY"] is True
    assert status["GROQ_API_KEY"] is False


def test_empty_string_env_var_counts_as_unset(monkeypatch):
    """A key set to an empty string (e.g. a blank .env template line) must
    not be reported as a real credential - bool("") is already False, but
    lock this in explicitly since it's exactly the .env.example shape."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    status = credentials_status()
    assert status["GEMINI_API_KEY"] is False

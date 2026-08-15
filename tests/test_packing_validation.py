"""Regression check against real Gemini 2.5 Flash token counts captured once
(tests/fixtures/gemini_token_cost_2026-08-16.json) so future changes to the
cost-model discussion don't require spending API calls again to re-verify
what was already found. This does NOT call any real API — it locks in the
captured numbers and the headline finding they support.
"""
import json
from pathlib import Path

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "gemini_token_cost_2026-08-16.json"


def load_fixture():
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_fixture_loads():
    data = load_fixture()
    assert len(data["samples"]) == 10


def test_real_gemini_token_cost_is_flat_across_dimensions():
    """The headline finding this fixture exists to preserve: Gemini 2.5
    Flash's real image token cost did not vary with image dimensions in the
    tested range (404x538 up to 1653x2339, and independently 100x100 to
    3000x4000). If a future re-run of experiments/packing_validation.py
    shows real costs that DO vary with size, that's a meaningful change
    worth re-investigating — this test exists to make that change visible."""
    data = load_fixture()
    real_tokens = {s["real_image_tokens"] for s in data["samples"]}
    assert real_tokens == {258}, (
        f"Expected flat 258 tokens across all samples (the captured finding), got {real_tokens}. "
        "If this fails after a genuine re-run with new real numbers, the finding may have "
        "changed (e.g. a Gemini model update) - investigate before assuming the fixture is stale."
    )


def test_independent_sanity_check_confirms_flat_cost():
    data = load_fixture()
    checks = data["independent_sanity_check"]["results"]
    image_tokens = {c["image_tokens"] for c in checks}
    assert image_tokens == {258}


def test_local_estimate_diverges_from_real_cost():
    """packing.py's current step-function-per-tile model does NOT match
    Gemini's real flat-cost behavior - this is the reason packing.py is
    flagged as not-yet-trustworthy for Gemini specifically. Locks in that
    the divergence is real and large, not a rounding-level discrepancy."""
    data = load_fixture()
    for sample in data["samples"]:
        assert abs(sample["pct_diff"]) > 20, (
            f"{sample['filename']}: expected a large, systematic divergence "
            "between local_estimate and real cost (the documented finding), "
            f"got only {sample['pct_diff']}%"
        )

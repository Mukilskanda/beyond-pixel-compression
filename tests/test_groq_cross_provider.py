"""Regression check against real Groq (qwen/qwen3.6-27b) token counts,
captured once (tests/fixtures/groq_token_cost_2026-08-16.json) so the
cross-provider finding doesn't require spending API calls again to
re-verify. Does NOT call any real API.
"""

import json
from pathlib import Path

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "groq_token_cost_2026-08-16.json"


def load_fixture():
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_fixture_loads():
    data = load_fixture()
    assert data["_meta"]["provider"] == "groq / qwen/qwen3.6-27b"


def test_metering_is_real_not_null():
    """The prerequisite check from the task: a provider that can't report
    its own cost can't answer the generalization question."""
    data = load_fixture()
    check = data["metering_sanity_check"]
    assert check["text_only_call"]["prompt_tokens"] > 0
    assert check["image_call"]["input_tokens"] > 0


def test_mechanism_check_shows_flat_cost():
    """Headline finding: unreduced and downscale_50 are byte-identical on
    every real sample tested."""
    data = load_fixture()
    for sample in data["mechanism_check"]["results"]:
        assert sample["unreduced_tokens"] == sample["downscale_50_tokens"]


def test_extreme_downscale_still_flat():
    """Rules out 'just rounds into the same tile bucket at 50%' — even a
    genuine 10% thumbnail costs the same as full resolution."""
    data = load_fixture()
    check = data["mechanism_check"]["extreme_downscale_check"]
    assert check["matches_unreduced"] is True


def test_detail_parameter_has_no_effect():
    """Contrasts with Gemini's media_resolution, which DID change cost.
    Groq's detail parameter is accepted but inert for this model."""
    data = load_fixture()
    tokens = {r["tokens"] for r in data["detail_parameter_check"]["results"]}
    assert len(tokens) == 1, f"Expected detail parameter to have zero effect, got varying tokens: {tokens}"


def test_full_comparison_all_pixel_baselines_identical_to_unreduced():
    """The decisive finding: every pixel-space transform (downscale, JPEG,
    crop, low-detail) is byte-identical to unreduced on every sample."""
    data = load_fixture()
    pixel_only_keys = ["unreduced", "uniform_downscale_50", "jpeg_quality_50", "fixed_ratio_crop_60", "low_detail"]
    for sample in data["full_go_no_go_comparison"]["samples"]:
        values = {sample[k] for k in pixel_only_keys}
        assert len(values) == 1, f"{sample['filename']}: expected all pixel baselines identical, got {values}"


def test_text_adding_conditions_cost_more_not_less():
    """ocr_everything and full_pipeline_hybrid add OCR text on top of an
    already-fixed image cost - they must cost MORE than the pixel baselines,
    not less, confirming the mechanism (adding content can only add cost
    when image cost itself is invariant)."""
    data = load_fixture()
    for sample in data["full_go_no_go_comparison"]["samples"]:
        assert sample["ocr_everything"] > sample["unreduced"]
        assert sample["full_pipeline_hybrid"] > sample["unreduced"]


def test_zero_spend():
    data = load_fixture()
    assert data["full_go_no_go_comparison"]["cost_usd"] == 0

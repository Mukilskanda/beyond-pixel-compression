"""Regression check against real Gemini 2.5 Flash token counts captured once
(tests/fixtures/gemini_token_cost_2026-08-16.json) so future changes to the
cost-model discussion don't require spending API calls again to re-verify
what was already found. This does NOT call any real API — it locks in the
captured numbers and the headline finding they support.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pixelpath.packing import GEMINI_2_5_FLASH_FLAT_IMAGE_TOKENS, gemini_2_5_flash_image_cost

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


def test_gemini_cost_function_returns_flat_constant():
    """gemini_2_5_flash_image_cost must always return the empirically-verified
    flat constant regardless of dimensions — that IS the finding, not an
    implementation detail to be smart about."""
    for w, h in [(10, 10), (32, 32), (100, 100), (1653, 2339), (8000, 8000)]:
        assert gemini_2_5_flash_image_cost(w, h) == GEMINI_2_5_FLASH_FLAT_IMAGE_TOKENS


def test_mechanism_investigation_extreme_sizes_confirm_flat_cost():
    """The follow-up investigation (10x10 to 8000x8000, well beyond the
    original 100x100-3000x4000 range) must show the same flat cost — this
    is what rules out 'flat only within the range originally tested'."""
    data = load_fixture()
    extreme = data["mechanism_investigation"]["extreme_size_test"]["results"]
    assert len(extreme) == 3
    assert {r["image_tokens"] for r in extreme} == {258}


def test_mechanism_investigation_content_variation_confirms_not_content_adaptive():
    """Blank white and dense random noise at the same resolution must cost
    the same — this is what rules out a content-adaptive quantization
    explanation for the flat cost."""
    data = load_fixture()
    content_results = data["mechanism_investigation"]["content_variation_test"]["results"]
    assert len(content_results) == 2
    assert {r["image_tokens"] for r in content_results} == {258}


def test_mechanism_investigation_documents_the_doc_contradiction():
    """Locks in that the documented Google formula was checked and found to
    contradict the empirical result — not silently assumed to agree."""
    data = load_fixture()
    doc_check = data["mechanism_investigation"]["documentation_check"]
    assert doc_check["model_specific_override_found"] is False
    assert "contradict" in doc_check["contradiction"].lower()

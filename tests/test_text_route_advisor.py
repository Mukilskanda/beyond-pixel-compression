import numpy as np
import pytest

from experiments.fit_text_route_advisor import FEATURE_NAMES, _features_vector, fit_and_evaluate
from profiler.features import ProfileFeatures


def test_features_vector_matches_declared_order():
    f = ProfileFeatures(
        text_density=0.1, whitespace_ratio=0.2, layout_regularity=0.3, color_entropy=0.4, spatial_variance=0.5
    )
    assert _features_vector(f) == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert len(FEATURE_NAMES) == 5


def _synthetic_dataset(n_per_class: int = 15) -> list[dict]:
    """A synthetic dataset with real, deliberate signal (safe examples have
    high whitespace_ratio, unsafe ones low) — used only to sanity-check the
    fitting/evaluation logic, not to claim anything about real documents."""
    rng = np.random.default_rng(0)
    dataset = []
    for _ in range(n_per_class):
        dataset.append(
            {
                "features": [0.1, float(rng.uniform(0.7, 0.9)), 0.3, 4.0, 100.0],
                "safe": 1,
            }
        )
        dataset.append(
            {
                "features": [0.1, float(rng.uniform(0.0, 0.2)), 0.3, 4.0, 100.0],
                "safe": 0,
            }
        )
    return dataset


def test_fit_and_evaluate_returns_valid_probabilities_and_coefficients():
    result = fit_and_evaluate(_synthetic_dataset())
    assert 0.0 <= result["loo_accuracy"] <= 1.0
    assert 0.0 <= result["baseline_always_safe_accuracy"] <= 1.0
    assert set(result["coefficients"].keys()) == set(FEATURE_NAMES)


def test_fit_and_evaluate_recovers_real_signal_on_separable_synthetic_data():
    """With a deliberately strong, clean synthetic signal (unlike the real
    n=32 dataset), the model should clearly beat the trivial baseline —
    this pins that the fitting/evaluation machinery itself is correct, so a
    real result failing to beat baseline (as it did on real data) is a
    finding about the real data, not a bug in this code."""
    result = fit_and_evaluate(_synthetic_dataset())
    assert result["beats_best_trivial_baseline"] is True
    assert result["loo_accuracy"] > 0.8


def test_fit_and_evaluate_reports_correct_class_counts():
    result = fit_and_evaluate(_synthetic_dataset(n_per_class=10))
    assert result["n"] == 20
    assert result["n_safe"] == 10
    assert result["n_unsafe"] == 10


def test_fit_and_evaluate_handles_all_safe_dataset_without_crashing():
    """An all-one-class dataset can't have a meaningful LOO fold (training
    on 33/34 of one class only) — this must degrade gracefully (skip
    unscoreable folds), not raise."""
    dataset = [{"features": [0.1, 0.5, 0.3, 4.0, 100.0], "safe": 1} for _ in range(10)]
    result = fit_and_evaluate(dataset)
    assert result["n_unsafe"] == 0
    assert result["baseline_always_unsafe_accuracy"] == 0.0

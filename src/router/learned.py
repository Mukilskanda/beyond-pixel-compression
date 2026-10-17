"""Week 9: learned router — a gradient-boosted classifier over Stage 1 features,
distilled from the label-generation sweep (plan §4). Must beat router/rules.py
(the v1 baseline) to be worth shipping; that comparison is the ablation itself.

STUB — needs real label data from the §4 sweep before it can be trained:
    (features, setting) -> (tokens saved, fidelity retained)
    label = argmax_setting [fidelity - lambda * cost]

Do not train this against synthetic fixtures — the whole point is learning
routing decisions an expert oracle sweep discovered, and synthetic fixtures
have no real API cost/fidelity signal to distill from.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from profiler.features import ProfileFeatures
from router.rules import Route


class LearnedRouter:
    def __init__(self, model=None):
        self.model = model  # sklearn GradientBoostingClassifier once trained
        self._feature_order = [
            "text_density",
            "whitespace_ratio",
            "layout_regularity",
            "color_entropy",
            "spatial_variance",
        ]

    def _to_vector(self, features: ProfileFeatures) -> list[float]:
        d = asdict(features)
        return [d[name] for name in self._feature_order]

    def predict(self, features: ProfileFeatures, query: str) -> Route:
        if self.model is None:
            raise RuntimeError(
                "LearnedRouter has no trained model — run train() first, or use router.rules.route() as the v1 baseline"
            )
        vector = self._to_vector(features)
        # TODO(week 9): incorporate query-intent features (currently rules.py
        # handles spatial-intent keywords separately; decide whether the
        # learned model should subsume that or keep it as a hard override).
        label = self.model.predict([vector])[0]
        return Route(label)

    def train(
        self, feature_rows: list[ProfileFeatures], labels: list[Route], save_path: str | Path | None = None
    ) -> None:
        from sklearn.ensemble import GradientBoostingClassifier

        X = [self._to_vector(f) for f in feature_rows]
        y = [label.value for label in labels]

        self.model = GradientBoostingClassifier(random_state=0)
        self.model.fit(X, y)

        if save_path is not None:
            import joblib

            joblib.dump(self.model, save_path)

    def load(self, path: str | Path) -> None:
        import joblib

        self.model = joblib.load(path)

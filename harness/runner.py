"""Experiment orchestration: run a (image, prompt) set through a provider, log results to SQLite."""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from typing import Callable, Iterable

try:
    from harness.cache import ResponseCache
    from harness.metrics import anls, exact_match
    from harness.providers import AnthropicProvider, OllamaProvider
except ImportError:  # running as a script from inside harness/
    from cache import ResponseCache
    from metrics import anls, exact_match
    from providers import AnthropicProvider, OllamaProvider


class ResultDB:
    def __init__(self, db_path: str | Path = "results/results.sqlite"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset TEXT,
                image_path TEXT,
                method TEXT,
                model TEXT,
                prompt TEXT,
                response_text TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                latency_s REAL,
                anls_score REAL,
                exact_match INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.commit()

    def log(self, dataset: str, image_path: str, method: str, result, ground_truths: list[str] | None = None, prompt: str | None = None) -> None:
        anls_score = None
        em_score = None
        if ground_truths:
            anls_score = anls(result.text, ground_truths)
            em_score = int(exact_match(result.text, ground_truths))

        self._conn.execute(
            "INSERT INTO runs (dataset, image_path, method, model, prompt, response_text, "
            "input_tokens, output_tokens, latency_s, anls_score, exact_match) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                dataset, image_path, method, result.model, prompt, result.text,
                result.input_tokens, result.output_tokens, result.latency_s,
                anls_score, em_score,
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def run_dataset(
    samples: Iterable[tuple[str, str, list[str]]],  # (image_path, question, ground_truth_answers)
    dataset_name: str,
    method: str,
    provider,
    db: ResultDB,
) -> None:
    for image_path, question, ground_truths in samples:
        result = provider.call(image_path, question)
        db.log(dataset_name, image_path, method, result, ground_truths, question)


def run_baseline_sweep(
    samples: Iterable[tuple[str, str, list[str]]],  # (image_path, question, ground_truth_answers)
    dataset_name: str,
    methods: dict[str, Callable[[str], bytes]],  # name -> fn(image_path) -> transformed bytes
    provider,
    db: ResultDB,
) -> None:
    """Week-2 go/no-go checkpoint: run every baseline transform against the same
    samples through one provider, so cost-fidelity numbers are directly comparable.
    """
    samples = list(samples)
    with tempfile.TemporaryDirectory() as tmp:
        for method_name, transform_fn in methods.items():
            for image_path, question, ground_truths in samples:
                transformed_bytes = transform_fn(image_path)
                tmp_path = Path(tmp) / f"{method_name}_{Path(image_path).name}"
                tmp_path.write_bytes(transformed_bytes)
                result = provider.call(str(tmp_path), question)
                db.log(dataset_name, image_path, method_name, result, ground_truths, question)


if __name__ == "__main__":
    cache = ResponseCache()
    db = ResultDB()

    # Dev iteration: local, free, unlimited.
    dev_provider = OllamaProvider(model="llava", cache=cache)

    # Real measurement runs (uncomment when ready to spend real API budget —
    # see plan §6 cost controls, and confirm the cache is warm first):
    # real_provider = AnthropicProvider(model="claude-opus-5", cache=cache)

    print("Harness ready. Load DocVQA via data/docvqa_loader.py, then call "
          "run_baseline_sweep() with baselines.baselines transforms for the "
          "week-2 checkpoint (see experiments/week2_baselines.yaml).")

import sqlite3

from harness.providers import CallResult
from harness.runner import ResultDB, run_baseline_sweep
from baselines.baselines import uniform_downscale, unreduced


class StubProvider:
    def __init__(self, answer: str = "500.00"):
        self.answer = answer

    def call(self, image_path, prompt, **kw):
        return CallResult(
            text=self.answer,
            input_tokens=len(open(image_path, "rb").read()) // 4,
            output_tokens=5,
            latency_s=0.001,
            model="stub",
            raw={},
        )


def test_result_db_creates_schema(tmp_path):
    db = ResultDB(tmp_path / "results.sqlite")
    conn = sqlite3.connect(tmp_path / "results.sqlite")
    cols = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
    assert {"dataset", "method", "input_tokens", "anls_score", "exact_match"} <= cols
    db.close()


def test_log_computes_fidelity_against_ground_truth(tmp_path, sample_image):
    db = ResultDB(tmp_path / "results.sqlite")
    result = StubProvider(answer="500.00").call(sample_image, "total?")
    db.log("test", sample_image, "unreduced", result, ground_truths=["$500.00"], prompt="total?")

    conn = sqlite3.connect(tmp_path / "results.sqlite")
    row = conn.execute("SELECT anls_score, exact_match FROM runs").fetchone()
    assert row[0] is not None and row[0] > 0.5
    db.close()


def test_run_baseline_sweep_logs_all_methods(tmp_path, sample_image):
    db = ResultDB(tmp_path / "results.sqlite")
    methods = {"unreduced": unreduced, "downscale_50": lambda p: uniform_downscale(p, 0.5)}
    samples = [(sample_image, "total?", ["$500.00"])]

    run_baseline_sweep(samples, "test", methods, StubProvider(answer="$500.00"), db)

    conn = sqlite3.connect(tmp_path / "results.sqlite")
    rows = conn.execute("SELECT method FROM runs").fetchall()
    assert {r[0] for r in rows} == {"unreduced", "downscale_50"}
    db.close()

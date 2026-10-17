import sqlite3

from harness.providers import CallResult
from harness.runner import ResultDB, run_baseline_sweep
from baselines.baselines import uniform_downscale, unreduced


class StubProvider:
    def __init__(self, answer: str = "500.00", finish_reason: str | None = None):
        self.answer = answer
        self.finish_reason = finish_reason

    def call(self, image_path, prompt, **kw):
        return CallResult(
            text=self.answer,
            input_tokens=len(open(image_path, "rb").read()) // 4,
            output_tokens=5,
            latency_s=0.001,
            model="stub",
            raw={},
            finish_reason=self.finish_reason,
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


# --- Fidelity pipeline audit (Step 1) regression coverage ---
# Locks in that ResultDB.log() never scores a truncated response as if it
# were a real fidelity measurement - the generalized fix for the
# qwen/qwen3.6-27b <think>-preamble bug (paper/draft.md §9.4), where a
# truncated response scored anls=0.0 indistinguishably from a genuine wrong
# answer.


def test_log_does_not_score_truncated_response(tmp_path, sample_image):
    db = ResultDB(tmp_path / "results.sqlite")
    result = StubProvider(answer="500", finish_reason="length").call(sample_image, "total?")
    db.log("test", sample_image, "unreduced", result, ground_truths=["$500.00"], prompt="total?")

    conn = sqlite3.connect(tmp_path / "results.sqlite")
    row = conn.execute("SELECT anls_score, exact_match, truncated, finish_reason FROM runs").fetchone()
    assert row[0] is None and row[1] is None, "truncated response must not get a real fidelity score"
    assert row[2] == 1
    assert row[3] == "length"
    db.close()


def test_log_scores_normally_when_not_truncated(tmp_path, sample_image):
    db = ResultDB(tmp_path / "results.sqlite")
    result = StubProvider(answer="$500.00", finish_reason="stop").call(sample_image, "total?")
    db.log("test", sample_image, "unreduced", result, ground_truths=["$500.00"], prompt="total?")

    conn = sqlite3.connect(tmp_path / "results.sqlite")
    row = conn.execute("SELECT anls_score, truncated FROM runs").fetchone()
    assert row[0] == 1.0
    assert row[1] == 0
    db.close()


def test_log_unknown_finish_reason_still_scores(tmp_path, sample_image):
    """finish_reason=None (e.g. OllamaProvider, which never populates it)
    must not be treated as truncated by default - unknown is not the same
    as confirmed-truncated."""
    db = ResultDB(tmp_path / "results.sqlite")
    result = StubProvider(answer="$500.00", finish_reason=None).call(sample_image, "total?")
    db.log("test", sample_image, "unreduced", result, ground_truths=["$500.00"], prompt="total?")

    conn = sqlite3.connect(tmp_path / "results.sqlite")
    row = conn.execute("SELECT anls_score, truncated FROM runs").fetchone()
    assert row[0] == 1.0
    assert row[1] == 0
    db.close()


def test_result_db_migrates_old_schema_missing_truncation_columns(tmp_path):
    """A DB created before the truncated/finish_reason columns existed must
    not break when opened by the updated ResultDB - real prior run data
    (results/results.sqlite) predates this change and must not be silently
    dropped or require manual migration."""
    db_path = tmp_path / "old_results.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset TEXT, image_path TEXT, method TEXT, model TEXT, prompt TEXT,
            response_text TEXT, input_tokens INTEGER, output_tokens INTEGER,
            latency_s REAL, anls_score REAL, exact_match INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute("INSERT INTO runs (dataset, method, response_text) VALUES ('old', 'unreduced', 'pre-existing row')")
    conn.commit()
    conn.close()

    db = ResultDB(db_path)  # must not raise
    conn = sqlite3.connect(db_path)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
    assert {"truncated", "finish_reason"} <= cols
    old_row = conn.execute("SELECT response_text, truncated FROM runs WHERE dataset='old'").fetchone()
    assert old_row[0] == "pre-existing row"  # prior real data preserved
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

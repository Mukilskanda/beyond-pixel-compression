"""Human audit of the fidelity judge (paper Sec. VI-A).

Input:  audit/judge_validation_sample_scored.csv
        100 judged responses sampled with seed 42. Column human_correct_1_or_0
        records whether the human AGREED with the judge (1) or not (0); it was
        filled in with audit/grade_judge_sample.py.
        The judge's own verdict is read from results/cache.sqlite.
Output: agreement, Cohen's kappa (overall and per dataset) and the confusion
        matrix on stdout. Paper: 86/100, kappa 0.63; DocVQA 40/48 (0.41);
        ChartQA 46/52 (0.74). The 3 baseline rows are DocVQA pages.

Usage (from the repository root): python audit/compute_agreement.py
"""

from __future__ import annotations

import csv
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "analysis")]

from harness.metrics import _strip_think_block  # noqa: E402
from revision_recount import load_judge_cache  # noqa: E402


def kappa(pairs: list[tuple[bool, bool]]) -> tuple[int, int, float]:
    n = len(pairs)
    agree = sum(j == h for j, h in pairs)
    pj = sum(j for j, _ in pairs) / n
    ph = sum(h for _, h in pairs) / n
    pe = pj * ph + (1 - pj) * (1 - ph)
    return agree, n, (agree / n - pe) / (1 - pe)


def main() -> None:
    judge = load_judge_cache()
    db = sqlite3.connect(ROOT / "results/results.sqlite")
    groups: dict[str, list[tuple[bool, bool]]] = {"all": [], "docvqa": [], "chartqa": []}
    with open(ROOT / "audit/judge_validation_sample_scored.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            prompt, resp = db.execute(
                "SELECT prompt, response_text FROM runs WHERE id=?", (int(row["row_id"]),)
            ).fetchone()
            j = judge[(prompt, _strip_think_block(resp) if resp else "")][0]
            h = j if row["human_correct_1_or_0"].strip() == "1" else (not j)
            ds = "docvqa" if row["dataset"].startswith("docvqa") else "chartqa"
            groups["all"].append((j, h))
            groups[ds].append((j, h))
    for name, pairs in groups.items():
        agree, n, k = kappa(pairs)
        print(f"{name:<8} agreement {agree}/{n} ({100 * agree / n:.1f}%)  Cohen's kappa {k:.2f}")
    v = groups["all"]
    print(
        "confusion (judge, human): "
        f"YES/YES={sum(j and h for j, h in v)}  YES/NO={sum(j and not h for j, h in v)}  "
        f"NO/YES={sum(h and not j for j, h in v)}  NO/NO={sum(not j and not h for j, h in v)}"
    )


if __name__ == "__main__":
    main()

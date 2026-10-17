"""Interactive grader used to build the human audit (paper Sec. VI-A).

Input:  audit/judge_validation_sample.csv (100 judged responses, seed 42)
Output: audit/judge_validation_sample_scored.csv, where human_correct_1_or_0 = 1
        means the human agreed with the judge's verdict. Resumable.
Agreement and Cohen's kappa are computed by audit/compute_agreement.py.

Usage (from the repository root): python audit/grade_judge_sample.py
"""
import csv
import sys

INPUT = "audit/judge_validation_sample.csv"
OUTPUT = "audit/judge_validation_sample_scored.csv"

with open(INPUT, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

start = 0
try:
    with open(OUTPUT, encoding="utf-8") as f:
        done = list(csv.DictReader(f))
    start = len(done)
    rows[:start] = done
    print(f"Resuming at row {start + 1}/{len(rows)}")
except FileNotFoundError:
    pass

agree = sum(1 for r in rows[:start] if r["human_correct_1_or_0"] == "1")

for i in range(start, len(rows)):
    r = rows[i]
    print(f"\n--- Row {i + 1}/{len(rows)} ---")
    print(f"Question: {r['question']}")
    print(f"Model response: {r['model_response']}")
    while True:
        ans = input("Agree with judge's scoring? (1=yes, 0=no, q=quit): ").strip()
        if ans in ("1", "0"):
            break
        if ans == "q":
            with open(OUTPUT, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows[:i])
            print(f"Saved progress at row {i}. Re-run to resume.")
            sys.exit(0)
        print("Type 1, 0, or q")
    r["human_correct_1_or_0"] = ans
    if ans == "1":
        agree += 1

    with open(OUTPUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows[: i + 1])

pct = 100 * agree / len(rows)
print(f"\nFinal human-AI agreement: {agree}/{len(rows)} = {pct:.1f}%")

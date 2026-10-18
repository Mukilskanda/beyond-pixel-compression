"""Revision-2 step 5b: single-annotator review sheets (no judge verdicts shown).

1. audit/flip_review.csv: every paired sample in which the image route and
   the text route disagree on the judge's verdict (the paper's 146 pairs:
   32 forward flips and 2 reversals), plus any disagreement in the Qwen3.8
   ChartQA replication (results/revision_chartqa_textroute.jsonl) once it
   exists. Columns: question, ground truth, both answers, an OCR excerpt
   around the answer, and empty columns for the author's labels. The judge's
   verdicts are used only to select the rows and are NOT written to the sheet.
2. audit/judge_validation_sample_blind.csv: the 100 audit answers without
   the judge's verdict or the earlier agreement label.

This is a single-annotator re-check by the author, not an independent
inter-rater agreement measurement.

Usage (from the repository root): python analysis/build_review_sheets.py
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "analysis")]

from harness.metrics import _strip_think_block  # noqa: E402
from revision_recount import best_rows, load_judge_cache  # noqa: E402

FLIPS = ROOT / "audit/flip_review.csv"
BLIND = ROOT / "audit/judge_validation_sample_blind.csv"
REPL = ROOT / "results/revision_chartqa_textroute.jsonl"


def ocr_excerpt(image_path: str, gts: list[str], width: int = 160) -> str:
    from textpath.compress import compress_markdown
    from textpath.ocr import image_to_markdown

    text = compress_markdown(image_to_markdown(image_path.replace("\\", "/")).markdown)
    flat = " ".join(text.split())
    for g in sorted(gts, key=len, reverse=True):
        m = re.search(re.escape(g.strip()), flat, re.IGNORECASE)
        if g.strip() and m:
            a, b = max(0, m.start() - width), min(len(flat), m.end() + width)
            return ("..." if a else "") + flat[a:b] + ("..." if b < len(flat) else "")
    return "[answer not found in OCR text] " + flat[: 2 * width] + ("..." if len(flat) > 2 * width else "")


def main() -> None:
    judge = load_judge_cache()
    verdicts = {}
    for key, r in best_rows().items():
        cand = _strip_think_block(r[5]) if r[5] else (r[5] or "")
        hit = judge.get((r[4], cand))
        if hit is not None:
            verdicts[key] = (hit[0], hit[1], r[4], cand, r[0])
    rows = []
    for img, model in sorted({(k[0], k[2]) for k in verdicts}):
        ur = verdicts.get((img, "unreduced", model))
        tm = "text_compressed" if (img, "text_compressed", model) in verdicts else "text_raw"
        tr = verdicts.get((img, tm, model))
        if not ur or not tr or ur[0] is None or tr[0] is None or ur[0] == tr[0]:
            continue
        gts = ur[1].split(" | ")
        rows.append({
            "case_id": f"P{len(rows) + 1:02d}", "source": "paper (Qwen3.6 / Gemini, Aug 2026)",
            "dataset": ur[4], "model": model, "image": Path(img.replace("\\", "/")).name,
            "question": ur[2], "ground_truth": ur[1], "image_route_answer": ur[3], "text_route_answer": tr[3],
            "ocr_excerpt": ocr_excerpt(img, gts),
            "human_label_image": "", "human_label_text": "", "failure_mode": "", "notes": "",
        })
    n_paper = len(rows)
    if REPL.exists():
        con = sqlite3.connect(ROOT / "results/results.sqlite")
        for line in REPL.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["image_correct"] is None or r["text_correct"] is None or r["image_correct"] == r["text_correct"]:
                continue
            ans = dict(con.execute(
                "SELECT method, response_text FROM runs WHERE dataset='rev2_chartqa_textroute_q38' AND image_path LIKE ?",
                (f"%{r['sample']}",)).fetchall())
            rows.append({
                "case_id": f"R{len(rows) - n_paper + 1:02d}", "source": "Qwen3.8 ChartQA replication (Oct 2026)",
                "dataset": "chartqa", "model": r["model"], "image": r["sample"], "question": r["question"],
                "ground_truth": " | ".join(r["ground_truths"]),
                "image_route_answer": _strip_think_block(ans.get("unreduced", "") or ""),
                "text_route_answer": _strip_think_block(ans.get("text_compressed", "") or ""),
                "ocr_excerpt": ocr_excerpt(f"data/chartqa_cache/{r['sample']}", r["ground_truths"]),
                "human_label_image": "", "human_label_text": "", "failure_mode": "", "notes": "",
            })
    with open(FLIPS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{FLIPS.name}: {n_paper} paper flips, {len(rows) - n_paper} replication flips")

    with open(ROOT / "audit/judge_validation_sample_scored.csv", encoding="utf-8") as f:
        audit = list(csv.DictReader(f))
    keep = [k for k in audit[0] if k not in ("human_correct_1_or_0", "notes")]
    with open(BLIND, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keep + ["human_label_correct", "notes"])
        w.writeheader()
        for a in audit:
            w.writerow({**{k: a[k] for k in keep}, "human_label_correct": "", "notes": ""})
    print(f"{BLIND.name}: {len(audit)} answers, no judge verdicts")


if __name__ == "__main__":
    main()

"""Revision-2 step D: ChartQA text route on Groq Qwen3.8-27B (separate replication).

The Sec. IV-C ChartQA result (n=13 pairs, McNemar p=0.06) used the withdrawn
Qwen3.6-27B and cannot be extended. This is a new replication on the successor
model and is never pooled with the Qwen3.6 data.

Items: 40 ChartQA test items chosen with seed 42 from the first 120 in loader
order, excluding every chart used by any earlier Qwen3.6 ChartQA run. The IDs
are written to results/revision_chartqa_textroute_ids.json before any call.

For each item, on Qwen3.8-27B with the dataset question:
  image route  the unreduced chart image;
  text route   the compressed Tesseract OCR text, no image.
Both answers are scored by the same judge (openai/gpt-oss-120b,
reasoning_effort=low, harness/judge.py). Input/output tokens and finish
reasons are logged; truncated answers are recorded but not scored.

Calls go through harness.quota.metered (clean stop at the Groq daily cap).
Logged to results/results.sqlite (dataset tag rev2_chartqa_textroute_q38) and
one row per item to results/revision_chartqa_textroute.jsonl (the progress
file; rerunning resumes after the last completed item).

Usage (from the repository root): python experiments/revision_chartqa_textroute.py
"""

from __future__ import annotations

import json
import random
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.chartqa_loader import load_chartqa
from harness import quota
from harness.cache import ResponseCache
from harness.judge import judge_correctness
from harness.providers import GroqProvider
from harness.runner import ResultDB
from textpath.compress import compress_markdown
from textpath.ocr import image_to_markdown

MODEL = "qwen/qwen3.8-27b"
JUDGE = "openai/gpt-oss-120b"
N_ITEMS = 40
POOL = 120
MAX_TOKENS = 2000
OUT = Path("results/revision_chartqa_textroute.jsonl")
IDS = Path("results/revision_chartqa_textroute_ids.json")


class MeteredJudge:
    def __init__(self, p):
        self.p, self.model = p, p.model

    def call_text_only(self, *a, **k):
        return quota.metered("groq", self.p.model, lambda: self.p.call_text_only(*a, **k), est_tokens=700)


def truncated(r) -> bool:
    return (r.finish_reason or "") in ("length", "max_tokens")


def choose_items() -> list[str]:
    if IDS.exists():
        return json.loads(IDS.read_text(encoding="utf-8"))["items"]
    used = {
        Path(p.replace("\\", "/")).name
        for (p,) in sqlite3.connect("results/results.sqlite").execute(
            "SELECT DISTINCT image_path FROM runs WHERE model='qwen/qwen3.6-27b' AND image_path LIKE '%chartqa_cache%'"
        )
    }
    pool = [Path(s.image_path).name for s in load_chartqa(limit=POOL)]
    eligible = [x for x in pool if x not in used]
    items = sorted(random.Random(42).sample(eligible, N_ITEMS), key=pool.index)
    IDS.write_text(json.dumps({"seed": 42, "pool": POOL, "excluded_qwen36_items": sorted(used), "items": items},
                              indent=1), encoding="utf-8")
    return items


def main() -> None:
    items = set(choose_items())
    cache = ResponseCache("results/cache.sqlite")
    model = GroqProvider(model=MODEL, cache=cache)
    judge = MeteredJudge(GroqProvider(model=JUDGE, cache=cache))
    db = ResultDB("results/results.sqlite")
    done = set()
    if OUT.exists():
        done = {json.loads(line)["sample"] for line in OUT.read_text(encoding="utf-8").splitlines()}
    for s in load_chartqa(limit=POOL):
        name = Path(s.image_path).name
        if name not in items or name in done:
            continue
        if not quota.allow("groq", MODEL, est_tokens=6_000) or not quota.allow("groq", JUDGE, est_tokens=1_400):
            print(f"STOP: daily budget reached ({quota.status()})")
            return
        try:
            img = quota.metered("groq", MODEL, lambda: model.call(s.image_path, s.question, max_tokens=MAX_TOKENS),
                                est_tokens=3_000, note=f"chartqa image {name}")
            text = compress_markdown(image_to_markdown(s.image_path).markdown)
            txt = quota.metered("groq", MODEL, lambda: model.call_text_only(text, s.question, max_tokens=MAX_TOKENS),
                                est_tokens=3_000, note=f"chartqa text {name}")
            v_img = None if truncated(img) else judge_correctness(s.question, img.text or "", s.answers, judge)
            v_txt = None if truncated(txt) else judge_correctness(s.question, txt.text or "", s.answers, judge)
        except quota.BudgetExhausted as exc:
            print(f"STOP: {exc}")
            return
        db.log(dataset="rev2_chartqa_textroute_q38", image_path=str(s.image_path), method="unreduced",
               result=img, ground_truths=s.answers, prompt=s.question)
        db.log(dataset="rev2_chartqa_textroute_q38", image_path=str(s.image_path), method="text_compressed",
               result=txt, ground_truths=s.answers, prompt=s.question)
        row = {"sample": name, "model": img.model, "question": s.question, "ground_truths": s.answers,
               "image_in": img.input_tokens, "image_out": img.output_tokens, "image_finish": img.finish_reason,
               "text_in": txt.input_tokens, "text_out": txt.output_tokens, "text_finish": txt.finish_reason,
               "image_correct": v_img, "text_correct": v_txt, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{name:<22} image {img.input_tokens}/{img.output_tokens} {v_img} | text {txt.input_tokens}/{txt.output_tokens} {v_txt}")
        time.sleep(6)
    print("ChartQA replication complete;", quota.status())


if __name__ == "__main__":
    main()

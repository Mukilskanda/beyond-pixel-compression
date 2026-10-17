"""Paired per-sample fidelity comparison: image-route (unreduced) vs
text-route (text_raw/text_compressed), using harness/judge.py (all cached
from experiments/judge_rescore.py runs - zero new API calls). Where a
(image_path, method, model) triple has multiple real-call attempts (e.g.
an original max_tokens=1024 run and a later max_tokens=2000 retry), prefers
the row with the most output_tokens (the more complete/least-truncated
attempt) so each sample is counted once, not once per retry.
"""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.docvqa_loader import load_docvqa
from harness.cache import ResponseCache
from harness.judge import judge_correctness
from harness.providers import GroqProvider


def main() -> None:
    conn = sqlite3.connect("results/results.sqlite")
    rows = conn.execute(
        "SELECT id, image_path, method, model, prompt, response_text, output_tokens FROM runs "
        "WHERE method IN ('unreduced','text_raw','text_compressed') AND truncated = 0 ORDER BY output_tokens DESC"
    ).fetchall()

    # Keep only the most-complete row per (image_path, method, model)
    best: dict[tuple[str, str, str], tuple] = {}
    for row in rows:
        row_id, image_path, method, model, prompt, response_text, output_tokens = row
        key = (image_path, method, model)
        if key not in best:
            best[key] = row

    samples = {str(s.image_path): s.answers for s in load_docvqa(limit=40)}
    cache = ResponseCache("results/cache.sqlite")
    judge = GroqProvider(model="openai/gpt-oss-120b", cache=cache)

    verdicts: dict[tuple[str, str, str], bool | None] = {}
    for key, row in best.items():
        row_id, image_path, method, model, prompt, response_text, output_tokens = row
        gts = samples.get(image_path)
        if not gts:
            continue
        verdicts[key] = judge_correctness(prompt, response_text or "", gts, judge)
        time.sleep(0.3)

    for provider_model in ("gemini-2.5-flash", "qwen/qwen3.6-27b"):
        image_paths = sorted({k[0] for k in verdicts if k[2] == provider_model})
        pairs_ur_tr = []
        pairs_ur_tc = []
        for path in image_paths:
            ur = verdicts.get((path, "unreduced", provider_model))
            tr = verdicts.get((path, "text_raw", provider_model))
            tc = verdicts.get((path, "text_compressed", provider_model))
            if ur is not None and tr is not None:
                pairs_ur_tr.append((path, ur, tr))
            if ur is not None and tc is not None:
                pairs_ur_tc.append((path, ur, tc))

        print(f"\n=== {provider_model}: unreduced vs text_raw ({len(pairs_ur_tr)} paired samples) ===")
        for path, ur, tr in pairs_ur_tr:
            flag = "" if ur == tr else "  <-- DIFFERS"
            print(f"  {Path(path).stem:<12} unreduced={ur!s:<6} text_raw={tr!s:<6}{flag}")
        both_correct = sum(1 for _, a, b in pairs_ur_tr if a and b)
        img_only = sum(1 for _, a, b in pairs_ur_tr if a and not b)
        txt_only = sum(1 for _, a, b in pairs_ur_tr if not a and b)
        both_wrong = sum(1 for _, a, b in pairs_ur_tr if not a and not b)
        print(
            f"  both correct={both_correct}, image-only-correct={img_only}, text-only-correct={txt_only}, both wrong={both_wrong}"
        )


if __name__ == "__main__":
    main()

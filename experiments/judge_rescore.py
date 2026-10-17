"""Re-score this session's real responses with harness/judge.judge_correctness()
instead of ANLS — the fidelity gap flagged in paper/draft.md §12.4-12.5:
ANLS penalized correct-but-verbose answers identically to wrong ones on
BOTH conditions (text-route and image-route), so no real fidelity verdict
could be drawn from raw ANLS scores.

Re-scores rows ALREADY in results/results.sqlite (id >= 29, today's
session) for methods unreduced/text_raw/text_compressed/min_conf_0/
min_conf_60 that are not truncated. No new Groq/Gemini calls for the
answers themselves - only judge calls (openai/gpt-oss-120b on Groq, a
different model family from qwen/qwen3.6-27b, avoiding the judge scoring
its own qwen-produced answers; free tier). Cached via ResponseCache like
any other call.

Usage: python experiments/judge_rescore.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.docvqa_loader import load_docvqa
from harness.cache import ResponseCache
from harness.judge import judge_correctness
from harness.providers import GroqProvider

JUDGE_MODEL = "openai/gpt-oss-120b"


def main() -> None:
    import sqlite3

    conn = sqlite3.connect("results/results.sqlite")
    rows = conn.execute(
        "SELECT id, image_path, method, model, prompt, response_text FROM runs "
        "WHERE id >= 29 AND method IN ('unreduced','text_raw','text_compressed','min_conf_0','min_conf_60') "
        "AND truncated = 0 ORDER BY id"
    ).fetchall()

    samples = {str(s.image_path): s.answers for s in load_docvqa(limit=40)}

    cache = ResponseCache("results/cache.sqlite")
    judge = GroqProvider(model=JUDGE_MODEL, cache=cache)

    results = []
    for row_id, image_path, method, model, prompt, response_text in rows:
        gts = samples.get(image_path)
        if not gts:
            print(f"  [row {row_id}] SKIP - no ground truth found for {image_path}")
            continue
        verdict = judge_correctness(prompt, response_text or "", gts, judge)
        results.append((image_path, method, model, verdict))
        print(f"  [row {row_id}] {model:<20}{method:<18}verdict={verdict}")
        time.sleep(1.0)  # stay comfortably under Groq's TPM

    print(f"\nTotal judged: {len(results)}")

    # Aggregate by (model, method)
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0, 0])  # correct, incorrect, unknown
    for _, method, model, verdict in results:
        key = (model, method)
        if verdict is True:
            agg[key][0] += 1
        elif verdict is False:
            agg[key][1] += 1
        else:
            agg[key][2] += 1

    print(f"\n{'model':<20}{'method':<20}{'correct':>10}{'incorrect':>12}{'unknown':>10}{'accuracy':>12}")
    for (model, method), (correct, incorrect, unknown) in sorted(agg.items()):
        total_scored = correct + incorrect
        acc = f"{correct / total_scored * 100:.1f}%" if total_scored else "n/a"
        print(f"{model:<20}{method:<20}{correct:>10}{incorrect:>12}{unknown:>10}{acc:>12}")


if __name__ == "__main__":
    main()

"""Judge-score the ChartQA text-route validation (experiments/chartqa_text_route_validation.py)
the same way experiments/judge_rescore.py does for DocVQA - same judge
model (openai/gpt-oss-120b on Groq), same judge_correctness() function,
same $0 free-tier discipline. Kept as a separate script (not a generalized
--dataset flag on judge_rescore.py) since this is one targeted replication
run, not a permanent multi-dataset harness feature.

Dedupes to the most-complete (highest output_tokens) row per
(image_path, method) - the retry-after-429 run re-logged already-cached
successes alongside newly-succeeded ones, so some (image_path, method)
pairs have 2 identical-content rows.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.chartqa_loader import load_chartqa
from harness.cache import ResponseCache
from harness.judge import judge_correctness
from harness.providers import GroqProvider

JUDGE_MODEL = "openai/gpt-oss-120b"


def main() -> None:
    import sqlite3

    conn = sqlite3.connect("results/results.sqlite")
    rows = conn.execute(
        "SELECT image_path, method, model, prompt, response_text, output_tokens FROM runs "
        "WHERE dataset = 'chartqa' AND truncated = 0 ORDER BY output_tokens DESC"
    ).fetchall()

    best: dict[tuple[str, str], tuple] = {}
    for row in rows:
        image_path, method, model, prompt, response_text, output_tokens = row
        key = (image_path, method)
        if key not in best:
            best[key] = row

    samples = {str(s.image_path): s.answers for s in load_chartqa(limit=15)}

    cache = ResponseCache("results/cache.sqlite")
    judge = GroqProvider(model=JUDGE_MODEL, cache=cache)

    results = []
    for (image_path, method), row in sorted(best.items()):
        _, _, model, prompt, response_text, _ = row
        gts = samples.get(image_path)
        if not gts:
            print(f"  SKIP - no ground truth for {image_path}")
            continue
        verdict = judge_correctness(prompt, response_text or "", gts, judge)
        results.append((image_path, method, model, verdict))
        print(f"  {Path(image_path).stem:<20}{method:<18}verdict={verdict}")
        time.sleep(1.0)

    from collections import defaultdict

    agg = defaultdict(lambda: [0, 0, 0])
    for _, method, model, verdict in results:
        if verdict is True:
            agg[method][0] += 1
        elif verdict is False:
            agg[method][1] += 1
        else:
            agg[method][2] += 1

    print(f"\n{'method':<20}{'correct':>10}{'incorrect':>12}{'unknown':>10}{'accuracy':>12}")
    for method, (correct, incorrect, unknown) in sorted(agg.items()):
        total_scored = correct + incorrect
        acc = f"{correct / total_scored * 100:.1f}%" if total_scored else "n/a"
        print(f"{method:<20}{correct:>10}{incorrect:>12}{unknown:>10}{acc:>12}")

    # Paired per-sample: unreduced vs text_raw
    verdict_map = {(p, m): v for p, m, _, v in results}
    image_paths = sorted({p for p, _ in best})
    print(f"\n--- paired unreduced vs text_raw ({len(image_paths)} images) ---")
    both_correct = img_only = txt_only = both_wrong = 0
    for path in image_paths:
        ur = verdict_map.get((path, "unreduced"))
        tr = verdict_map.get((path, "text_raw"))
        if ur is None or tr is None:
            continue
        if ur and tr:
            both_correct += 1
        elif ur and not tr:
            img_only += 1
        elif not ur and tr:
            txt_only += 1
        else:
            both_wrong += 1
    print(
        f"both correct={both_correct}, image-only-correct={img_only}, text-only-correct={txt_only}, both wrong={both_wrong}"
    )


if __name__ == "__main__":
    main()

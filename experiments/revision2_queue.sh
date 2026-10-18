#!/usr/bin/env bash
# Revision-2 run queue, order locked on 18 Oct 2026 (Option 3):
#   1. A  Qwen3.8 pixel-reduction sweep (Groq)            - must complete
#   2. D  ChartQA text route on Qwen3.8, n=20 committed   - must complete
#   3. C  remaining Table IV windows, from leftover quota:
#         Gemini windows (separate quota, never competes with A or D)
#         Groq/DocVQA windows (only after A and D are complete)
# Every 30 minutes each runner resumes where it stopped and exits cleanly at
# its provider's limit (harness/quota.py). Usage: ./experiments/revision2_queue.sh [ad|c]
#   ad (default) exits once A and D are complete; c runs the Groq/DocVQA windows.
cd "$(dirname "$0")/.."
LOG=results/revision2_queue.log
MODE="${1:-ad}"
lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
until [ "$(date +%Y%m%d%H)" -ge 2026102123 ]; do
  {
    echo "=== $(date '+%F %T') mode=$MODE ==="
    if [ "$MODE" = ad ]; then
      python experiments/qwen38_pixel_sweep.py
      [ "$(lines results/qwen38_pixel_sweep.jsonl)" -ge 120 ] && python experiments/revision_chartqa_textroute.py
      python experiments/revision_table3.py --provider gemini --dataset docvqa --offsets 16,20,24,28 --conditions sep,multi,tiled --sleep 6
    else
      python experiments/revision_table3.py --provider groq --dataset docvqa --offsets 20,24,28 --conditions sep,tiled,pertile --sleep 10
      python experiments/revision_table3.py --provider gemini --dataset docvqa --offsets 16,20,24,28 --conditions sep,multi,tiled --sleep 6
    fi
    python -c "import sys; sys.path[:0]=['.','src']; from harness import quota; print(quota.status())"
  } >> "$LOG" 2>&1
  if [ "$MODE" = ad ] && [ "$(lines results/qwen38_pixel_sweep.jsonl)" -ge 120 ] && [ "$(lines results/revision_chartqa_textroute.jsonl)" -ge 20 ]; then
    echo "A and D complete at $(date '+%F %T')" | tee -a "$LOG"
    exit 0
  fi
  sleep 1800
done

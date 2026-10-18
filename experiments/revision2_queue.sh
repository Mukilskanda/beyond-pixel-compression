#!/usr/bin/env bash
# Revision-2 run queue (18-21 Oct 2026). Re-runs every runner in priority
# order every 30 minutes; each runner resumes where it stopped and exits
# cleanly at its provider's daily cap (harness/quota.py), so cached or
# completed work is never re-billed. Stops after 21 Oct 23:00 IST.
#   A  Qwen3.8 pixel-reduction sweep          (Groq)
#   B  Gemini size sweep, logged              (Gemini)
#   C  Table IV windows: Gemini, Groq ChartQA, Groq DocVQA
#   D  ChartQA text route on Qwen3.8          (Groq)
cd "$(dirname "$0")/.."
LOG=results/revision2_queue.log
until [ "$(date +%Y%m%d%H)" -ge 2026102123 ]; do
  {
    echo "=== $(date '+%F %T') ==="
    python experiments/qwen38_pixel_sweep.py
    python experiments/revision_gemini_size_sweep.py
    python experiments/revision_table3.py --provider gemini --dataset docvqa --offsets 16,20,24,28 --conditions sep,multi,tiled --sleep 6
    python experiments/revision_table3.py --provider groq --dataset chartqa --offsets 8,12,16 --conditions sep,tiled --sleep 10
    python experiments/revision_table3.py --provider groq --dataset docvqa --offsets 20,24,28 --conditions sep,tiled,pertile --sleep 10
    python experiments/revision_chartqa_textroute.py
    python -c "import sys; sys.path[:0]=['.','src']; from harness import quota; print(quota.status())"
  } >> "$LOG" 2>&1
  sleep 1800
done

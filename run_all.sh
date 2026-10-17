#!/usr/bin/env bash
# Reproduce the paper's tables, figures and statistics from the logged data.
# No API keys or network access are needed for these targets.
#   ./run_all.sh install   create .venv and install pinned requirements
#   ./run_all.sh tables    regenerate and check every reported number (120 values)
#   ./run_all.sh figures   regenerate the paper's figures into figures/
#   ./run_all.sh audit     judge vs. human agreement and Cohen's kappa
#   ./run_all.sh test      unit tests
#   ./run_all.sh all       tables, figures, audit, test
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
case "${1:-all}" in
  install) python -m venv .venv && { . .venv/bin/activate 2>/dev/null || . .venv/Scripts/activate; }
           pip install -r requirements.txt && pip install -e . ;;
  tables)  "$PY" analysis/reproduce_all.py ;;
  figures) "$PY" analysis/make_figures.py ;;
  audit)   "$PY" audit/compute_agreement.py ;;
  test)    "$PY" -m pytest tests/ -q ;;
  all)     "$0" tables && "$0" figures && "$0" audit && "$0" test ;;
  *)       echo "unknown target: $1"; exit 2 ;;
esac

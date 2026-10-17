#!/usr/bin/env bash
# Reproduce the paper's tables, figures and statistics from the logged data.
# No API keys or network access are needed for these targets.
#   ./run_all.sh install   create .venv and install pinned requirements
#   ./run_all.sh tables    regenerate and check every reported number (120 values)
#   ./run_all.sh figures   regenerate paper/figures/*.pdf
#   ./run_all.sh audit     judge vs. human agreement and Cohen's kappa
#   ./run_all.sh test      unit tests
#   ./run_all.sh paper     compile paper/main.pdf (needs pdflatex + bibtex)
#   ./run_all.sh all       tables, figures, audit, test
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
case "${1:-all}" in
  install) python -m venv .venv && . .venv/bin/activate 2>/dev/null || . .venv/Scripts/activate
           pip install -r requirements.txt && pip install -e . ;;
  tables)  "$PY" analysis/reproduce_all.py ;;
  figures) "$PY" analysis/make_figures.py ;;
  audit)   "$PY" audit/compute_agreement.py ;;
  test)    "$PY" -m pytest tests/ -q ;;
  paper)   export SOURCE_DATE_EPOCH=1792195200 FORCE_SOURCE_DATE=1   # fixed timestamp: byte-identical PDF
           cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null && bibtex main >/dev/null \
             && pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex | tail -1 ;;
  all)     "$0" tables && "$0" figures && "$0" audit && "$0" test ;;
  *)       echo "unknown target: $1"; exit 2 ;;
esac

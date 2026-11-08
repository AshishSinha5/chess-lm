#!/bin/bash
# Rebuild everything in results/ from the run logs. Run after each training stage, then commit results/.
set -e
cd "$(dirname "$0")/.."
source /workspace/venv/bin/activate
python scripts/summarize_runs.py "$@"
python scripts/summarize_tokens.py
python scripts/summarize_rate.py
python scripts/summarize_eval.py
python scripts/summarize_strict.py
python scripts/fit_scaling.py
if [ -f scripts/make_figures.py ]; then python scripts/make_figures.py; fi

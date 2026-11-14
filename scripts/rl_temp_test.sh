#!/bin/bash
# W4 step 5a: sampling temperature and start-point test (docs/RL_PLAN.md, baseline 5).
# 400 sampled games per cell: 100 openings x 2 colors x opponents 1320 and 1500, no legal mask.
# Usage: bash scripts/rl_temp_test.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/src STOCKFISH=${STOCKFISH:-/usr/local/bin/stockfish} OMP_NUM_THREADS=4
for CK in size_m bandit_m; do
  for T in 1.0 0.7 0.5 0.3; do
    python3 -m chesslm.rl.evaluate --ckpt $CK --mode sample --temperature $T --games 100 --split val \
      --opponents elo:1320,elo:1500 --engines 10 --seed 0 --out results/rl/temp_test/${CK}_T$T.json 2>&1 | grep -v saved
  done
done

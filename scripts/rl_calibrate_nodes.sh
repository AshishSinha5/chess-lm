#!/bin/bash
# Calibrate the Stockfish node limit (docs/RL_PLAN.md, DECISIONS G31). Run after scripts/setup_pod.sh.
# Phase "sweep": base M at several node limits. Pick the limit where M lenient Elo is near the stage-1 value (1410).
# Phase "sizes": base XS, S, M, L at one node limit. Check that the size order and gaps look like stage 1.
# Usage: bash scripts/rl_calibrate_nodes.sh sweep "40000 55000" [openings]
#        bash scripts/rl_calibrate_nodes.sh sizes 45000 [openings]      (openings: per opponent, both colors; default 200)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/src STOCKFISH=${STOCKFISH:-/usr/local/bin/stockfish} OMP_NUM_THREADS=4
PHASE=$1; ARG=$2; OPEN=${3:-200}
OUT=results/rl/calibration; mkdir -p $OUT
run() {  # model nodes
  python3 -m chesslm.rl.evaluate --ckpt "$1" --opponents elo:1320,elo:1500 --games "$OPEN" --split val --seed 0 \
    --engines 10 --nodes "$2" --out "$OUT/$1_n$2.json" 2>&1 | grep -v -i warning | cut -c1-80,300-700
}
if [ "$PHASE" = sweep ]; then
  for N in $ARG; do echo "== size_m nodes $N"; run size_m "$N"; done
else
  for M in size_xs size_s size_m size_l; do echo "== $M nodes $ARG"; run "$M" "$ARG"; done
fi
echo CALIBRATION DONE

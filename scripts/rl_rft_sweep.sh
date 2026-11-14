#!/bin/bash
# Cheap RFT sweep (DECISIONS G35): offline SFT on the saved round-1 wins at several settings, then a greedy eval of each,
# and of base M and rounds 1, 3 and 4 of the full RFT run. 1200 games per model: 300 val openings x 2 colors x opponents 1320, 1500.
# Usage: bash scripts/rl_rft_sweep.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/src STOCKFISH=${STOCKFISH:-/usr/local/bin/stockfish} OMP_NUM_THREADS=4
R=results/rl/rft_m/round1_rollouts.jsonl
mkdir -p results/rl/rft_sweep
for S in "3e-5 2" "1e-4 2" "3e-4 2" "1e-4 6"; do
  set -- $S
  python3 -m chesslm.rl.rft_offline --ckpt size_m --rollouts $R --tag lr$1_ep$2 --lr $1 --epochs $2
done
ev() {  # $1 = ckpt (name or path), $2 = output tag
  python3 -m chesslm.rl.evaluate --ckpt $1 --mode greedy_legal --games 300 --split val --opponents elo:1320,elo:1500 \
    --engines 10 --seed 0 --out results/rl/rft_sweep/$2.json 2>&1 | grep -v saved | cut -c1-330
}
ev size_m base_m
ev /workspace/ckpt/rft_m/round1.pt rft_m_round1_lr1e-5_ep2
ev /workspace/ckpt/rft_m/round3.pt rft_m_round3
ev /workspace/ckpt/rft_m/round4.pt rft_m_round4
for T in lr3e-5_ep2 lr1e-4_ep2 lr3e-4_ep2 lr1e-4_ep6; do ev /workspace/ckpt/rft_sweep/$T/final.pt $T; done
echo SWEEP DONE

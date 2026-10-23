#!/bin/bash
# Token sweep (DECISIONS G16): finish each saved flat-phase checkpoint with a short decay.
# Copy size_<m>/tok_<T>.pt to tok_<m>_<T>m/last.pt. Resume with max_tokens = 1.25 x the checkpoint tokens,
# so the decay starts at once and lasts 25% of the flat part, as in the full run.
# Run again after a stop: a finished branch is skipped, an unfinished branch resumes from last.pt.
# Usage: bash scripts/token_sweep.sh ["sizes"] ["points"]
cd "$(dirname "$0")/.." || exit 1
source /workspace/venv/bin/activate
SIZES=${1:-"l m"}
POINTS=${2:-"125000000 250000000 500000000"}
declare -A LR=( [xs]=3e-3 [s]=2e-3 [m]=1.5e-3 [l]=1.2e-3 )
for M in $SIZES; do
  for T in $POINTS; do
    NAME="tok_${M}_$((T / 1000000))m"
    SRC="/workspace/ckpt/size_$M/tok_$T.pt"
    DIR="/workspace/ckpt/$NAME"
    if [ -f "$DIR/final.pt" ]; then echo "skip $NAME (done)"; continue; fi
    mkdir -p "$DIR"
    MAX=$(python -c "import torch; print(int(1.25 * torch.load('$SRC', map_location='cpu')['tokens']))")
    if [ ! -f "$DIR/last.pt" ]; then cp "$SRC" "$DIR/last.pt"; fi
    if [ ! -f "$DIR/config.json" ]; then
      python -c "
import json
c = json.load(open('/workspace/ckpt/size_$M/config.json'))
c.update(name='$NAME', max_tokens=$MAX, save_at='', branch_from='size_$M/tok_$T.pt')
json.dump(c, open('$DIR/config.json', 'w'), indent=1)"
    fi
    python -m chesslm.train --model "$M" --name "$NAME" --lr "${LR[$M]}" --seed 0 --max_tokens "$MAX" \
      --decay_frac 0.2 --eval_every 500 --eval_batches 100 --ckpt_every 1000 2>&1 | grep --line-buffered -v Warning
  done
done
echo TOKEN SWEEP DONE

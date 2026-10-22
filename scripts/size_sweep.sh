#!/bin/bash
# Size sweep: train XS, S, M, L on the same data and seed. Rate rule: lr = 3e-3 x 256 / d_model (DECISIONS G11).
# Run again after a stop: a finished size is skipped, an unfinished size resumes from last.pt.
# Usage: bash scripts/size_sweep.sh [tokens] ["sizes"]
cd "$(dirname "$0")/.." || exit 1
source /workspace/venv/bin/activate
TOKENS=${1:-1e9}
SIZES=${2:-"xs s m l"}
declare -A LR=( [xs]=3e-3 [s]=2e-3 [m]=1.5e-3 [l]=1.2e-3 )
for M in $SIZES; do
  NAME="size_$M"
  if [ -f "/workspace/ckpt/$NAME/final.pt" ]; then echo "skip $NAME (done)"; continue; fi
  python -m chesslm.train --model "$M" --name "$NAME" --lr "${LR[$M]}" --seed 0 --max_tokens "$TOKENS" \
    --save_at 1.25e8,2.5e8,5e8 --eval_every 500 --eval_batches 100 --ckpt_every 1000 2>&1 | grep --line-buffered -v Warning
done
echo SWEEP DONE

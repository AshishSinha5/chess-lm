#!/bin/bash
# Seed check: repeat the M and L size-sweep runs with seed 1. Same rates as the size sweep (DECISIONS G11).
# Run again after a stop: a finished run is skipped, an unfinished run resumes from last.pt.
# Usage: bash scripts/seed_check.sh [tokens] ["sizes"]
cd "$(dirname "$0")/.." || exit 1
source /workspace/venv/bin/activate
TOKENS=${1:-1e9}
SIZES=${2:-"m l"}
declare -A LR=( [xs]=3e-3 [s]=2e-3 [m]=1.5e-3 [l]=1.2e-3 )
for M in $SIZES; do
  NAME="seed_${M}_s1"
  if [ -f "/workspace/ckpt/$NAME/final.pt" ]; then echo "skip $NAME (done)"; continue; fi
  python -m chesslm.train --model "$M" --name "$NAME" --lr "${LR[$M]}" --seed 1 --max_tokens "$TOKENS" \
    --eval_every 500 --eval_batches 100 --ckpt_every 1000 2>&1 | grep --line-buffered -v Warning
done
echo SEED CHECK DONE

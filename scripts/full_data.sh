#!/bin/bash
# L check: train L and M on all the data (1.6B target tokens, one pass, no repeats). DECISIONS G24.
# Same settings as the 1B size sweep (G13); only --max_tokens changes. One pass is 1.615B targets.
# Run again after a stop: a finished size is skipped, an unfinished size resumes from last.pt.
# Usage: bash scripts/full_data.sh ["sizes"]
cd "$(dirname "$0")/.." || exit 1
source /workspace/venv/bin/activate
SIZES=${1:-"l m"}
declare -A LR=( [xs]=3e-3 [s]=2e-3 [m]=1.5e-3 [l]=1.2e-3 )
for M in $SIZES; do
  NAME="full_$M"
  if [ -f "/workspace/ckpt/$NAME/final.pt" ]; then echo "skip $NAME (done)"; continue; fi
  python -m chesslm.train --model "$M" --name "$NAME" --lr "${LR[$M]}" --seed 0 --max_tokens 1.6e9 \
    --save_at 1.25e8,2.5e8,5e8,1e9 --eval_every 500 --eval_batches 100 --ckpt_every 1000 2>&1 | grep --line-buffered -v Warning
done
echo FULL DONE

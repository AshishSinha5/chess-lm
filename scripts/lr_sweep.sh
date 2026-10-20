#!/bin/bash
# Learning-rate sweep on XS. One run per rate, 0.1B tokens each, run one after another.
# Usage: bash scripts/lr_sweep.sh [model] [tokens] "rates"
MODEL=${1:-xs}
TOKENS=${2:-1e8}
RATES=${3:-"1e-3 2e-3 3e-3 4.5e-3 6e-3 1e-2 1.5e-2"}
for LR in $RATES; do
  python -m chesslm.train --model "$MODEL" --name "lr_${MODEL}_${LR}" --lr "$LR" \
    --max_tokens "$TOKENS" --eval_every 400 --ckpt_every 100000 --eval_batches 100 2>&1 | grep -v Warning
done
echo SWEEP DONE

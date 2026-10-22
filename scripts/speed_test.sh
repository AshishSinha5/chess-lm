#!/bin/bash
# Speed test: run each model size for a short time. Record tokens/s and peak GPU memory.
# Usage: bash scripts/speed_test.sh "s m l" [tokens]
SIZES=${1:-"s m l"}
TOKENS=${2:-1.5e7}
for M in $SIZES; do
  rm -rf "/workspace/ckpt/speed_$M"
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 2; done > "/tmp/mem_$M.txt" ) &
  SAMPLER=$!
  START=$(date +%s)
  python -m chesslm.train --model "$M" --name "speed_$M" --lr 3e-3 --max_tokens "$TOKENS" \
    --eval_every 100000 --ckpt_every 100000 --log_every 25 > "/tmp/speed_$M.log" 2>&1
  kill $SAMPLER
  echo "$M wall_s=$(( $(date +%s) - START )) peak_mib=$(sort -n /tmp/mem_$M.txt | tail -1)"
done
echo SPEED DONE

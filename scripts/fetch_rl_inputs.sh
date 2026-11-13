#!/bin/bash
# Fetch the RL inputs from the B2 backup of the stage-1 pod: base checkpoints and Stockfish (docs/RL_PLAN.md, W4).
# It uses the rclone remote that the pod config defines. This file holds no keys.
# Run again at any time: rclone skips files that exist with the same size (--size-only).
# Usage: bash scripts/fetch_rl_inputs.sh ["runs"]   (default: "size_xs size_m size_l")
set -euo pipefail
FLAGS="--local-no-set-modtime --size-only"  # the /workspace volume refuses chtimes
SRC=${B2_SRC:-b2:backity-backup/chess-lm-project}
RUNS=${1:-"size_xs size_m size_l"}
mkdir -p /workspace/ckpt /workspace/bin

for R in $RUNS; do
  echo "== $R"
  rclone copy $FLAGS "$SRC/ckpt/$R" "/workspace/ckpt/$R" --include "final.pt" --include "config.json" --include "log.jsonl"
done

echo "== stockfish"
rclone copy $FLAGS "$SRC/bin" /workspace/bin --include "stockfish"
# /workspace is geesefs (S3 FUSE): no chmod, so no file there can run.
# Copy the binary to the container disk. The container disk is wiped on stop, so run this script on each new pod.
install -m 755 /workspace/bin/stockfish /usr/local/bin/stockfish
# Check that the binary runs and print its version line.
printf 'quit\n' | timeout 10 /usr/local/bin/stockfish  # prints one banner line, then exits

ls -la /workspace/ckpt/*/final.pt /usr/local/bin/stockfish
echo "Set STOCKFISH=/usr/local/bin/stockfish for the eval and RL code."
echo FETCH DONE

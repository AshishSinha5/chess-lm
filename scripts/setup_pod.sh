#!/bin/bash
# New-pod setup for stage 2 on a geesefs /workspace (S3 FUSE). See docs/RUNBOOK.md, Setup.
# geesefs allows no chmod and small random reads take about 0.5 s. So:
#   1. fetch checkpoints and Stockfish (Stockfish goes to the container disk),
#   2. install the Python packages on the container disk (the system torch is used),
#   3. read the data splits once, so later random reads come from the page cache (about 1.6 ms).
# Usage: bash scripts/setup_pod.sh [--train]   (--train also warms the 3.3 GB train split, for SFT runs)
set -euo pipefail
cd "$(dirname "$0")/.."

bash scripts/fetch_rl_inputs.sh

echo "== python packages"
REQ=$(mktemp)
grep -v -e '^torch' -e '^#' requirements.txt > "$REQ"
python3 -m pip install -q -r "$REQ" 2>&1 | grep -v -i -e "warning" -e "notice" -e "root user" || true
rm -f "$REQ"
python3 -c "import torch, chess; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'chess', chess.__version__)"

echo "== warm the page cache"
D=/workspace/data/lichess_1800
SPLITS="val test"
[ "${1:-}" = "--train" ] && SPLITS="$SPLITS train"
for S in $SPLITS; do
  T0=$SECONDS; cat "$D/$S"/* > /dev/null; echo "$S: $((SECONDS - T0)) s"
done

echo "CPU quota: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null) (quota/period). Keep OMP_NUM_THREADS and the engine count below it."
echo "Now run: export PYTHONPATH=$PWD/src STOCKFISH=/usr/local/bin/stockfish OMP_NUM_THREADS=4"
echo SETUP DONE

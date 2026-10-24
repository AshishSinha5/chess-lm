#!/bin/bash
# Rate check (DECISIONS G20): train one size for 1B tokens at a second rate, with flat-phase checkpoints.
# Then finish each checkpoint with a short decay, as in the token sweep (G16), to get the full loss-vs-tokens curve.
# Run names: rate_<m>_<lr> (full run), rate_<m>_<lr>_<T>m (branches).
# Run again after a stop: a finished run is skipped, an unfinished run resumes from last.pt.
# Usage: bash scripts/rate_check.sh [model] [lr] ["points"]
cd "$(dirname "$0")/.." || exit 1
source /workspace/venv/bin/activate
M=${1:-l}
LR=${2:-6e-4}
POINTS=${3:-"125000000 250000000 500000000"}
BASE="rate_${M}_${LR}"
if [ -f "/workspace/ckpt/$BASE/final.pt" ]; then echo "skip $BASE (done)"; else
  python -m chesslm.train --model "$M" --name "$BASE" --lr "$LR" --seed 0 --max_tokens 1e9 \
    --save_at 1.25e8,2.5e8,5e8 --eval_every 500 --eval_batches 100 --ckpt_every 1000 2>&1 | grep --line-buffered -v Warning
fi
for T in $POINTS; do
  NAME="${BASE}_$((T / 1000000))m"
  SRC="/workspace/ckpt/$BASE/tok_$T.pt"
  DIR="/workspace/ckpt/$NAME"
  if [ -f "$DIR/final.pt" ]; then echo "skip $NAME (done)"; continue; fi
  if [ ! -f "$SRC" ]; then echo "missing $SRC"; exit 1; fi
  mkdir -p "$DIR"
  MAX=$(python -c "import torch; print(int(1.25 * torch.load('$SRC', map_location='cpu')['tokens']))")
  if [ ! -f "$DIR/last.pt" ]; then cp "$SRC" "$DIR/last.pt"; fi
  if [ ! -f "$DIR/config.json" ]; then
    python -c "
import json
c = json.load(open('/workspace/ckpt/$BASE/config.json'))
c.update(name='$NAME', max_tokens=$MAX, save_at='', branch_from='$BASE/tok_$T.pt')
json.dump(c, open('$DIR/config.json', 'w'), indent=1)"
  fi
  python -m chesslm.train --model "$M" --name "$NAME" --lr "$LR" --seed 0 --max_tokens "$MAX" \
    --decay_frac 0.2 --eval_every 500 --eval_batches 100 --ckpt_every 1000 2>&1 | grep --line-buffered -v Warning
done
echo RATE CHECK DONE

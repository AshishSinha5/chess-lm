# Chess LM

Small GPT-style models trained from scratch on Lichess games. Goals: measure strength vs Stockfish, probe for an internal board representation. Full plan: `PLAN.md` (TBD).

## Status
Planning + infra done. Tokenizers done. Data pipeline next. **No results yet.**

## Setup (RunPod, A100 80GB)
- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (PyTorch 2.8.0), repo + venv under `/workspace`
- Python 3.12.3, python-chess 1.11.2, Stockfish 16 (`/usr/games/stockfish`; `UCI_Elo` min 1320)

```bash
python3 -m venv --system-site-packages /workspace/venv && . /workspace/venv/bin/activate
pip install -e . pytest && pytest -q
```

## Layout
`src/chesslm/{data,tokenizers,model,eval,probe,utils}`, `configs/`, `scripts/`, `tests/`, `results/`, `figures/`

## Done
- UCI move-token and UCI char-level tokenizers, round-trip + legality tests (3 pass)

## Next
- Lichess download, filter, shard; benchmark PGN→UCI games/s/core
- GPT model + training loop; overfit XS on 1k games
- Measure real tokens/s and TFLOPs

## Notes
- Elo will be Stockfish-anchored, not Lichess/FIDE.
- Data: Lichess open database (CC0).

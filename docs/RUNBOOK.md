# Runbook

Working notes for the project: status, setup, data steps, the step list and the new-pod checklist.
Headline results: `README.md`. Dated history: `results/EXPERIMENT_LOG.md`. Reasons: `DECISIONS.md`.

## Status
Data is built (1.64B training tokens). The size sweep (XS, S, M, L at 1B tokens) and the seed check for all four sizes are done. The small gain from M to L is real. The token sweep (all four sizes) and the Stockfish evaluation are done. Strength rises with size, and the gain from M to L is smaller. A rate check on L shows the L rate is not the cause of the small step (G22). Probing on M is done: M builds a board representation that a linear probe can read, beats move bookkeeping late in the game, and uses it to choose moves (G23). Probes on XS and L show the board gets better with size (G27). Probe errors by piece type: M is best on queens, rooks and bishops, where move bookkeeping is worst (G25). The L check is done (G24): at 1.6B tokens L reaches 1.431 and M reaches 1.4823. The L lead is 0.051, against 0.054 at 1B. Data is not the cause of the small M-to-L step. The scaling law (G29) fits loss against size and tokens, and Elo is close to linear in loss. Numbers are in `README.md` and `results/README.md`.

Stage 2 (RL post-training) started 2026-11-13. The plan is `docs/RL_PLAN.md` (approved, cap $100). W4 steps 1 and 2 are done: setup on the new pod, and the base M eval matches stage 1. W4 step 3 is done: the batched rollout engine (3.24 games/s at 10 engines). Next: the legality bandit.

## Setup (RunPod, A100 80GB)
Pod from 2026-11-13 on (stage 2): `/workspace` is a geesefs mount (S3 FUSE). Measured limits:
- `chmod` and setting file times fail. So no file on `/workspace` can run, and a venv on it fails.
- A small random read takes about 470 ms. After one full read of a file, a random read takes about 1.6 ms (page cache).
- Checkpoint save and load work (200 MB in about 4 s). File locks and SQLite work.

Keep data, checkpoints and the repo on `/workspace`. On each new pod:
1. Run `bash scripts/setup_pod.sh`. It runs `scripts/fetch_rl_inputs.sh` (checkpoints from B2, Stockfish to `/usr/local/bin`), installs the Python packages on the container disk, and reads the val and test splits once. Add `--train` before an SFT or training run (3.3 GB).
2. Run `export PYTHONPATH=/workspace/chess-lm/src STOCKFISH=/usr/local/bin/stockfish OMP_NUM_THREADS=4`.
   Caution: `nproc` shows 116, but the cgroup quota (`/sys/fs/cgroup/cpu.max`) is 11.9 CPUs. Without `OMP_NUM_THREADS`, torch starts 58 CPU threads and runs about 15 times slower (the RL tests: 95 s against 6 s). Keep the Stockfish engine count at 10: at 11 the container was already throttled in 20% of the quota periods.
3. The stage-1 scripts `source /workspace/venv/bin/activate`. On this pod, call `python3 -m ...` directly instead.

Caution: geesefs can upload a new file a short time after the write. Wait about a minute after the last checkpoint save before you stop the pod. Copy RL results to B2 at the end of each run.

Stage-1 pod (until 2026-11-07):
- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (PyTorch 2.8.0).
- Python 3.12.3, python-chess 1.11.2, Stockfish 19 (`/workspace/bin/stockfish`, `UCI_Elo` minimum 1320).
- Keep the repo and venv in `/workspace`. The container disk is wiped on stop.

```bash
python3 -m venv --system-site-packages /workspace/venv && . /workspace/venv/bin/activate
pip install -e . pytest && pytest -q
```

## Done
- UCI move-token and UCI character tokenizers. 3 round-trip tests pass.
- Streaming reader, header filter and PGN-to-UCI replay (`src/chesslm/data/`, `scripts/bench_pgn.py`).
- Fixed move vocabulary: 1,971 tokens (all possible UCI moves plus 3 special tokens). No data pass needed to build it.
- Shard writer `src/chesslm/data/shards.py`. Output is uint16 `tokens.bin` and `offsets.npy` per split.

- GPT model (RoPE), loader and training loop: `model/gpt.py`, `data/dataset.py`, `train.py`. 8 tests pass.
- Parameter counts: XS 5.2M, S 11.4M, M 26.2M, L 50.4M (vocab 1,971).
- Overfit gate passed: XS on 1,000 games reaches 0.094 loss per move. The entropy floor is 0.0927.
- Speed and memory per size: `results/EXPERIMENT_LOG.md`.
- Evaluation harness against Stockfish: `src/chesslm/eval/play.py`.
- Probe code (probes per layer, 5 controls, intervention): `src/chesslm/probe/`. 4 tests pass.

## Measurements
Month 2026-09 (29 GB), first 30k games only. Filter: both players 1800+, 3+0 or slower, normal end.

| Item | Value |
|---|---|
| Games kept | 12.0% (early-month slice, may be biased toward bullet) |
| Average length | 72 plies |
| Replay, 1 worker | 475 games/s (34k tokens/s) |
| Replay, 8 workers | 3,509 games/s (253k tokens/s) |
| 1B tokens, 8 workers | about 1.1 hours of replay |
| Stream and filter | about 21k games/s, one process |

## Steps
1. Shard build for 2026-09 and 2026-08 (both players 1800+). Output: `/workspace/data/lichess_1800` (1.64B train tokens). Raw files deleted.
2. Learning-rate sweep on XS (`bash scripts/lr_sweep.sh`). 2e-3 to 6e-3 give equal loss. Choice: 3e-3 (G9).
3. Speed test (G10) and L learning-rate check (G11): 3e-3 fails on L. Rule: lr = 3e-3 x 256 / d_model.
4. Size sweep XS, S, M, L at 1B tokens. Final loss per move 1.789, 1.643, 1.519, 1.465 (G13).
5. Seed check for all four sizes (G15). The seed spread is 0.002 to 0.006, so the small step from M to L is real.
6. Token sweep for all four sizes (G16 to G18). The small step from M to L is there at every budget.
7. Evaluation against Stockfish (G14, G19). Elo on the nominal scale: XS 1182, S 1296, M 1410, L 1479. Strict rule (an illegal move loses, G28): about 934, 1113, 1300, 1376.
8. Rate check on L at 6e-4 (G20, G22). 1.2e-3 is at or near the best rate at 1B tokens.
9. Probing on M with all 5 controls and the intervention (G21, G23), and the variance-matched control (G26).
10. L check (G24). L 1.431 and M 1.4823 at 1.6B tokens. The L lead is 0.051 (1B: 0.054), so data is not the cause.
11. Probes on XS and L (G27) and probe errors by piece type (G25). The write-up is `docs/PROBING.md`.
12. Scaling law (G29): loss = E + A/N^0.32 + B/D^0.66 fits the 16 sweep runs (RMSE 0.007). It predicts both 1.6B-token runs to within 0.013. Elo is close to linear in loss.
13. Blog post draft: `docs/BLOG.md`.

Stage 2 (RL, plan `docs/RL_PLAN.md`):
14. W4 step 1: new-pod setup on geesefs (`scripts/setup_pod.sh`, `scripts/fetch_rl_inputs.sh`). Base checkpoints and Stockfish 19 from B2.
15. W4 step 2: base M check. Lenient 1417 (1388 to 1449), strict 1333 (1297 to 1368). Both inside the stage-1 intervals: PASS. Next: batched rollout engine and speed test.
16. W4 step 3: batched rollout engine (`src/chesslm/rl/`), exit tests 1 to 3, speed test. 3.24 games/s with 10 engines (2.09 before the async loop). Base M on the new code: 1461 (1444 to 1480) lenient, 1343 (1323 to 1362) strict, 1200 games. It reads about 25 Elo above the stage-1 code on this pod, because Stockfish searches fewer nodes when the CPU is busy. Compare Elo only within one pod and engine count (log step 26). Next: legality bandit.

## Month-wide count (2026-09)
Full pass over 89.6M games. Tokens are estimated from the last move number. Other filters are as above.

| Both players at least | Games | Tokens |
|---|---|---|
| 1600 | 20.1M | 1.33B |
| 1800 | 11.0M | 0.74B |
| 2000 | 4.2M | 0.29B |

One month at 1800 gives 0.74B tokens, not 1B. Script: `scripts/count_month.py` (about 36 minutes).

## Data download
```bash
python scripts/fetch_month.py 2026-09 /workspace/raw 16   # parallel, resumable, about 13 min
python -m chesslm.data.shards /workspace/data/lichess_1800 24 2026-09 2026-08
```
Check progress: `tail -1 /workspace/shards.log`. Done when it shows `DONE` and `meta.json` exists. Raw files are about 29 GB each.

## New pod checklist
The container disk is wiped on stop. `/workspace` stays.
1. Update `~/.ssh/config` (new IP and port).
2. `apt-get install -y stockfish rsync`
3. `git config --global user.name/user.email` and `credential.helper 'store --file /workspace/.git-credentials'`
4. Install Claude Code and log in. Then run `bash scripts/restore_claude_state.sh` to link memory and history.
5. Before you stop a pod, run `bash scripts/sync_claude_state.sh`.

## Train
```bash
python -m chesslm.train --model xs --name xs_run --data /workspace/data/lichess_1800 --max_tokens 1e9
```
Checkpoints: `/workspace/ckpt/NAME`. The run resumes if you start it again. W&B is offline (`wandb sync` later).

## Notes
- Decision: two months at the 1800 filter. One month gives only 0.74B tokens.
- Split: by CRC32 of the game ID. 98% train, 1% val, 1% test.
- Elo is anchored to Stockfish. It is not a Lichess or FIDE rating.
- Data: Lichess open database, <https://database.lichess.org> (CC0).
- Blog post: `docs/BLOG.md`.
- Tokenizer: the move token only.
- Licence: Apache 2.0 for the code and the weights (`LICENSE`).


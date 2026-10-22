# Experiment log

A dated list of every step. Each entry gives the action, the run names, and the result.
The reasons for each choice are in `DECISIONS.md`. The numbers for each run are in `runs.csv`.
Add a new entry after each stage. Dates are UTC.

```mermaid
flowchart LR
  A[1 Setup] --> B[2 Tokenizers]
  B --> C[3 Data]
  C --> D[4 Overfit gate]
  D --> E[5 XS learning rate]
  E --> F[6 Speed test]
  F --> G[7 L learning rate]
  G --> H[8 Size sweep]
  H --> I[9 Token sweep]
  I --> J[10 Evaluation]
  J --> K[11 Probing]
  style I fill:#ddd
  style J fill:#ddd
  style K fill:#ddd
```
Steps 1 to 7 are done. Step 8 is running. Grey steps are not started.

## Done

### 1. Setup (2026-10-18)
- RunPod pod with one A100 PCIe 80GB. PyTorch 2.8.0, Python 3.12.3, python-chess 1.11.2, Stockfish 16.
- Repo on GitHub. Venv, data and checkpoints live on the volume `/workspace`.
- Pod restarted on 2026-10-22. The volume kept the data, repo and venv. Stockfish and the git settings were lost and set again.

### 2. Tokenizers (2026-10-18)
- UCI move token (1 token per move) and UCI character token. Round-trip tests pass.
- Fixed vocabulary: 1,971 tokens (3 special tokens and every possible UCI move). See DECISIONS B.

### 3. Data (2026-10-18)
- Speed tests: stream and filter about 21k games/s with one process. Replay with 8 workers about 3,509 games/s.
- Month count, 2026-09: 89.6M games. Both players at least 1800 gives 11.0M games and 0.74B tokens (`data/month_counts.csv`). One month is not enough for 1B tokens, so we use two months.
- Download: 16 parallel connections, about 13 minutes per month (DECISIONS G8).
- Shard build, months 2026-09 and 2026-08. Filter: both players 1800+, 3+0 or slower, normal end, 10 to 256 plies.

| Item | Value |
|---|---|
| Games read | 181.5M |
| Games kept | 22.8M |
| Games that failed to parse | 72k (0.3% of kept) |
| Train | 22.37M games, 1.637B tokens |
| Validation | 228k games, 16.7M tokens |
| Test | 228k games, 16.7M tokens |

- Full numbers: `data/dataset.json`. The raw files (56 GB) were deleted on 2026-10-20.
- Git problem found on 2026-10-20: the line `data/` in `.gitignore` also hid `src/chesslm/data/`. Fixed in commit 30b465f.

### 4. Overfit gate (2026-10-18)
- Run `overfit_xs`: XS on 1,000 games, 40M tokens.
- Result: loss per move 0.0939. The entropy floor is 0.0927. The gate passes. DECISIONS G7.

### 5. XS learning rate (2026-10-20)
- Runs `lr_xs_*`: 7 rates from 1e-3 to 1.5e-2, 0.1B tokens each, seed 0. Then seed 1 for 2e-3, 3e-3 and 4.5e-3 (`*_s1`).
- Command: `bash scripts/lr_sweep.sh` and `bash scripts/lr_sweep.sh xs 1e8 "2e-3 3e-3 4.5e-3" 1`.
- Result: 2e-3 to 6e-3 give the same loss within noise (about 2.31 to 2.38). 1e-2 and 1.5e-2 fail (3.51 and 3.60).
- Choice for XS: 3e-3. DECISIONS G9.

### 6. Speed test (2026-10-22)
- Runs `speed_s`, `speed_m`, `speed_l`: 15M tokens each. Command: `bash scripts/speed_test.sh`.
- Steady speed was measured in the size sweep (`size_*` runs).

| Model | Tokens/s | Peak GPU memory | Time for 1B tokens |
|---|---|---|---|
| XS | about 1.9M | not measured | about 9 min |
| S | about 1.16M | 9.9 GB | about 15 min |
| M | about 0.61M | 14.8 GB | about 27 min |
| L | about 0.35M | 21.3 GB | about 48 min |

- DECISIONS G10.

### 7. L learning rate (2026-10-22)
- Runs `lr_l_1e-3`, `lr_l_2e-3`, `lr_l_3e-3`: 0.1B tokens each, seed 0.
- Result: 1.959, 1.945 and 3.332. At 3e-3 the loss rose after step 600 and the gradient norm reached 5.0.
- Rule for all sizes: lr = 3e-3 x 256 / d_model. That gives XS 3e-3, S 2e-3, M 1.5e-3, L 1.2e-3. DECISIONS G11.

## Running

### 8. Size sweep (started 2026-10-22)
- Runs `size_xs`, `size_s`, `size_m`, `size_l`: 1B tokens each, seed 0, same data order. Command: `bash scripts/size_sweep.sh`.
- Extra checkpoints at 0.125B, 0.25B and 0.5B tokens, for the token sweep.
- XS final loss 1.789. S final loss 1.643. M and L: see `runs.csv`.

## Planned
9. **Token sweep on L.** Short decay runs from the saved checkpoints at 0.125B, 0.25B and 0.5B tokens. Output: `token_sweep.csv`.
10. **Evaluation.** Legal-move rate and matches against Stockfish. Elo with confidence intervals. Output: `eval/`.
11. **Probing on M.** Linear probes per layer, with controls and one intervention. Output: `probe/`.
12. **Figures and write-up.** `scripts/make_figures.py` makes `figures/`.

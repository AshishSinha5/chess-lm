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
  H --> S[9 Seed check]
  S --> I[10 Token sweep L, M]
  I --> T[11 Token sweep XS, S]
  T --> J[12 Evaluation]
  J --> R[13 Rate check L]
  R --> K[14 Probing M]
```
Steps 1 to 14 are done. Grey steps are not started.

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

### 8. Size sweep (2026-10-22)
- Runs `size_xs`, `size_s`, `size_m`, `size_l`: 1B tokens each, seed 0, same data order. Command: `bash scripts/size_sweep.sh`.
- Rates: XS 3e-3, S 2e-3, M 1.5e-3, L 1.2e-3 (rule from step 7). Extra checkpoints at 0.125B, 0.25B and 0.5B tokens.
- No errors. Total training time about 1.7 hours.

| Model | Params | Final val loss per move | Change from the size below | Tokens/s | Share of A100 peak |
|---|---|---|---|---|---|
| XS | 5.2M | 1.789 | | 1.89M | 19% |
| S | 11.4M | 1.643 | -0.146 | 1.16M | 25% |
| M | 26.2M | 1.519 | -0.124 | 0.61M | 31% |
| L | 50.4M | **1.465** | -0.054 | 0.35M | 34% |

- Loss falls with size, but the step from M to L is less than half of the earlier steps.
- During the flat phase the loss is higher than at the end. L at about 0.5B tokens: 1.650. L at 1B tokens, after the fall: 1.465.
- Seed spread was measured only at 0.1B tokens on XS (up to 0.045). The sweep used one seed.
- DECISIONS G13.

### 9. Seed check (2026-10-23)
- Runs `seed_m_s1` and `seed_l_s1`: 1B tokens each, seed 1. Same rates as the size sweep. Command: `bash scripts/seed_check.sh`.
- No errors. Training time about 75 minutes.

| Model | Seed 0 | Seed 1 | Difference | Step from the size below (seed 0 / seed 1) |
|---|---|---|---|---|
| M | 1.5192 | 1.5173 | 0.0019 | |
| L | 1.4652 | 1.4622 | 0.0030 | -0.0540 / -0.0551 |

- The seed spread is about 0.003. The M-to-L step is less than half of the earlier steps (-0.146, -0.124). The smaller step is real for this setup.
- XS and S still have one seed at 1B tokens.
- The check does not give the cause. The token sweep tests the data limit.
- Other changes on this date: Stockfish 19 stored in `/workspace/bin/stockfish` (the old pod had Stockfish 16, lost on restart). Evaluation harness `src/chesslm/eval/play.py` and `scripts/summarize_eval.py` written and tested on a short run. No full evaluation yet. DECISIONS G14.
- DECISIONS G15 and G16.

### 10. Token sweep (2026-10-23)
- Runs `tok_l_125m`, `tok_l_250m`, `tok_l_500m`, `tok_m_125m`, `tok_m_250m`, `tok_m_500m`. Command: `bash scripts/token_sweep.sh`. Design: DECISIONS G16.
- Each run resumes from the size-sweep checkpoint at T tokens and decays over 0.25 x T more tokens. Seed 0. No errors. Total time about 34 minutes (the estimate was 17; each branch also pays for compile time and checkpoint loading).

| Total tokens | L | M | M minus L |
|---|---|---|---|
| 0.156B | 1.785 | 1.861 | 0.076 |
| 0.313B | 1.616 | 1.682 | 0.067 |
| 0.625B | 1.512 | 1.568 | 0.056 |
| 1.0B (full run) | 1.465 | 1.519 | 0.054 |

- Both models still gain at 1B tokens. L gains slightly less than M (0.320 and 0.342 over the range). The lead of L over M shrinks from 0.076 to 0.054.
- The result fits a weak data limit on L but does not prove one. S and XS branches would show if the small step also appears at small budgets.
- Output: `results/token_sweep.csv`, written by `scripts/summarize_tokens.py`.
- DECISIONS G17.

### 11. Token sweep for XS and S (2026-10-23)
- Runs `tok_xs_125m`, `tok_xs_250m`, `tok_xs_500m`, `tok_s_125m`, `tok_s_250m`, `tok_s_500m`. Command: `bash scripts/token_sweep.sh "xs s"`. Design: DECISIONS G16. Seed 0. No errors.
- Time about 30 minutes (it shared the GPU with step 12).

| Total tokens | XS | S | M | L | XS to S | S to M | M to L |
|---|---|---|---|---|---|---|---|
| 0.156B | 2.111 | 1.995 | 1.861 | 1.785 | 0.117 | 0.134 | 0.076 |
| 0.313B | 1.949 | 1.812 | 1.682 | 1.616 | 0.137 | 0.129 | 0.067 |
| 0.625B | 1.834 | 1.694 | 1.568 | 1.512 | 0.140 | 0.126 | 0.056 |
| 1.0B | 1.789 | 1.643 | 1.519 | 1.465 | 0.146 | 0.123 | 0.054 |

- The step from M to L is the smallest step at every budget. It is 0.57 of the S-to-M step at 0.156B and 0.44 at 1.0B.
- The cause of the small step is still open. A data limit explains only the part that grows with tokens.
- Output: `results/token_sweep.csv` (16 points). DECISIONS G18.

### 12. Evaluation against Stockfish (2026-10-23)
- Models `size_xs`, `size_s`, `size_m`, `size_l`. Command: `python -m chesslm.eval.play --ckpt <model> --legal_games 5000 --opponents elo:1320,elo:1500 --games 100 --workers 16`. Stockfish 19, 0.05 seconds per move. Method: G14.
- 400 games per model, 1600 in total. Time about 9 minutes.

| Model | Legal-move rate | Same move as human | Score vs 1320 | Score vs 1500 | Elo (95% interval) |
|---|---|---|---|---|---|
| XS | 98.4% | 46.2% | 0.30 | 0.15 | 1182 (1144 to 1219) |
| S | 99.4% | 49.1% | 0.45 | 0.26 | 1296 (1266 to 1329) |
| M | 99.6% | 50.8% | 0.62 | 0.38 | 1410 (1378 to 1444) |
| L | 99.5% | 52.1% | 0.71 | 0.47 | 1479 (1443 to 1510) |

- Elo steps: +114, +114, +69. The M-to-L step is smaller, as in the loss.
- The Elo is on the nominal Stockfish scale, not a human rating (G14, G19).
- Output: `results/eval/` (`legal_moves.csv`, `stockfish_games.csv`, `elo.csv`, and one `.json` per model). DECISIONS G19.

### 13. Rate check on L (2026-10-24)
- Command: `bash scripts/rate_check.sh l 6e-4`. Runs `rate_l_6e-4` (1B tokens, flat-phase checkpoints at 0.125B, 0.25B, 0.5B) and the branches `rate_l_6e-4_125m`, `_250m`, `_500m` (short decay, G16). Seed 0.
- Time: about 87 minutes (11:46 to 13:13). My probe tests ran on the same machine at the start and slowed training to 85-175k tokens/s for about 25 minutes.

| Total tokens | L, lr 1.2e-3 | L, lr 6e-4 | 6e-4 minus 1.2e-3 |
|---|---|---|---|
| 0.156B | 1.785 | 1.877 | +0.092 |
| 0.313B | 1.616 | 1.664 | +0.048 |
| 0.625B | 1.512 | 1.532 | +0.020 |
| 1.0B | 1.465 | 1.473 | +0.008 |

- The lower rate is worse at every budget. The gap shrinks as tokens grow.
- In the flat phase, 6e-4 was ahead after about 0.55B tokens (by 0.006 at 0.62B). The decay reversed this.
- Output: `results/rate_check.csv` (8 points). DECISIONS G22.

### 14. Probing M (2026-10-24)
- Commands: `python -m chesslm.probe.run layers --ckpt size_m --mlp`, then `python -m chesslm.probe.run intervene --ckpt size_m`. Design: G21.
- 100k positions from 20k test-split games, split 80/20 by game. Time: about 10 minutes for `layers`, 5 minutes for `intervene`.
- Probes, balanced accuracy on non-empty squares (whole-board match in brackets):

| Layer | M, linear | M, MLP | Random-init M, linear |
|---|---|---|---|
| 0 | 0.401 (1.6%) | 0.404 | 0.382 |
| 1 | 0.846 (31.6%) | 0.836 | 0.605 |
| 4 | 0.907 (41.8%) | 0.907 | 0.617 |
| 8 | 0.952 (53.8%) | 0.963 (63.7%) | 0.614 |

- Other controls: majority class 0.141, shuffled labels about 0.155, move-history baseline 0.943 (57.9%). Player-relative target at layer 8: 0.933.
- Whole-board match by ply, layer 8 against the move-history baseline: 1-10: M 93% / MLP 99% / history 100%. 41-60: 25% / 36% / 16%. 81+: 19% / 31% / 10%.
- Intervention at layers 2, 4, 6; ranks 32 and 128; seeds 0, 1, 2. At layer 4, rank 32: loss +0.293 for probe directions against +0.038 for random directions. Legal-move rate 97.8% against 99.4% (99.5% with no change).
- Output: `results/probe/probe_by_layer.csv`, `probe_by_ply.csv`, `controls.csv`, `intervention.csv`. DECISIONS G23.

### 15. Probe errors by piece type, M (2026-10-29)
- Command: `python -m chesslm.probe.errors --ckpt size_m --layer 8`. CPU only, about 2 minutes. It uses the cache of step 14.
- It refits the linear probe at layer 8 and the move-history baseline. The refit gives the same balanced accuracy as step 14 (0.9522, 0.9429).
- Recall by piece type (white / black), linear probe against move history:

| Piece | Linear probe | Move history |
|---|---|---|
| Pawn | 0.989 / 0.989 | 0.994 / 0.995 |
| Knight | 0.948 / 0.947 | 0.984 / 0.987 |
| Bishop | 0.946 / 0.946 | 0.938 / 0.940 |
| Rook | 0.948 / 0.951 | 0.939 / 0.940 |
| Queen | 0.925 / 0.927 | 0.861 / 0.869 |
| King | 0.952 / 0.958 | 0.936 / 0.935 |

- Most common probe error: the right piece type with the wrong colour (knight 2.3 to 2.6%, bishop and queen about 2%).
- Queen errors go next to rook (1.6%) and bishop (1.2%). King errors are mostly "empty" (1.2 to 1.6%).
- Ghost pieces on empty squares: 0.19% where no move touched the square, 0.62% where a piece left it (move history: 0.05% and 0.24%).
- Output: `results/probe/errors_by_piece.csv`, `ghosts.csv`, `confusion.csv`. DECISIONS G25.

### 16. Probe figures (2026-10-29)
- Command: `python scripts/make_figures.py`. It reads only CSV files from `results/`, so it needs no GPU. New dependency: matplotlib.
- Output: `results/figures/probe_by_layer.png`, `probe_by_ply.png`, `intervention.png`, `piece_recall.png`. All 4 are in `docs/PROBING.md`.
- Example boards (2026-11-07): `python -m chesslm.probe.example` refits the layer-8 linear probe on the CPU (check: bal_acc 0.9522, as in `probe_by_layer.csv`). It picks the most typical test game and writes `results/probe/example_game.csv` and `example_game_moves.csv`. `make_figures.py` draws `example_boards.png`. The game has 0, 2 and 2 wrong squares at plies 5, 25 and 39. Used in `docs/PROBING.md` section 3.1.

### 17. L check, part 1: L at 1.6B tokens (2026-10-29)
- Command: `bash scripts/full_data.sh`. Run: `full_l`. Design: G24. Time: 17:25 to 19:11 UTC, about 342k tokens/s.
- Result: val loss per move 1.431 at 1.6B tokens. At 1B tokens (`size_l`) it was 1.465, so the extra 0.6B tokens give -0.034.
- `full_m` is not run yet. The queue was stopped after L because the pod closes. Without M, the L lead at 1.6B is unknown.
- Extra: flat-phase checkpoint `tok_1000000000.pt` (not decayed, not comparable with the 1B runs).
- Note: the test suite ran on the CPU during this run and slowed it to about 230k tokens/s for 10 minutes.

### 18. L check, part 2: M at 1.6B tokens (2026-11-07)
- Command: `bash scripts/full_data.sh m`. Run: `full_m`. Design: G24. About 45 minutes, about 603k tokens/s. A first start was stopped by hand after 2 minutes and restarted clean.
- Result: val loss per move 1.4823 at 1.6B tokens. At 1B (`size_m`) it was 1.519, so the extra tokens give -0.037.
- L lead over M: 0.051 at 1.6B (1.4823 against 1.431), 0.054 at 1B. The change is -0.003, inside the seed spread. Data is not the cause of the small M-to-L step (G24).

### 19. Release: weights on Hugging Face, local GUI, GIFs (2026-11-07)
- Weights: `python scripts/export_hf.py --upload`. It writes `model.safetensors` and `config.json` for the 1B size-sweep runs (`size_xs`, `size_s`, `size_m`, `size_l`, seed 0) to `sinha/chess-lm`. Weights only, 22 to 195 MB. The script checks that the logits match the checkpoint. Card: `docs/HF_MODEL_CARD.md`. Licence: Apache 2.0 (`LICENSE`).
- GUI: `scripts/play_gui.py`. A local web page, standard library only. Tested with XS over HTTP: 3 full games, all ended correctly. The browser page is not tested.
- GIFs: `python scripts/make_gifs.py`. Each model plays white for 8 moves against Stockfish at UCI_Elo 1320, on CPU. Output: `docs/gifs/play_{xs,s,m,l}.gif` (loop, no controls) and `games.json`. A rerun can differ because Stockfish is not seeded.

### 20. Seed check for XS and S (2026-11-07)
- Command: `bash scripts/seed_check.sh 1e9 "xs s"`. Runs: `seed_xs_s1`, `seed_s_s1` (seed 1, same rates and data as the size sweep).
- Result (val loss per move): XS 1.783 (seed 0: 1.789, difference 0.006). S 1.6379 (seed 0: 1.6427, difference 0.0048). The M and L differences were 0.0019 and 0.0030 (G15).
- The seed spread is 0.002 to 0.006 and is larger for the small models. It is far below the step between sizes (0.054 to 0.146).

### 21. Variance-matched intervention on M (2026-11-07)
- Command: `python -m chesslm.probe.run intervene --ckpt size_m --layers 2,4,6 --seeds 0,1,2 --ranks 32,128`. Design: G26. About 25 minutes on the GPU. New code: `matched_directions` in `probe/run.py`, new column `var_frac`, new kind `random_matched`.
- The board directions carry 8% to 10% of the hidden-state variance, plain random directions 6.4%.
- Loss increase at rank 32 (blocks 2, 4, 6): board +0.149, +0.293, +0.312. Random +0.022, +0.038, +0.050. Random with the same variance +0.040, +0.044, +0.063.
- The board directions hurt 3.7, 6.7 and 5.0 times more than the matched random directions. The "M uses the board" result stands (G26, `docs/PROBING.md` section 7).
- The probe and random arms match the old numbers (largest loss difference 0.0002).

### 22. Probing on L and XS (2026-11-07)
- Commands: `python -m chesslm.probe.run layers --ckpt size_l --mlp` and `--ckpt size_xs --mlp` (about 4 minutes each, GPU), then `python -m chesslm.probe.errors --ckpt size_l --layer 9 --device cuda` and `--ckpt size_xs --layer 6`. Design: G21. Result: G27.
- Linear probe, best layer: XS 0.866 (layer 6), M 0.952 (layer 8), L 0.968 (layer 9) balanced accuracy. MLP probe: 0.929, 0.963, 0.973. Whole board right: 23.6%, 53.8%, 65.1%.
- Controls (best layer): random-init 0.545, 0.619, 0.640. Shuffled labels 0.148, 0.158, 0.163. Move history 0.943.
- Ghost pieces on vacated squares: XS 2.7%, M 0.62%, L 0.32% (move history 0.24%).
- New figure: `results/figures/board_by_size.png` (`make_figures.py`, `board_by_size`).

### 23. Strict and lenient illegal-move results (2026-11-07)
- Command: `python -m chesslm.eval.play --ckpt size_X ... --out results/eval/strict_rerun/size_X.json` for XS, S, M, L (about 2 minutes each, 16 workers). Summary: `python scripts/summarize_strict.py` (also in `build_results.sh`). Design: G28.
- Elo, lenient (rerun): 1184, 1289, 1394, 1454. Elo, strict: 934, 1113, 1300, 1376. Gap: 250, 176, 94, 78.
- Games with at least one illegal top move (of 400): 333, 273, 176, 149.
- Output: `results/eval/strict_vs_lenient.csv`. The published `elo.csv` is unchanged.

## Planned
24. **Project write-up.** The other figures in `scripts/make_figures.py` (loss against size, token sweep, Elo), blog post, credibility checklist gaps (strict and lenient done, G28). Optional: illegal-move error analysis.
25. **Close out.** Fresh-clone test, final commit, sync of the Claude state.

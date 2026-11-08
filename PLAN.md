# Plan

Goal: train 4 GPT-style models (about 5M to 50M parameters) on Lichess games. Measure their strength against Stockfish. Probe them for board state.

Out of scope: beating strong engines, RL, search, engine evals as input, more than 2B tokens or 50M parameters, production serving.

## Infra
- RunPod on-demand, 1x A100 80GB (about $1.80/hr), region US-MO-1.
- Volume at `/workspace`. Put everything that must last there. Run training in `tmux`. Stop the pod when idle.
- Never put keys, IPs or RunPod IDs in the repo.

## Milestones
```
 W0 setup ──> W1 data+tokenizers+XS ──> W2 scaling+eval ──> W3 probes+demo+blog
```

**Weekend 1: data, tokenizers, first model**
- Benchmark PGN-to-UCI speed. Decide the token budget.
- Build train/val/test shards (split by game hash).
- Pass tokenizer round-trip tests. Overfit XS on 1k games.
- Measure real tokens/s and TFLOPs. Update the cost table.

**Weekend 2: scaling and evaluation**
- Tokenizer ablation: XS and S, 2 or 3 schemes, same game count.
- Size sweep XS/S/M/L at 1B tokens. Token sweep on L at 0.125/0.25/0.5/1B.
- Legal-move metrics, Stockfish matches, Elo with confidence intervals.
- Error analysis: illegal moves by ply, piece and phase.

**Weekend 3: probes, demo, write-up**
- Cache activations for M. Train linear probes per layer.
- Run controls. Run at least 1 intervention with a random control.
- Write `docs/PROBING.md`: method for a new reader, the 3 results (present, beyond bookkeeping, used), limits, reading list. Add errors by piece type and a variance-matched intervention control.
- Probe L and XS with the same method and controls (board quality by size).
- L check: L and M on all the data, 1.6B tokens (DECISIONS G24).
- Export the demo. Write the blog post (6+ plots). Test a fresh clone.

## Data
- Source: Lichess monthly dumps, standard rated games. Stream them. Do not store them.
- Filters (to tune): both players 1800 or more, time control 3+0 or slower, normal termination, 10 to 256 plies.
- Format: `<bos> m1 ... mN <eos>`, stored as uint16 (move token) or uint8 (character) memmaps. Split by game ID: 98/1/1.
- Estimate: 1B move tokens is about 12 to 14M games and 2GB. Measure this.
- Risk: python-chess replay uses CPU. Use multiprocessing.

## Tokenizers
| Scheme | Vocab | Tokens per move | Context |
|---|---|---|---|
| A. UCI move token (default) | 2-4k | 1 | 256 |
| B. UCI character | about 30 | 4-5 | about 1.3k |
| C. Split square (optional) | about 70 | 2 | about 512 |

Compare on validation loss per move, legal-move rate and a short Stockfish match. Match on number of games, not tokens.

## Model and training
- Decoder-only, pre-norm, GELU MLP (4x), tied embeddings, bf16, `torch.compile`, SDPA attention. Head size 64.

| Name | d_model | Layers | Heads | Parameters | Context |
|---|---|---|---|---|---|
| XS | 256 | 6 | 4 | about 5.5M | 256 |
| S | 384 | 6 | 6 | about 12M | 256 |
| M | 512 | 8 | 8 | about 27M | 256 |
| L | 640 | 10 | 10 | about 51M | 256 |

- AdamW (0.9, 0.95), weight decay 0.1, clip 1.0. Warmup, then constant, then decay (WSD), so mid-run checkpoints stay valid.
- About 64k tokens per step. One game per sequence, padding masked.
- Compute estimate: 11 to 13 A100 hours, $20 to $25. This assumes 40 TFLOPs. Measure it in Weekend 1.
- Budget cap: $50 (assumed, confirm).

## Evaluation
- **Legal moves:** 10k test positions by phase. Report greedy and sampled rates, and a game-survival curve.
- **Stockfish:** pin the version. 1 thread, 50 ms per move. Opponents: random mover, Skill Level 0-3, `UCI_Elo` 1320 to 2100. 100 games per pair, alternate colors. Illegal move is a loss (strict). Also report resample 5 times.
- **Elo:** maximum likelihood over opponents scoring 5% to 95%. Bootstrap 95% intervals (1,000 resamples). State it is Stockfish-anchored.

## Probing
- Models: M (main), then L and XS. Data: 100k positions from 20k held-out games. Split by game.
- Targets: piece on each square (13 classes), player-relative variant. Side to move is tied to ply parity, so do not headline it.
- Layers: embedding output and after each block, at the last token of each move.
- Probes: linear (main), small MLP (comparison).
- Metrics: balanced accuracy on non-empty squares, whole-board match, accuracy by layer.
- Controls (all required): random-init model, shuffled labels, move-history baseline, majority class, embedding-only probe.
- Intervention: remove the probe subspace at layer l. Compare with a random subspace of the same rank. Use 3+ layers and 3+ seeds.
- Cite prior probing work for chess and Othello GPTs. This is a replication-style study.

## Risks
| Risk | Fallback |
|---|---|
| Budget overrun | Cheaper GPU for small runs, drop stretch runs, stop the pod when idle |
| Slow data prep | Smaller token budget, more workers |
| Model scores 0 against all Stockfish levels | Add weaker anchors, report legal rate and loss |
| Elo calibration | State the caveat, show intervals |
| Probes show only trivial signal | Publish the negative result with controls |
| IP changes on restart | Update `~/.ssh/config`, keep all state in `/workspace` |

## Logging
Every result goes into git as a small text file. Checkpoints and W&B files stay on the volume.
- Source: `/workspace/ckpt/RUN/log.jsonl` and `config.json`. Each run writes both.
- `scripts/summarize_runs.py` reads them and writes `results/runs.csv`, `results/curves/`, `results/configs/` and the table in `results/README.md`.
- `bash scripts/build_results.sh` rebuilds all of `results/`. Run it after each stage. Then commit `results/`.
- `results/EXPERIMENT_LOG.md` is the dated list of steps. `DECISIONS.md` holds the reasons.
- **After every step** (build, run, sweep, evaluation, probe) do the logging at once and without being asked. The 6 steps are in `CLAUDE.md`, section Log results after every step. Commit and push only when the user asks.

| Stage | Files in `results/` |
|---|---|
| Data | `data/dataset.json`, `data/month_counts.csv` |
| Training runs (all sweeps) | `runs.csv`, `curves/`, `configs/` |
| Token sweep | `token_sweep.csv` |
| Scaling law | `scaling_fit.csv`, `scaling_points.csv`, made by `scripts/fit_scaling.py` |
| Rate check | `rate_check.csv` |
| Evaluation | `eval/legal_moves.csv`, `eval/stockfish_games.csv`, `eval/elo.csv` |
| Probing | `probe/probe_by_layer.csv`, `probe/probe_by_ply.csv`, `probe/controls.csv`, `probe/intervention.csv`, `probe/errors_by_piece.csv`, `probe/ghosts.csv`, `probe/confusion.csv`. Made by `python -m chesslm.probe.run` (needs the GPU) and `chesslm.probe.errors` (CPU), not by `build_results.sh` |
| Figures | `results/figures/*.png`, made by `scripts/make_figures.py` |

## Open questions
- Budget cap ($50?). Final volume size. Rating filter (1800?).
- Logging: decided. W&B offline plus `results/` files in git. Position encoding: RoPE or learned.
- Conditioning tokens: none in v1.

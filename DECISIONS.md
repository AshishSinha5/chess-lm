# Decisions log

Each entry has the choice, the reason, the alternatives, and words to search if you want to learn more.
Status: **decided**, **proposed** (I will do it unless you object), or **open**.
I add an entry each time we make a technical choice.

## How to read this
```
data -> tokens -> model (embedding -> N x [attention + MLP] -> output) -> loss -> optimizer step
```
The model reads a game as a list of move tokens. It predicts the next move at every position. Training lowers the error of that prediction.

---

## A. Data

**A1. Use games from players rated 1800+ (both). Status: decided.**
- Why: strong players make fewer random moves. The model then learns real chess patterns.
- Alternatives: 1600 (more data, weaker games), 2000 (cleaner, too little data).
- Learn: "data quality vs quantity language models".

**A2. Use two months (2026-09, 2026-08). Status: decided.**
- Why: one month gives 0.74B tokens. We want 1B or more for training.
- Learn: "dataset size scaling".

**A3. Split by game ID hash, 98% / 1% / 1%. Status: decided.**
- Why: the same game must never be in both train and test. A hash gives the same split every run.
- Learn: "train test leakage".

**A4. Fixed vocabulary of 1,971 tokens (all possible UCI moves + 3 special). Status: decided.**
- Why: no extra pass over the data. The same vocabulary works for any month.
- Alternative: build the vocabulary from the moves seen in data (smaller, but changes per dataset).
- Learn: "tokenization vocabulary size".

**A5. Store tokens as uint16 in one binary file plus an offsets file. Status: decided.**
- Why: 1,971 fits in 16 bits. A memory-mapped file loads fast and uses little RAM.
- Learn: "numpy memmap".

**A6. Tokenizer types: move token (default) vs character. Status: decided.**
- Why: a move token gives short sequences (cheap). Characters give a tiny vocabulary but sequences 4-5x longer. We test both and report the difference.
- Learn: "word vs character level language model".

## B. Model

**B1. Decoder-only transformer (GPT style). Status: decided.**
- What: the model sees moves 1..k and predicts move k+1. It cannot see the future (causal mask).
- Why: the same method as GPT. It is simple and scales well.
- Learn: "decoder-only transformer", "causal language modeling".

**B2. Attention. Status: decided.**
- What: for each position, the model looks at earlier positions and takes a weighted mix of their information. The weights are learned.
- Why: a chess move depends on earlier moves. Attention links them directly.
- Cost: grows with the square of sequence length. This is why character-level costs more.
- Learn: "self-attention explained", "multi-head attention".

**B3. FlashAttention through PyTorch SDPA. Status: decided.**
- What: a faster, lower-memory way to compute the same attention result.
- Why: no loss in quality, and a large speed gain on the A100.
- Learn: "FlashAttention", "torch scaled_dot_product_attention".

**B4. Head size 64. Status: decided.**
- Why: a common value that runs well on GPUs. Number of heads = d_model / 64.
- Learn: "attention head dimension".

**B5. Position information: RoPE. Status: decided.**
- What: the model must know the order of moves. RoPE rotates the query and key vectors by an angle that depends on position.
- Why: standard in modern models. It encodes relative distance and needs no extra learned table.
- Alternative: learned absolute positions (simpler; one vector per position, 258 positions here). Fine for fixed-length games.
- Learn: "rotary position embedding RoPE".

**B6. Pre-norm (normalise before each block). Status: decided.**
- Why: training is more stable than post-norm, especially without careful tuning.
- Learn: "pre-LN vs post-LN transformer".

**B7. GELU MLP, hidden size 4x d_model. Status: decided.**
- Why: the standard GPT block. No reason to change it.
- Learn: "GELU activation", "transformer feed-forward network".

**B8. Tied embeddings (input and output share one table). Status: decided.**
- Why: fewer parameters, and it often helps small models.
- Learn: "weight tying language model".

**B9. No dropout. Status: decided.**
- Why: we see each game about once. The model does not overfit, so dropout only slows learning.
- Learn: "dropout overfitting epochs".

**B10. Context length 258. Status: decided.**
- Why: the longest game is 256 moves + 1 start token + 1 end token.

## C. Numbers and speed

**C1. Precision: bf16 for compute, fp32 for the weights. Status: decided.**
- What: "precision" is how many bits store each number. fp32 = 32 bits (accurate, slow). bf16 = 16 bits (about 2x faster, half the memory).
- Why bf16 and not fp16: bf16 has the same number range as fp32, so no loss scaling is needed. fp16 can overflow. The A100 supports bf16.
- How: PyTorch autocast. Weights and optimizer state stay in fp32 ("mixed precision").
- Learn: "bfloat16 vs float16", "mixed precision training".

**C2. torch.compile. Status: decided.**
- What: PyTorch fuses many small GPU operations into fewer large ones.
- Why: more tokens per second for little code. If it breaks, we turn it off.
- Learn: "torch.compile".

## D. Training

**D1. Optimizer: AdamW, betas (0.9, 0.95), weight decay 0.1. Status: decided.**
- Why: the usual setting for transformer language models. Beta2 of 0.95 reacts faster than the default 0.999.
- Learn: "AdamW optimizer", "weight decay".

**D2. Gradient clipping at 1.0. Status: decided.**
- Why: stops a rare large gradient from breaking training.
- Learn: "gradient clipping".

**D3. Learning-rate schedule: warmup, constant, decay (WSD). Status: decided.**
- What: the learning rate rises slowly, stays flat, then falls near the end.
- Why: a checkpoint from the flat part is still a good model. We use this for the token sweep (0.125B, 0.25B, 0.5B) without separate full runs.
- Alternative: cosine decay (needs the total length known in advance).
- Learn: "warmup stable decay schedule", "cosine learning rate schedule".

**D4. Batch of about 64k tokens per step (about 870 games). Status: proposed.**
- Why: large enough for stable gradients, small enough to get many steps. A 1B run is about 15,600 steps.
- Learn: "batch size critical batch size language models".

**D5. One game per sequence, grouped by length, padding ignored in the loss. Status: decided.**
- Why: simple, and every game starts at a clean position. This makes probing clean later.
- Alternative: pack many games in one sequence (less waste, but games leak into each other's context).
- Learn: "sequence packing vs padding".

**D6. Same seed and data order for all model sizes. Status: decided.**
- Why: then any difference comes from size, not from luck in the data.

## E. Going from XS to L

**E1. Four sizes: XS (5M), S (11M), M (26M), L (50M). Status: decided.**
- What: we make the model wider (d_model), deeper (layers), and add heads. Head size stays 64.
- Why: this gives a clean curve of strength vs size.
- Learn: "scaling laws for neural language models".

**E2. Train all sizes on 1B tokens. Status: decided.**
- Why: one fixed data budget makes the comparison fair. Small models see more data than the "compute-optimal" amount (about 20 tokens per parameter). Large models see about the right amount.
- Learn: "Chinchilla scaling laws".

**E3. Tune the learning rate on XS, then scale it down for bigger models. Status: proposed.**
- Why: bigger models need a smaller learning rate. Tuning every size is too expensive.
- Learn: "learning rate vs model size", "muP".

**E4. Run gates before large runs. Status: decided.**
1. Overfit 1k games with XS. Loss must fall near zero. If it does not, there is a bug.
2. Measure real tokens per second. Then fix the cost estimate.

## F. Evaluation and probing (planned)

**F1. Report legal-move rate and strength against Stockfish. Status: decided.**
- Learn: "Elo rating maximum likelihood", "Stockfish UCI_Elo".

**F2. Strict rule: an illegal move loses. Also report a lenient rule. Status: decided.**

**F3. Probing. Status: decided.**
- What: train a small linear model to read the board from the hidden activations of the network. If it works, the network has an internal board.
- Needs controls (random-init network, shuffled labels), or the result can mislead.
- Learn: "linear probing interpretability", "Othello-GPT world model".

## G. Added during implementation

**G1. Logging: W&B in offline mode. Status: decided.**
- Why: no account or key needed. Run `wandb sync <dir>` later to upload. Switching to online is one environment variable.
- Learn: "Weights & Biases offline mode".

**G2. LayerNorm, no bias terms in linear layers. Status: decided.**
- Why: GPT-style default. Fewer parameters, same quality.
- Alternative: RMSNorm (slightly cheaper, used in Llama).
- Learn: "LayerNorm vs RMSNorm".

**G3. Weight init: normal(0.02), residual outputs scaled by 1/sqrt(2 x layers). Status: decided.**
- Why: keeps the signal size stable as depth grows.
- Learn: "GPT-2 weight initialization".

**G4. Length buckets of 16, fixed batch shape per bucket. Status: decided.**
- Why: torch.compile builds one fast program per tensor shape. About 17 shapes is fine. Many shapes would be slow.
- Cost: the last partial batch of each bucket is dropped. With 20M games this is negligible. With 1,000 games it drops most games, so tiny tests need a small batch.
- Learn: "dynamic shapes torch.compile", "bucketing padding".

**G5. Turn off dynamic shapes in torch.compile. Status: decided.**
- Why: with automatic dynamic shapes on, a shape change crashed the compiler (stride assertion). Our shapes are fixed per bucket, so we do not need it.

**G6. Token count = number of target positions (real tokens, no padding). Status: decided.**
- Why: the learning-rate schedule uses tokens seen / budget. This does not depend on the batch layout.

**G7. Overfit gate: compare with the entropy floor, not zero. Status: decided.**
- What: games that share an opening prefix split later. No model can predict the split. The floor is the best possible loss on that data (0.0927 per move for 1,000 games).
- Result: XS reached 0.094. The gate passes.
- Learn: "memorization sanity check overfit single batch".

**G8. Download months to disk with parallel range requests, then process the local file. Status: decided.**
- Why: streaming over one connection fell from 43k games/s to 5k games/s (about 1 MB/s). Each extra connection gave 3-6 MB/s, so 16 connections give about 40 MB/s. A month downloads in about 13 minutes.
- Cost: 29-30 GB of disk per month in `/workspace/raw`. We can delete the files after the shards are built.
- Learn: "HTTP range requests", "parallel download".
- Note: `pkill -f` matches its own shell command line. Use a pattern like `[c]hesslm` to avoid killing the caller.
- Update: the raw files were deleted after the shard build (2026-10-20).

**G9. Learning-rate sweep: 7 rates on XS, 0.1B tokens each, same schedule. Status: done, 3e-3 proposed.**
- What: rates 1e-3, 2e-3, 3e-3, 4.5e-3, 6e-3, 1e-2, 1.5e-2. Each step is about 1.5x the last (a log grid). We pick the rate with the lowest validation loss. Script: `scripts/lr_sweep.sh`.
- Why: a short run ranks the rates well enough. 0.1B tokens costs about 2-3 minutes per run.
- Risk: the best rate for a short run can be a little higher than for a long run. Check the winner and its neighbour at 1B tokens if they are close.
- Measured speed: XS trains at about 1.2-1.6M tokens/s after the first minute (compile time).
- Result (val loss per move at 0.1B tokens): 1e-3 2.484, 2e-3 2.320, 3e-3 2.376, 4.5e-3 2.340, 6e-3 2.375, 1e-2 3.506, 1.5e-2 3.597.
- Reading: 2e-3 to 6e-3 is a flat zone (within 0.06). At 1e-2 the loss jumps by more than 1.1, so the rate is too high there. The order inside the flat zone is not clear, because 3e-3 is worse than both neighbours (noise).
- Seed check (seed 1, same 0.1B tokens): 2e-3 2.343, 3e-3 2.331, 4.5e-3 2.312. The order flips between seeds. Mean of 2 seeds: 2e-3 2.332, 3e-3 2.353, 4.5e-3 2.326. The gaps (0.03 or less) are the same size as the seed noise (up to 0.045), so the 3 rates are equal at this length.
- Choice: 3e-3, the middle of the flat zone and 3x below the rate where training fails.
- Learn: "learning rate sweep log scale", "learning rate range test".

**G10. Speed test for S, M and L; cost per 1B tokens. Status: measured.**
- What: 15M tokens per size on the A100, lr 3e-3. Script: `scripts/speed_test.sh`.
- Result: tokens/s (steady) and peak GPU memory. XS 1.5M. S 1.1M, 9.9 GB. M 0.61M, 14.8 GB. L 0.35M, 21.3 GB. Model FLOPs use (mfu) rises with size: S 25%, M 31%, L 34%.
- Time for 1B tokens: XS 11 min, S 15 min, M 27 min, L 48 min. Total about 1.7 hours, about $3 at $1.80/hour.
- Meaning: the budget is not a problem. Memory is far below 80 GB, so we can run two small jobs at once if we want.
- Learn: "model FLOPs utilization MFU", "tokens per second GPU throughput".

**G11. Learning rate by size: lr = 3e-3 x 256 / d_model. Status: proposed.**
- What: XS 3e-3, S 2e-3, M 1.5e-3, L 1.2e-3. The rate falls in proportion to the width (d_model).
- Evidence (L, 0.1B tokens, seed 0, val loss per move): 1e-3 1.959, 2e-3 1.945, 3e-3 3.332 (max grad norm 5.0, training failed). On XS, 3e-3 was in the flat zone. So the safe rate fell with size, as expected in E3.
- Why this rule: 1e-3 and 2e-3 are equal on L within noise. 1.2e-3 is inside that range and 2.5x below the failure point. The rule gives S and M values between the two tested ends.
- Risk: S and M rates are not tested. Check S and M with one short run each if the size sweep shows odd results.
- Learn: "muP maximal update parametrization", "learning rate scaling with width".

**G12. Keep results in git as small text files made by a script. Status: decided.**
- What: `scripts/summarize_runs.py` reads each run's `log.jsonl` and `config.json`. It writes `results/runs.csv`, `results/curves/`, `results/configs/` and the table in `README.md`. A dated list of steps is in `results/EXPERIMENT_LOG.md`.
- Why: before this, the results existed only in `ckpt/` and `wandb/`, which git ignores. A lost volume would lose them. Generated files also stop copy mistakes in the docs.
- Alternatives: commit the raw logs (large, noisy); use W&B only (needs an account, and the offline files are not in git).
- Rule: no checkpoints in git (`*.pt` is ignored). Only CSV, JSON, Markdown and PNG.
- Learn: "experiment tracking reproducibility", "ML experiment log best practices".

**G13. Size sweep result: loss falls with size, with smaller steps at the top. Status: measured.**
- What: XS, S, M, L on 1B tokens, one seed, rate by the G11 rule. Final val loss per move: 1.789, 1.643, 1.519, 1.465.
- Reading: the steps are -0.146, -0.124 and -0.054. L has twice the parameters of M, but gains less than half as much.
- Possible causes (not tested): L may be limited by data (tokens per parameter at 1B tokens: XS 191, S 88, M 38, L 20; the common guide is about 20); the L rate of 1.2e-3 is a little off, because it was checked only at 0.1B tokens; one seed gives noise.
- The final fall matters: L is 1.650 near 0.5B tokens in the flat phase and 1.465 at the end.
- Next: the token sweep on L shows if more data still helps. See `results/EXPERIMENT_LOG.md`, step 8.
- Update: the seed check (G15) shows that the small step from M to L is not noise. The token sweep (G17) shows both models still gain from more data.
- Learn: "Chinchilla scaling laws", "diminishing returns model size", "compute-optimal training".

**G14. Evaluation harness: how the model plays and how we compute Elo. Status: decided.**
- What: `src/chesslm/eval/play.py` measures 3 things. Results go to `results/eval/` through `scripts/summarize_eval.py`.
  1. Legal-move rate. The model sees a real game up to a random position. We take its top move with no legal mask. We check if the move is legal. We also check if it equals the human move.
  2. Games against Stockfish 19 with the `UCI_Elo` setting. Stockfish gets 0.05 seconds per move. In these games the model plays its best legal move, so one illegal move does not end the game. This is the lenient rule of F2. The strict rate is the legal-move rate from item 1.
  3. Elo. One rating is fitted to all games. The 95% interval comes from resampling the games 1000 times.
- Openings: the first 6 plies come from real validation games. The model plays each opening once as white and once as black. Both sides then face the same positions.
- Limit: the Elo is on the nominal Stockfish scale. At 0.05 seconds, Stockfish plays below its nominal level. Do not read the number as an Elo on a human rating list. Use it to compare our models with each other.
- Limit: a game stops at 256 plies and counts as a draw, the same limit as the training data.
- Why Stockfish is a file in `/workspace/bin`: the system folders reset when the pod restarts, and `/workspace` stays.
- Alternatives: play models against each other (no outside scale); use a fixed search depth (stronger Stockfish, slower games); sample moves with a temperature (more variety, but more illegal moves).
- Learn: "Elo rating maximum likelihood", "bootstrap confidence interval", "Stockfish UCI_Elo", "UCI protocol".

**G16. Token sweep: finish each saved checkpoint with a short decay. Status: decided.**
- What: for each saved checkpoint at T tokens (0.125B, 0.25B, 0.5B), copy it to a new run folder as `last.pt`. Resume with `--max_tokens` = 1.25 x T and `--decay_frac 0.2`. The rate stays at peak until T, then falls in a straight line to 10% of peak. Do this for L, and for M as a control.
- Why: a checkpoint from the flat phase has not been through the decay, so its loss is too high. Compare it with the decayed 1B model and the curve looks worse than it is. The short decay gives each point a fair comparison.
- The decay shape matches the full run. The full run has 0.8B tokens of flat rate and 0.2B tokens of decay, so the decay is 25% of the flat part. Each branch has the same ratio.
- The x-axis is the total tokens seen: 1.25 x T. That is 0.156B, 0.3125B and 0.625B. The full run stays at 1.0B.
- Cost: 0.25 x T extra tokens per branch. About 10 minutes for all three L branches.
- Limit: each branch uses one seed. The M seed spread at 1B tokens (G15) shows how large the noise is.
- Limit: the test shows if L gains from more data. It does not test the learning rate of L. That needs a second rate on L.
- Alternatives: four full runs of L (about 1.7 times the cost of the largest point, with no flat-phase trick); cosine schedule runs (the length must be known in advance).
- Learn: "warmup stable decay schedule", "learning rate annealing checkpoints", "data scaling law".

**G15. Seed check: the small step from M to L is real. Status: measured.**
- What: M and L again at 1B tokens with seed 1. Same rates and data as the size sweep. Only the seed changed. Command: `bash scripts/seed_check.sh`.
- Result (val loss per move):

| Model | Seed 0 | Seed 1 | Difference |
|---|---|---|---|
| M | 1.5192 | 1.5173 | 0.0019 |
| L | 1.4652 | 1.4622 | 0.0030 |

- The step from M to L is 0.0540 with seed 0 and 0.0551 with seed 1. The earlier steps are 0.146 (XS to S) and 0.124 (S to M). The seed spread is about 0.003. The M-to-L step is less than half of the earlier steps, and the gap is about 20 times the spread. So the smaller step is a real result for this setup (rate rule G11, 1B tokens), not noise.
- Limit: XS and S have one seed at 1B tokens. Their steps have no measured spread. We assume it is also small, because M and L are small.
- Limit: the check does not give the cause. The two causes in G13 stay open: L may be limited by data (20 tokens per parameter), and the L rate (1.2e-3) was checked only at 0.1B tokens. The token sweep (G16) tests the first cause. A second rate on L would test the second.
- Learn: "seed variance in deep learning", "run-to-run variability", "Chinchilla scaling laws".

**G17. Token sweep result: both models still gain from more data, and L gains a little less than M. Status: measured.**
- What: L and M at 0.156B, 0.313B and 0.625B total tokens (short-decay branches, G16), plus the full 1B runs. Seed 0. Val loss per move:

| Total tokens | L | M | M minus L |
|---|---|---|---|
| 0.156B | 1.785 | 1.861 | 0.076 |
| 0.313B | 1.616 | 1.682 | 0.067 |
| 0.625B | 1.512 | 1.568 | 0.056 |
| 1.0B | 1.465 | 1.519 | 0.054 |

- No plateau. Both curves still fall at 1B tokens. From 0.625B to 1.0B, L gains 0.047 and M gains 0.049.
- The gain per doubling of tokens falls for both models (L: 0.169, then 0.104).
- L gains a little less than M. Over the whole range, L gains 0.320 and M gains 0.342. The advantage of L over M shrinks from 0.076 to 0.054. The difference of 0.022 is about 7 times the seed spread (G15, about 0.003). It is small but not noise.
- Reading: the result fits a weak data limit on L, but it does not prove one. L is still ahead of M at every budget, and a data-limited L would stop gaining. It does not.
- What the sweep cannot show: whether the small M-to-L step also appears at the small budgets. G18 answers this with S and XS branches.
- Next: the S and XS branches (done, G18). Then a second learning rate on L (48 minutes) or L on all 1.64B tokens (about 77 minutes).
- Limit: one seed. Branches share the data order of the full run, so each branch point is not independent of the 1B run.
- Learn: "data scaling law", "Chinchilla scaling laws", "learning curve power law".

**G18. Token sweep for S and XS: the small M-to-L step is there at every budget. Status: measured.**
- What: the same short-decay branches (G16) for S and XS. Seed 0. Val loss per move, and the step from each size to the next:

| Total tokens | XS | S | M | L | XS to S | S to M | M to L |
|---|---|---|---|---|---|---|---|
| 0.156B | 2.111 | 1.995 | 1.861 | 1.785 | 0.117 | 0.134 | 0.076 |
| 0.313B | 1.949 | 1.812 | 1.682 | 1.616 | 0.137 | 0.129 | 0.067 |
| 0.625B | 1.834 | 1.694 | 1.568 | 1.512 | 0.140 | 0.126 | 0.056 |
| 1.0B | 1.789 | 1.643 | 1.519 | 1.465 | 0.146 | 0.123 | 0.054 |

- The M-to-L step is smaller than the S-to-M step at every budget. It is 0.57 times the S-to-M step at 0.156B tokens and 0.44 times at 1.0B. So the small step is not only a data effect. It is already there at the smallest budget.
- The trend with tokens differs by size. The step from XS to S grows with tokens (0.117 to 0.146). The step from S to M is almost flat. The step from M to L falls (0.076 to 0.054). L gains less from extra data than the smaller sizes do.
- Two parts, then. Part one is present from the start (0.57 at 0.156B). Part two grows with tokens (0.57 falls to 0.44).
- Which cause fits which part (not tested):
  - A learning rate that is off would hurt at every budget, so it could explain part one. But in the L rate check (G11, 0.1B tokens) 1e-3 and 2e-3 differ by only 0.014, so a large rate error at short budgets looks unlikely.
  - A data limit grows with tokens, so it could explain part two.
  - The cause of part one is unexplained.
- Next options: L with a second rate for the full 1B tokens (48 minutes); L on all 1.64B tokens (about 77 minutes). Choice not made yet.
- Limit: one seed per point. The seed spread at 1B tokens is about 0.003 (G15), much smaller than the differences here.
- Learn: "scaling law fit", "power law learning curve", "compute-optimal training".

**G19. Evaluation result: strength rises with size, and the M-to-L gain is smaller. Status: measured.**
- What: XS, S, M and L at 1B tokens against Stockfish 19 (`UCI_Elo` 1320 and 1500, 0.05 seconds per move). 100 openings per opponent, both colors, so 400 games per model. Method: G14.

| Model | Legal-move rate | Same move as human | Score vs 1320 | Score vs 1500 | Elo (95% interval) |
|---|---|---|---|---|---|
| XS | 98.4% | 46.2% | 0.30 | 0.15 | 1182 (1144 to 1219) |
| S | 99.4% | 49.1% | 0.45 | 0.26 | 1296 (1266 to 1329) |
| M | 99.6% | 50.8% | 0.62 | 0.38 | 1410 (1378 to 1444) |
| L | 99.5% | 52.1% | 0.71 | 0.47 | 1479 (1443 to 1510) |

- Elo steps: XS to S +114, S to M +114, M to L +69. The loss steps are 0.146, 0.123 and 0.054. The M-to-L gain is smaller in both measures. The Elo step falls by about 40% and the loss step by about 56%.
- The intervals of M and L barely overlap. The difference of 69 is about 3 times its standard error (about 24).
- Legal moves: all sizes are above 98%. M and L are equal. The rate for XS is lower.
- L scores 0.47 against Elo 1500, near an even match. So the L rating is inside the tested range, not extrapolated.
- Limit: the Elo is on the nominal Stockfish scale. At 0.05 seconds per move, Stockfish plays below its nominal level (G14). Compare the models with each other. Do not read the value as a human rating.
- Limit: the interval comes from resampling the games. It does not include the variation between training seeds or between sets of openings.
- Limit: in the games, the model plays its best legal move, so an illegal move never ends a game. The legal-move rate is measured on its own.
- Learn: "Elo rating", "bootstrap confidence interval", "Stockfish skill level versus Elo".

## Open
- Budget cap: $50 assumed.

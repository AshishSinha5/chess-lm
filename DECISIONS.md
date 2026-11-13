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
- Update (2026-11-07): the tokenizer ablation is not run in this project. All results use the move token (1,971 tokens). Reason: the character scheme makes sequences 4 to 5 times longer, so a fair comparison needs separate runs at equal game count for several sizes. The project questions (scaling, strength, the board in the model) do not depend on the tokenizer. The choice is listed under "Later" in the README. Not tested: whether a character tokenizer changes the loss, the legal-move rate or the board.

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
| S (added 2026-11-07) | 1.6427 | 1.6379 | 0.0048 |
| XS (added 2026-11-07) | 1.789 | 1.783 | 0.0060 |

- The step from M to L is 0.0540 with seed 0 and 0.0551 with seed 1. The earlier steps are 0.146 (XS to S) and 0.124 (S to M). The seed spread is about 0.003. The M-to-L step is less than half of the earlier steps, and the gap is about 20 times the spread. So the smaller step is a real result for this setup (rate rule G11, 1B tokens), not noise.
- Limit: XS and S had one seed at 1B tokens. Update 2026-11-07: seed 1 is done for both. The spread is 0.0048 (S) and 0.006 (XS), above the 0.003 of M and L, and far below the steps between sizes.
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
  - Correction (2026-10-24): the data-limit reading above is wrong. A model short of data gains more from extra tokens, so L's lead over M would grow with tokens. Here it shrinks (0.076 to 0.054), so L gains less from extra data, not more. Each run also reads each game once (1B of 1.64B tokens), so L never repeats data. 20 tokens per parameter is the compute-optimal ratio, not a limit. A rate too high for long runs fits part two better: it gives fast early progress, then a noise floor late. G20 tests this.
- Next options: L with a second rate for the full 1B tokens (48 minutes); L on all 1.64B tokens (about 77 minutes). Choice: the second rate first (G20).
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

**G20. Rate check on L: a lower rate for the full 1B tokens. Status: done, result in G22.**
- What: L at lr 6e-4 (half of the G11 rate, 1.2e-3) for 1B tokens, seed 0. Save flat-phase checkpoints at 0.125B, 0.25B and 0.5B tokens. Finish each with the short decay of G16. Script: `scripts/rate_check.sh l 6e-4`. Output: `results/rate_check.csv`.
- Why: the rate is the only open cause that fits both parts of the small M-to-L step (G18). L failed at 3e-3, and 1.2e-3 is only 2.5 times lower. The L rate was checked only at 0.1B tokens, where 2e-3 was a little better than 1e-3. Short runs often prefer a higher rate than long runs.
- Why lower, not higher: a higher rate is closer to the failure point. A factor of 2 gives an effect well above the seed spread (about 0.003, G15).
- How to read the result at 1B (baseline 1.465):
  - 0.03 or more below 1.465: the rate is a main cause. Retune L, then likely M.
  - Within 0.01 of 1.465: 1.2e-3 is near the best rate. The rate is not the cause.
  - Worse at short budgets but better at 1B: the best rate depends on run length. This explains the shrinking lead.
  - Worse at all budgets: try 1.7e-3 next.
- Alternative not chosen now: L and M on all 1.64B tokens (about 2 hours). The token sweep (G17, G18) already tested the data-limit prediction at 4 budgets, and it failed each time. Run it only if this check explains nothing.
- Limit: the M rate (1.5e-3) is also untested. If L gains a lot, M may gain too. Then measure the step again with both rates tuned.
- Cost: about 48 minutes for the full run and about 15 minutes for the branches.
- Learn: "learning rate schedule length dependence", "loss noise floor learning rate", "edge of stability".

**G21. Probe design: positions, targets, controls and intervention. Status: done, result in G23.**
- What: code in `src/chesslm/probe/`. Command: `python -m chesslm.probe.run layers --ckpt size_m`, then `intervene`.
- Positions: 20k games from the test split (no run trains or evaluates on it), up to 5 random positions per game, so about 100k positions. A position is the hidden state at the token of move k, where the model predicts move k+1. Probe train and test sets split 80/20 by game.
- Targets: piece class on each of 64 squares (13 classes). Variant: player-relative (mine/theirs, ranks mirrored when black is to move).
- Probe: linear (main) and an MLP with 1,024 hidden units (comparison). AdamW, at least 3,000 steps. Rate 1e-2 for linear, 1e-3 for MLP. In the first test, 1e-3 with too few steps left the linear probe near its random start.
- Metrics: balanced accuracy on non-empty squares (mean recall over the 12 piece classes), accuracy on all squares, whole-board match. Also accuracy by ply at the best layer, to compare with the illegal-move analysis.
- Controls: majority class per square; shuffled labels (whole boards moved to other positions); random-init model with the same shape; embedding-only (layer 0); move-history baseline.
- Move-history baseline: per square, the last event (untouched, vacated, white arrived, black arrived) and the from-square of the last move that landed there, plus the side to move. It does no board simulation, so it misses captures by en passant, the rook in castling and promotions. It shows how much simple bookkeeping of moves gives.
- Intervention: at block output l, remove the top-k directions of the probe weights (by singular value) from the residual stream at all positions. Compare with k random orthonormal directions. Layers 2, 4, 6; seeds 0, 1, 2; k = 32 and 128. Measures: loss on the human move, legal-move rate, same-move rate, and the board probe at the last layer.
- Why k < 512: the probe weights span all 512 directions of M, so removing the full span removes everything. The top directions carry most of the probe's signal.
- Early note (300 games, too small to use): the random-init model scored close to trained M. The full run did not confirm it (G23). The small test had too few positions and an undertrained probe.
- Learn: "linear probing representations", "probe selectivity control tasks Hewitt Liang", "Othello-GPT world model", "amnesic probing INLP".

**G22. Rate check result: the L rate is not the cause of the small M-to-L step. Status: measured.**
- What: L at lr 6e-4 against the G11 rate 1.2e-3, at 4 budgets (G20). Seed 0. Val loss per move:

| Total tokens | 1.2e-3 | 6e-4 | Difference |
|---|---|---|---|
| 0.156B | 1.785 | 1.877 | +0.092 |
| 0.313B | 1.616 | 1.664 | +0.048 |
| 0.625B | 1.512 | 1.532 | +0.020 |
| 1.0B | 1.465 | 1.473 | +0.008 |

- Reading against G20: the 1B result is within 0.01 of 1.465. 1.2e-3 is at or near the best rate for L at 1B tokens.
- The lower rate is worse at every budget, so G20 says to try 1.7e-3 next. We do not run it. A halving of the rate changes the 1B loss by only 0.008, and the gap shrinks with tokens. A higher rate is unlikely to give the 0.04 or more needed to explain the step at 1B. It is also closer to the failure at 3e-3 (G11).
- In the flat phase, 6e-4 led by 0.006 at 0.62B tokens. The decay reversed this. So a loss in the flat phase is a poor guide to the final loss. The decay removes most of the noise that a higher rate adds.
- Limit: at short budgets the rate matters more (0.092 at 0.156B). A higher rate could lower L at 0.156B and so explain part of the small step there (G18, part one). The M rate is also untested at short budgets. This stays open.
- Conclusion for the project: the small M-to-L step is a measured result with no known cause. The data limit (G18 correction) and the L rate at 1B (this entry) do not explain it. We report it as the open result of the scaling study.
- Learn: "learning rate sensitivity scaling", "learning rate annealing loss drop", "WSD schedule warmup stable decay".

**G23. Probing result: M builds a board representation, beats bookkeeping late in the game, and uses it. Status: measured.**
- What: probes, 5 controls and the intervention on M (G21). Numbers are in `results/probe/` and in `results/EXPERIMENT_LOG.md`, step 14.
- Present: a linear probe reads the board from M with balanced accuracy 0.952 at layer 8 (whole board right in 53.8% of positions). Random-init M gives 0.614, shuffled labels 0.155, majority class 0.141. Training adds about 0.34 over the same architecture with random weights.
- Built by layer: 0.40 at the embedding, 0.85 after block 1, then a slow rise to 0.95. Most of the board appears in the first block.
- Linear form: the MLP probe gains little (0.963 against 0.952). The board is readable with a weighted sum.
- Frame: the absolute target (white/black) reads better than the player-relative target (0.952 against 0.933). This is unlike Othello-GPT.
- Against bookkeeping: overall, the move-history baseline is close (0.943). By ply the picture differs. Early (ply 1-20) bookkeeping is better. After ply 40, M is better: whole-board match 25% against 16% at ply 41-60, and 19% against 10% after ply 80 (MLP: 36% and 31%). The two methods make different errors. So M tracks state that simple bookkeeping of moves cannot.
- Used: removing the top 32 probe directions at a layer raises the loss on human moves 6 to 8 times more than 32 random directions (layer 4: +0.293 against +0.038). The legal-move rate falls to 97.8% against 99.4%. The board probe at the last layer drops to 1.8% whole-board match against 40.7%. Seeds agree within 0.005 in loss.
- Limit: random directions are a weak control in one way. Probe directions may lie along high-variance directions of the residual stream, which any change would hurt more. A stronger control is random directions matched in variance. Not run.
- Limit: rank 128 (a quarter of the 512 dimensions) hurts a lot even with random directions. Rank 32 gives the clean comparison.
- Limit: one model (M), one seed. L and XS not probed.
- Learn: "linear probe world model chess", "causal intervention representation ablation", "selectivity probe control".

**G24. L check: L and M on all the data (1.6B tokens). Status: measured. Data is not the cause.**
- What: fresh L and M runs at 1.6B target tokens, seed 0. The settings are the same as the 1B size sweep (G13). Only `--max_tokens` changes. Command: `bash scripts/full_data.sh`. Run names: `full_l`, `full_m`.
- Why: G20 says to run this check if the rate check explains nothing, and G22 explained nothing. It tests the data limit once more, at a budget above 1B tokens.
- Why 1.6B, not 1.64B: one pass over the data is 1.615B target tokens. Each game has one token that is not a target, so 22.4M of the 1.637B tokens are never predicted. At 1.6B no game repeats.
- Why fresh runs, not branches from the 0.5B checkpoint: the warmup is 2% of `max_tokens` (`train.py:29`). A branch keeps the 20M-token warmup of the 1B run. A fresh run has a 32M-token warmup, the same method as every 1B run.
- How to read the result. The L lead over M is 0.054 at 1B tokens (1.519 against 1.465):
  - The lead grows by 0.01 or more: L was short of data. The step grows with data.
  - The lead changes by 0.005 or less (about the seed spread, G15): data is not the cause.
  - The lead shrinks by more than 0.005: the same trend as the token sweep (G17). L gains less from data.
- Extra: a flat-phase checkpoint at 1B tokens (`tok_1000000000.pt`). It is not decayed, so do not compare it with the decayed 1B runs.
- Limit: one seed. If the lead grows, run a second L seed to confirm it.
- Limit: 1.6B tokens is 1.6 times the old budget. A small data effect may need a larger factor to show.
- Cost: about 76 minutes for L and about 45 minutes for M.
- Part 1 (2026-10-29): L at 1.6B gives val loss 1.431 (1B: 1.465, so -0.034). M at 1.6B is still to run, so the lead is not known yet. Next: `bash scripts/full_data.sh m`.
- Part 2 (2026-11-07): M at 1.6B gives val loss 1.4823 (1B: 1.519, so -0.037). L gains -0.034. The L lead over M is 0.051 at 1.6B, against 0.054 at 1B. The change is -0.003, inside the 0.005 band (seed spread about 0.003, G15).
- Result: data is not the cause of the small M-to-L step. L gains slightly less from extra data than M, the same trend as the token sweep (G17). With the L rate check (G22), both planned hypotheses are out. The step is a measured result with no known cause. Untested: the L shape (depth against width).
- Closed (2026-11-07), untested: the most likely cause is the noise floor of the task. The model predicts the move of a human player rated 1800 or more, and humans choose between good moves at random. That part of the loss no model can remove. The loss steps (1.789, 1.643, 1.519, 1.465) and the Elo steps (+114, +114, +69) both flatten at M to L, and L gains less from extra data than M. A second, less likely cause is the L shape (depth against width). No more GPU hours go to this.
- Decision: no second L seed at 1.6B. That run was only for a lead that grows.
- Learn: "data-constrained scaling laws", "Chinchilla tokens per parameter", "single epoch training".

**G25. Probe errors by piece type on M: M is best where bookkeeping is worst. Status: measured.**
- What: test errors of the layer-8 linear probe, split by true class and by error kind (wrong piece, missed piece, ghost piece). Same for the move-history baseline. Code: `src/chesslm/probe/errors.py`. Numbers: `results/EXPERIMENT_LOG.md`, step 15.
- Result: the two methods fail on different pieces. Move history is better on pawns and knights. The probe on M is better on bishops, rooks, kings and most on queens (0.93 against 0.86).
- Why bookkeeping fails there: it knows only where pieces moved, not which piece. A long-range piece (queen, rook, bishop) leaves many from-squares, and castling moves the rook without its own move. Pawns and knights have few start squares, so the from-square names them.
- Main probe error: the right piece type with the wrong colour (about 2% for knights, bishops and queens). So M stores piece type and colour along partly separate directions. This is a hypothesis, not a test.
- Queen errors go to rook and bishop, the two pieces that move like parts of a queen.
- Ghost pieces are 3 times more common on squares that a piece left than on untouched squares (0.62% against 0.19%). M sometimes keeps a piece after it moves away.
- Limit: one model, one layer, one probe seed. The MLP probe is not split by piece.
- Next: the same split for L and XS (G24 queue). A test of the colour hypothesis: probe piece type and colour as two separate targets.
- Learn: "confusion matrix", "precision and recall", "superposition features linear representation".

## Open
- Budget cap: $50 assumed.

**G26. Variance-matched control for the intervention: the board directions matter beyond their variance. Status: measured.**
- What: G21 and G23 remove 32 board directions and compare with 32 random directions. The board directions may carry more hidden-state variance than random ones, and removing variance alone could explain the loss. This run adds a third arm: random directions with the same variance as the board directions.
- How: `matched_directions` in `src/chesslm/probe/run.py`. It draws Q = orth(cov^(t/2) Z) from the hidden-state covariance of the train positions. It bisects t until the variance of Q equals the variance of the board directions. The CSV column `var_frac` is the share of total variance. New kind: `random_matched`.
- Command: `python -m chesslm.probe.run intervene --ckpt size_m --layers 2,4,6 --seeds 0,1,2 --ranks 32,128`. Same cache and 3 probe seeds. The probe and random arms reproduce the G23 numbers (largest loss difference 0.0002).
- Variance share at rank 32: board directions 10.4% (block 2), 8.0% (block 4), 9.3% (block 6). Plain random 6.4%. The board directions do carry more variance, as feared.
- Loss increase at rank 32, mean of 3 seeds (no change: 1.507):

| Block | Board directions | Random | Random, same variance | Board over matched |
|---|---|---|---|---|
| 2 | +0.149 | +0.022 | +0.040 | 3.7 times |
| 4 | +0.293 | +0.038 | +0.044 | 6.7 times |
| 6 | +0.312 | +0.050 | +0.063 | 5.0 times |

- Result: matching the variance raises the random loss a little (+0.022 to +0.040 at block 2). It does not explain the board effect. The board directions still hurt 3.7 to 6.7 times more. The claim "M uses the board" stands, with a smaller factor than the 6 to 8 times against plain random.
- Rank 128: random directions already carry 25.5% of the variance, more than the board directions (14% to 21%). The matched arm then equals plain random. The board directions still hurt more (+0.46, +0.76, +0.61 against +0.30, +0.46, +0.35). Rank 128 stays a poor comparison, because random removal hurts a lot too.
- Limit: matching uses total variance of the 32 directions. It does not match each direction one by one, or the variance in the directions that matter for the next move.
- Limit: one model (M), one cache (20k games, seed 0), 3 probe seeds.
- Learn: "ablation controls", "matched controls in interpretability", "covariance and principal components".

**G27. Probing on XS and L: the board gets better with size. Status: measured.**
- What: the layer probes and 5 controls of G21 on XS and L (1B-token runs, seed 0), then the piece-type errors of G25. Same 20k test games and the same cache design as M. Commands: `python -m chesslm.probe.run layers --ckpt size_l --mlp` (and `size_xs`), then `python -m chesslm.probe.errors --ckpt size_l --layer 9` (and `size_xs --layer 6`).
- Result: linear probe balanced accuracy 0.866 (XS), 0.952 (M), 0.968 (L). Whole board right 23.6%, 53.8%, 65.1%. The step from M to L (0.016) is smaller than from XS to M (0.086).
- XS is below the move-history control (0.943) at every ply range. M beats it from ply 41. L beats it from ply 21.
- Queens are the hardest piece at every size. Ghost pieces fall with size (vacated squares: 2.7%, 0.62%, 0.32%).
- The board improves more than the loss between M and L (board +0.016, +11 points of whole board; loss only -0.054). This is a note, and it does not explain the small M-to-L loss step (G24).
- Limit: one seed per size. S is not probed. The intervention (G23, G26) was run on M only, so "L uses its board" is not tested.
- Output: `results/probe/*.csv` (rows for `size_xs` and `size_l`), `results/figures/board_by_size.png`, `docs/PROBING.md` section 9.
- Learn: "scaling of internal representations", "probing across model sizes".

**G28. Strict and lenient illegal-move results: the legal-move filter is worth 80 to 250 Elo. Status: measured.**
- What: the F2 rule says an illegal move loses (strict) and we also report a lenient rule. The published Elo (G14, G19) is lenient: an illegal top move is skipped and the model plays its best legal move. This run adds the strict Elo. The harness (`src/chesslm/eval/play.py`) now records, in every game, each model turn where the top move over all tokens was illegal. A game with one such turn scores 0 under the strict rule.
- Command: `python -m chesslm.eval.play --ckpt size_X --legal_games 5000 --opponents elo:1320,elo:1500 --games 100 --workers 16 --out results/eval/strict_rerun/size_X.json` for the 4 sizes. Summary: `python scripts/summarize_strict.py` writes `results/eval/strict_vs_lenient.csv`.
- Same openings, 400 games per model. Stockfish is not seeded, so the rerun games differ from the published ones. Both ratings below come from the same rerun games.

| Model | Top move legal (positions) | Games with an illegal move | Elo, lenient | Elo, strict | Gap |
|---|---|---|---|---|---|
| XS | 98.4% | 333 of 400 | 1184 (1150 to 1219) | 934 (855 to 991) | 250 |
| S | 99.4% | 273 of 400 | 1289 (1255 to 1321) | 1113 (1063 to 1154) | 176 |
| M | 99.6% | 176 of 400 | 1394 (1362 to 1425) | 1300 (1266 to 1334) | 94 |
| L | 99.5% | 149 of 400 | 1454 (1422 to 1486) | 1376 (1342 to 1410) | 78 |

- Result: a small illegal-move rate adds up over a game. Even for L, 37% of games contain at least one illegal top move (0.66 per game). The strict rule costs 250 Elo for XS and 78 for L.
- The size order stays the same under both rules. The strict steps are 179, 187 and 76 Elo (XS to S, S to M, M to L). The lenient steps in the rerun are 105, 105 and 60.
- The rerun lenient numbers (1184, 1289, 1394, 1454) match the published ones (1182, 1296, 1410, 1479) within the published 95% intervals. The published numbers stay the headline lenient numbers.
- Correction to my earlier guess: I expected the gap to be small for M and L. It is 78 to 94 Elo, so it is not negligible.
- Limit: the strict rule here uses the raw top move over all tokens at every model turn. A model that is allowed to retry would sit between the two rules. We did not measure that.
- Limit: one seed per model and one Stockfish setting (0.05 seconds, `UCI_Elo` 1320 and 1500).
- Learn: "rule-following in language models", "legal move rate chess LM".


**G29. Scaling law: loss follows a two-term power law, and Elo is close to linear in loss. Status: measured.**
- What: fit loss = E + A/N^alpha + B/D^beta to the 16 token-sweep runs (4 sizes x 4 budgets, `results/token_sweep.csv`). N is the parameter count and D is the tokens seen. A grid over alpha and beta, with least squares for E, A and B. Command: `python scripts/fit_scaling.py`.
- Result: alpha 0.32 (95% interval 0.21 to 0.43), beta 0.66 (0.55 to 0.81), E 1.005. RMSE 0.0069. The data term falls faster than the size term, so at 1B tokens more tokens help more than more parameters.
- Check: the fit never saw the 1.6B-token runs. It predicts full_m within 0.008 and full_l within 0.013.
- The fit predicts a gap of 0.072 between M and L at 1B tokens. The measured gap is 0.054. So the small M-to-L step is partly a fit residual of 0.018, and the law does not explain it away.
- Elo against loss (4 sizes, 1B tokens): lenient 82 Elo per 0.1 loss, strict 137 Elo per 0.1 loss. R2 above 0.99 for both.
- Limit: the short-budget points are checkpoints finished with a short decay, not separate runs. The intervals come from a bootstrap of 16 points and are rough. E is near the edge of its interval. Elo has only 4 points per line, and it is measured at 1B tokens only. One seed per point.
- Output: `results/scaling_fit.csv`, `results/scaling_points.csv`, `results/figures/scaling_law.png`.
- Learn: "Chinchilla parametric loss fit", "scaling laws for neural language models".


**G30. RL rollout engine: Stockfish in worker processes, 10 engines, two play modes. Status: measured.**
- What: `src/chesslm/rl/rollout.py` plays many games at once. One batched GPU forward pass serves all games where the model is to move. Stockfish runs in a pool of worker processes, one engine per process (`rl/engine_worker.py`). A finished game frees its slot at once (continuous batching). Mode `sample` draws a token from the full softmax, and an illegal token ends the game as a loss (the RL policy). Mode `greedy_legal` is the stage-1 rule: best legal move by score, and it counts raw top tokens that are illegal. So lenient and strict scores come from the same games (G28).
- Why processes: the first version ran 11 engines as threads in the main process. python-chess parses every engine output line in Python, so the engine threads took the interpreter time (GIL). One model forward pass took 190 ms, against 8 ms alone. Adding engines did not raise throughput (1.0 to 1.1 games/s for 8 to 16 engines). With processes the forward pass takes 25 to 30 ms and games per second rise with the engine count.
- Why 10 engines: the container quota is 11.9 CPUs (`nproc` shows 116). At 11 engines the container was paused in 20% of the quota periods. A fixed time per move on a busy CPU makes Stockfish weaker, which would move the Elo scale. Throughput at 11 is only 6% above 10.
- Why torch threads are set to 4: torch starts one thread per `nproc` CPU (58 here), against the quota of 11.9. CPU work ran about 15 times slower until `OMP_NUM_THREADS=4` was set.
- Result: see `results/EXPERIMENT_LOG.md`, step 26. The first loop moved all games by one ply per tick and waited for the slowest engine, so the engines were idle half of the time (2.09 games/s). The async loop lets each game go on as soon as its own engine answers: 3.24 games/s, engine use 78%. Each game has its own random generator, seeded from the run seed and the job index, and samples with the Gumbel-max trick. So the games do not depend on the batch.
- Finding: at a fixed 50 ms per move, Stockfish searches fewer nodes when the CPU is busy (37.8k nodes with 4 engines, 24.5k with 10). The opponent gets weaker and the Elo of the model rises. So Elo values from different pods or engine counts do not compare. Every eval stores `sf_nodes_per_move`.
- Open: limit Stockfish by nodes instead of time, so the opponent does not depend on the load. Needs a new calibration of the opponent strength. Not done. The user decides.
- Learn: "Python GIL I/O threads", "continuous batching", "cgroup cpu.max throttling", "python-chess engine asyncio".

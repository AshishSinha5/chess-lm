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

## Open
- Budget cap: $50 assumed.

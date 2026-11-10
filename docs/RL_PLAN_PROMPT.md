# Prompt: plan the RL post-training stage

PLAN ONLY. Do not write implementation code yet. Read `CLAUDE.md`, `PLAN.md`, `DECISIONS.md`, `results/EXPERIMENT_LOG.md`, `docs/PROBING.md` and `docs/RUNBOOK.md` first. Follow `CLAUDE.md` for style, diagrams and logging.

## Project and goal
Chess LM (done, public) trained 4 GPT models (XS/S/M/L, 5M to 50M parameters) on Lichess games, with one token per move. Add an RL post-training stage. The goal is the same as before: a public repo that shows real depth in deep learning and LLMs. Credibility comes from findings, ablations and honest analysis, not only working code. I must be able to explain every design choice in an interview. I am new to RL training details, so give each choice a DECISIONS entry with search terms, and plan a teaching doc for each new method (as `docs/PROBING.md` did).

## What the first stage gives us (use as the base, do not redo)
- Strength vs Stockfish: lenient Elo 1182 (XS) to 1479 (L). Strict Elo (one illegal move = loss): 934 to 1376. L plays an illegal top move in 37% of games.
- Loss follows E + A/N^0.32 + B/D^0.66. Elo is close to linear in loss.
- Linear probes read the board from the hidden states (balanced accuracy 0.866 XS, 0.952 M, 0.968 L). Removing the probe directions in M hurts 3.7 to 6.7 times more than variance-matched random directions.
- Existing harness: Stockfish matches, legal-move metrics, error analysis by ply/piece/phase, probe and intervention code, GUI, HF weights.

## Research questions (the plan must say which it will answer)
1. Does RL raise strength, and how much for each size? Compare with the SFT baselines.
2. Does RL cut the illegal-move rate? This is the gap between lenient and strict Elo.
3. What does RL do to the internal board? Re-run the probes and the intervention on the RL models. Does the board get better, stay the same or degrade?
4. Does RL gain depend on model size, in the way SFT gain does?
5. What does RL cost in KL distance from the base model, and in move diversity?

## Scope for the plan to decide, with a recommendation for each
- **Algorithm.** Compare at least: rejection sampling or expert iteration (simple baseline), and one policy-gradient method (GRPO or PPO). Say which is the main method and why. Explain each in plain words.
- **Reward.** Options: legality only, game result vs Stockfish, result plus legality penalty, Stockfish evaluation change per move. State what each option teaches and its risk of reward hacking. `PLAN.md` said engine evals as input were out of scope. Decide whether an engine reward is in scope for RL, and say why.
- **Opponent.** Stockfish at fixed Elo levels, a curriculum of levels, or self-play. Handle the case where the reward is almost always a loss at first.
- **Starting points.** Which sizes to post-train (suggest M and L, maybe XS for fast tests). Use the 1B-token checkpoints or the 1.6B-token ones.
- **KL control.** KL penalty to the SFT model, its coefficient, and how to log it.
- **Baselines the RL result must beat.** More SFT steps, SFT on winning games only, and sampling temperature or legal-move masking at inference time. A mask at inference is a strong, cheap baseline for the illegal-move gap. The plan must test it.
- **Evaluation.** Reuse the Stockfish match harness and Elo method with confidence intervals. Use games that were not seen in training. Keep the strict and lenient rules. Add legal-move rate and move diversity.
- **Ablations (pick the 3 to 5 with most value).** Reward type, KL coefficient, algorithm, opponent level, model size.

## Constraints
- 1x A100 80GB on RunPod on-demand (about $1.80/hr). Give a GPU-hour and cost table, and measure real games per second before you fix the budget. Stockfish CPU time is likely the bottleneck. Check this.
- Everything that lasts lives in `/workspace`. Train in `tmux`. No keys or pod IDs in the repo. No checkpoints or W&B files in git.
- No commit, push or HF upload unless I ask.
- Out of scope: search at inference (MCTS), models above 50M parameters, beating strong engines, tuning for leaderboard Elo.

## Required output: a plan in the same format as `PLAN.md`
1. Goal and out-of-scope list.
2. Research questions mapped to experiments, and a decision rule for each (what result would make us say yes or no).
3. Milestones by weekend, each with exit tests. Start with the cheapest experiment that can fail fast (small model, legality-only reward).
4. Reward, algorithm and opponent design, each with a diagram and a table of the alternatives.
5. Infra and cost table.
6. Logging: the CSV and file names that fit the existing `results/` layout, the new steps for `build_results.sh`, and the new entries in `docs/RUNBOOK.md`.
7. Risks (reward hacking, collapse of move diversity, unstable training, Stockfish cost) with a check or fallback for each.
8. Docs and write-up: a teaching doc for RL, an update to the blog, and a list of 6 or more plots.
9. Mark every guess `[ASSUMPTION]` and every unmeasured number `[UNVERIFIED]`. End with a short list of questions for me. Ask only what you cannot decide with a sensible default.

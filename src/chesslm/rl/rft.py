"""Rejection-sampling fine-tuning, also called expert iteration (docs/RL_PLAN.md, W4 step 5b; DECISIONS G34).

One round:
  1. Play games with the current model against a Stockfish mixture. The model samples its moves at temperature T
     with no legal mask. An illegal token ends the game as a loss (stage-1 strict rule).
  2. Keep the games the model won (--keep_draws also keeps draws).
  3. Fine-tune on the MODEL tokens of the kept games. Opening plies and opponent moves are masked out of the loss.
     The loss is the stage-1 next-move cross entropy, so this is SFT on the model's own wins.
  4. Evaluate: greedy strict and lenient Elo (same openings and opponents as the step-29 anchor), and the loss on
     human moves (the cost, Q5).
The next round uses the updated model. Openings come from the TRAIN split; evaluation uses the val split.

Usage: python3 -m chesslm.rl.rft --ckpt size_m --name rft_m --rounds 4 --games 10000 --temperature 0.7
Saves /workspace/ckpt/NAME/round{r}.pt and final.pt (stage-1 format).
Logs: results/rl/rft.csv (one row per round),
results/rl/NAME/config.json (arguments, model config, git hash, library versions),
results/rl/NAME/steps.csv (one row per SFT step: loss, lr, grad norm, tokens, tokens per second),
/workspace/ckpt/NAME/last.pt (weights, optimizer, rng, round: rerun the same command to resume after a crash),
results/rl/NAME/round{r}_rollouts.jsonl (every rollout game with tokens, model positions, log-probs and a kept flag),
results/rl/NAME/round{r}_eval.json (the greedy eval games). No W&B."""
import argparse
import csv
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from chesslm.eval.play import TOK, load_model
from chesslm.rl.bandit import base_logprobs, evaluate as heldout_metrics, heldout
from chesslm.rl.rollout import EnginePool, make_jobs, openings, play_games, summarize

REPO = Path(__file__).resolve().parents[3]
FIELDS = ["name", "ckpt", "round", "games", "temperature", "opponents", "lr", "epochs", "keep_draws", "win_rate", "draw_rate",
          "illegal_end_rate", "mean_plies", "kept_games", "kept_tokens", "train_loss_first", "train_loss_last",
          "greedy_strict_elo", "greedy_strict_lo", "greedy_strict_hi", "greedy_lenient_elo", "greedy_lenient_lo",
          "greedy_lenient_hi", "legal_mass", "top1_legal", "human_nll", "kl_to_base", "play_seconds", "total_seconds"]


def build_batch(games, device="cuda"):
    """Padded inputs, targets and a loss mask. The mask is 1 only where the target is a MODEL move.
    game.tokens = [bos, ply 1, ply 2, ...]; game.model_pos lists the index t of each model token.
    The logits at index t-1 predict tokens[t], so the mask is set at t-1."""
    T = max(len(g.tokens) for g in games) - 1
    x = torch.full((len(games), T), TOK.pad, dtype=torch.long)
    y = torch.full((len(games), T), TOK.pad, dtype=torch.long)
    m = torch.zeros(len(games), T)
    for i, g in enumerate(games):
        t = torch.tensor(g.tokens)
        x[i, : len(t) - 1], y[i, : len(t) - 1] = t[:-1], t[1:]
        m[i, torch.tensor(g.model_pos) - 1] = 1.0
    return x.to(device), y.to(device), m.to(device)


def sft_loss(model, games, device="cuda", amp=True):
    """Mean cross entropy over the model tokens of the games."""
    x, y, m = build_batch(games, device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and str(device).startswith("cuda")):
        logits = model(x)[0]
    ce = F.cross_entropy(logits.float().transpose(1, 2), y, reduction="none")
    return (ce * m).sum() / m.sum()


def finetune(model, opt, kept, epochs, batch, rng, device="cuda", clip=1.0):
    """Epochs over the kept games, shuffled. Returns one dict per step:
    epoch, step, loss, lr, grad_norm, tokens (model tokens in the batch), tok_s (those tokens per second)."""
    model.train()
    steps = []
    for ep in range(1, epochs + 1):
        order = rng.permutation(len(kept))
        for i in range(0, len(order), batch):
            t0 = time.time()
            chunk = [kept[j] for j in order[i: i + batch]]
            loss = sft_loss(model, chunk, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            toks = sum(len(g.model_pos) for g in chunk)
            steps.append({"epoch": ep, "step": len(steps) + 1, "loss": loss.item(), "lr": opt.param_groups[0]["lr"],
                          "grad_norm": float(gn), "tokens": toks, "tok_s": toks / max(time.time() - t0, 1e-9)})
    model.eval()
    return steps


STEP_FIELDS = ["name", "round", "epoch", "step", "loss", "lr", "grad_norm", "tokens", "tok_s"]


def game_row(g, kept):
    """Full record of one rollout game: result fields, the token ids, and the positions of the model tokens."""
    return {**g.record(), "kept": kept, "tokens": g.tokens, "model_pos": g.model_pos,
            "model_logp": [round(x, 4) for x in g.logp]}


def git_hash():
    r = subprocess.run(["git", "-c", f"safe.directory={REPO}", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip()


def write_config(path, args, model, ckpt_start):
    """Everything needed to repeat the run: all arguments, model config, parameter count, code version, libraries."""
    cfg = {"args": vars(args), "model_cfg": model.cfg.__dict__, "n_params": sum(q.numel() for q in model.parameters()),
           "git": git_hash(), "torch": torch.__version__, "python": platform.python_version(),
           "start_ckpt": str(ckpt_start), "cpus_nproc": os.cpu_count(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(path, "w"), indent=1)
    return cfg


# Arguments that may differ between a run and its resume. All others must match, or the resume is refused.
RESUME_FREE = {"rounds", "out", "resume", "engines"}


def save_state(model, opt, rng, rnd, path):
    """Full state after a round: weights, optimizer, numpy rng, and the round number (for --resume)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model_cfg": model.cfg.__dict__, "model": model.state_dict(), "opt": opt.state_dict(),
                "rng": rng.bit_generator.state, "round": rnd}, tmp)
    tmp.rename(path)


def save(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_cfg": model.cfg.__dict__, "model": model.state_dict()}, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="start checkpoint: run name under /workspace/ckpt, or a .pt path")
    p.add_argument("--name", required=True, help="output run name: /workspace/ckpt/NAME/")
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--games", type=int, default=10000, help="games per round")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--opponents", default="elo:1320,elo:1500,elo:1700")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=64, help="games per SFT step")
    p.add_argument("--keep_draws", action="store_true")
    p.add_argument("--engines", type=int, default=10)
    p.add_argument("--movetime", type=float, default=0.05)
    p.add_argument("--eval_games", type=int, default=100, help="val openings per opponent for the greedy eval")
    p.add_argument("--held_n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                   help="continue from /workspace/ckpt/NAME/last.pt if it exists (default). --no-resume refuses to reuse a name")
    p.add_argument("--out", default=str(Path(__file__).parents[3] / "results/rl/rft.csv"))
    a = p.parse_args()
    device = "cuda"
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    model, base = load_model(a.ckpt), load_model(a.ckpt)
    held = heldout(a.held_n, seed=123)
    blp = base_logprobs(base, held)
    del base
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    opps = a.opponents.split(",")
    n_open = a.games // (2 * len(opps))
    eval_jobs = make_jobs(openings(a.eval_games, 1, "val"), ["elo:1320", "elo:1500"])  # same as the step-29 anchor
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gdir = out.parent / a.name  # per-run files: config, steps, per-round games
    gdir.mkdir(parents=True, exist_ok=True)
    step_out = gdir / "steps.csv"
    ckdir = Path("/workspace/ckpt") / a.name
    last_pt, start_round = ckdir / "last.pt", 1
    if last_pt.exists():
        if not a.resume:
            raise SystemExit(f"{last_pt} exists: use a new --name, or --resume")
        old = json.load(open(gdir / "config.json"))["args"]
        diff = {k: (old.get(k), v) for k, v in vars(a).items() if k not in RESUME_FREE and old.get(k) != v}
        if diff:
            raise SystemExit(f"resume refused, arguments differ from config.json: {diff}")
        st = torch.load(last_pt, map_location=device)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        rng.bit_generator.state = st["rng"]
        start_round = st["round"] + 1
        print(f"resumed after round {st['round']}", flush=True)
    else:
        write_config(gdir / "config.json", a, model, a.ckpt)
    new = not out.exists()
    t_start = time.time()
    with EnginePool(a.engines, movetime=a.movetime) as pool, open(out, "a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        new_steps = not step_out.exists()
        fs = open(step_out, "a", newline="")
        ws = csv.DictWriter(fs, STEP_FIELDS)
        if new_steps:
            ws.writeheader()
        for r in range(start_round, a.rounds + 1):
            t0 = time.time()
            jobs = make_jobs(openings(n_open, 1000 * a.seed + r, "train"), opps)
            games = play_games(model, jobs, pool, mode="sample", batch=4 * a.engines, temperature=a.temperature,
                               seed=1000 * a.seed + r)
            play_s = time.time() - t0
            n = len(games)
            wins = [g for g in games if g.score == 1.0]
            draws = [g for g in games if g.score == 0.5]
            kept = wins + (draws if a.keep_draws else [])
            kept_tokens = sum(len(g.model_pos) for g in kept)
            print(f"round {r}: {n} games in {play_s:.0f} s, wins {len(wins)}, draws {len(draws)}, "
                  f"illegal ends {sum(g.illegal for g in games)}, kept tokens {kept_tokens}", flush=True)
            kept_ids = {id(g) for g in kept}
            with open(gdir / f"round{r}_rollouts.jsonl", "w") as fg:
                for g in games:
                    fg.write(json.dumps(game_row(g, id(g) in kept_ids)) + "\n")
            steps = finetune(model, opt, kept, a.epochs, a.batch, rng, device)
            for s_ in steps:
                ws.writerow({"name": a.name, "round": r, **{k: round(v, 5) if isinstance(v, float) else v for k, v in s_.items()}})
            fs.flush()
            first, last = steps[0]["loss"], steps[-1]["loss"]
            ev = summarize(play_games(model, eval_jobs, pool, mode="greedy_legal", batch=4 * a.engines, seed=0))
            json.dump(ev, open(gdir / f"round{r}_eval.json", "w"))
            hm = heldout_metrics(model, held, blp)
            row = {"name": a.name, "ckpt": a.ckpt, "round": r, "games": n, "temperature": a.temperature,
                   "opponents": "+".join(opps), "lr": a.lr, "epochs": a.epochs, "keep_draws": a.keep_draws,
                   "win_rate": round(len(wins) / n, 4), "draw_rate": round(len(draws) / n, 4),
                   "illegal_end_rate": round(sum(g.illegal for g in games) / n, 4),
                   "mean_plies": round(float(np.mean([g.plies for g in games])), 1),
                   "kept_games": len(kept), "kept_tokens": kept_tokens, "train_loss_first": round(first, 4),
                   "train_loss_last": round(last, 4),
                   "greedy_strict_elo": round(ev["rating_strict"]["elo"], 1), "greedy_strict_lo": round(ev["rating_strict"]["ci95"][0], 1),
                   "greedy_strict_hi": round(ev["rating_strict"]["ci95"][1], 1), "greedy_lenient_elo": round(ev["rating"]["elo"], 1),
                   "greedy_lenient_lo": round(ev["rating"]["ci95"][0], 1), "greedy_lenient_hi": round(ev["rating"]["ci95"][1], 1),
                   **{k: round(hm[k], 5) for k in ("legal_mass", "top1_legal", "human_nll", "kl_to_base")},
                   "play_seconds": round(play_s, 1), "total_seconds": round(time.time() - t_start, 1)}
            w.writerow(row)
            f.flush()
            print(json.dumps({k: row[k] for k in ("round", "win_rate", "illegal_end_rate", "kept_games", "train_loss_first",
                                                  "train_loss_last", "greedy_strict_elo", "greedy_lenient_elo", "legal_mass",
                                                  "human_nll", "kl_to_base")}), flush=True)
            d = Path("/workspace/ckpt") / a.name
            save(model, d / f"round{r}.pt")
            save(model, d / "final.pt")
            save_state(model, opt, rng, r, d / "last.pt")


if __name__ == "__main__":
    main()

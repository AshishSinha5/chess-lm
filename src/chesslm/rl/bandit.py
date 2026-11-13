"""Legality bandit (docs/RL_PLAN.md, W4 step 4). One-step RL with no engine: a fail-fast test of the RL loop.

At each position (a random prefix of a human game) the policy samples K moves from the full softmax, with no legal
mask. Reward is 1 if the move is legal and 0 if it is not (a special token is illegal). The loss is the GRPO loss on
one step: advantage = (reward - group mean) / group std, policy-gradient term on the sampled tokens, plus beta times
the k3 estimate of KL(policy || frozen base) (DECISIONS G32).

Metrics on fixed held-out positions (random prefixes of validation games), all exact (no sampling noise):
  legal_mass   probability mass the policy puts on legal moves = the legal rate when sampling at temperature 1.
               This is the number that decides how many games end on an illegal token (DECISIONS G30).
  top1_legal   share of positions where the most likely token is legal (the stage-1 "legal rate").
  human_match  share where the most likely token is the human move. human_nll: loss on the human move (cost, Q5).
  kl_to_base   KL(policy || base), mean over positions, in nats.

Usage: python3 -m chesslm.rl.bandit --ckpt size_xs --name bandit_xs --steps 600 --lr 3e-5 --beta 0.02
Saves /workspace/ckpt/NAME/final.pt in the stage-1 format. Appends rows to results/rl/bandit.csv."""
import argparse
import csv
import time
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn.functional as F

from chesslm.data.dataset import Shard
from chesslm.eval.play import TOK, load_model
from chesslm.tokenizers.uci_move import SPECIALS

V = TOK.vocab_size
DATA = "/workspace/data/lichess_1800"
FIELDS = ["name", "ckpt", "loss", "lr", "beta", "batch", "group", "seed", "step", "positions_seen", "train_legal", "legal_mass",
          "top1_legal", "human_match", "human_nll", "kl_to_base", "seconds"]


def game_ucis(shard, g):
    ids = shard.tokens[shard.offs[g]: shard.offs[g + 1]]
    return [TOK.itos[int(i)] for i in ids if i >= len(SPECIALS)]


def legal_mask(board):
    """Bool vector over the vocabulary: True for the tokens that are legal moves in this position."""
    m = torch.zeros(V, dtype=torch.bool)
    m[[TOK.stoi[mv.uci()] for mv in board.legal_moves]] = True
    return m


def make_batch(shard, game_ids, rng):
    """One random position per game: the history before a random move, the legal mask, and the human move."""
    hist, human, masks = [], [], []
    for g in game_ids:
        u = game_ucis(shard, int(g))
        k = int(rng.integers(0, len(u)))
        board = chess.Board()
        for m in u[:k]:
            board.push_uci(m)
        hist.append(u[:k])
        human.append(TOK.stoi[u[k]])
        masks.append(legal_mask(board))
    return hist, torch.tensor(human), torch.stack(masks)


def next_token_logits(model, hist, device="cuda", amp=True):
    """Logits for the next token after each history. Gradients flow if the caller does not disable them."""
    seqs = [[TOK.bos] + [TOK.stoi[m] for m in h] for h in hist]
    T = max(len(s) for s in seqs)
    x = torch.full((len(seqs), T), TOK.pad, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.tensor(s)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and str(device).startswith("cuda")):
        logits = model(x.to(device))[0]
    last = torch.tensor([len(s) - 1 for s in seqs], device=logits.device)
    return logits[torch.arange(len(seqs), device=logits.device), last].float()


def group_advantage(r, eps=1e-4):
    """(reward - group mean) / group std, per row. A row where all rewards are equal gets advantage 0."""
    return (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + eps)


def train_step(model, base, opt, hist, mask, K, beta, gen, device="cuda", amp=True, clip=1.0, loss_type="grpo"):
    """One bandit update on a batch of positions. Returns (mean reward, mean KL to the base model).
    loss_type "grpo": the RL loss described in the module docstring (sampled moves, group advantage, k3 KL).
    loss_type "exact": -log(legal mass) with the exact KL. It uses the legal mask on all tokens, so it is not RL.
    It is a comparison: how much of the cost of the bandit comes from learning by sampling."""
    mask = mask.to(device)
    logp = F.log_softmax(next_token_logits(model, hist, device, amp), dim=-1)
    with torch.no_grad():
        base_full = F.log_softmax(next_token_logits(base, hist, device, amp), dim=-1)
    if loss_type == "exact":
        r = logp.masked_fill(~mask, float("-inf")).logsumexp(-1)  # log legal mass
        kl = (logp.exp() * (logp - base_full)).sum(-1).mean()
        loss = -r.mean() + beta * kl
        reward = r.exp().mean().item()
    else:
        with torch.no_grad():
            tok = torch.multinomial(logp.exp(), K, replacement=True, generator=gen)  # (B, K) sampled moves
            r = mask.gather(1, tok).float()  # 1 if legal
            adv = group_advantage(r)
            base_logp = base_full.gather(1, tok)
        lp = logp.gather(1, tok)
        kl = (torch.exp(base_logp - lp) - (base_logp - lp) - 1).mean()  # k3: KL(policy || base), sampled from the policy
        loss = -(adv * lp).mean() + beta * kl
        reward = r.mean().item()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    opt.step()
    return reward, kl.item()


def heldout(n, seed, split="val"):
    """Fixed evaluation positions: one random prefix of each of n random games of a split."""
    shard = Shard(DATA, split)
    rng = np.random.default_rng(seed)
    games = rng.choice(shard.n, n, replace=False)
    return make_batch(shard, games, rng)


@torch.no_grad()
def evaluate(model, held, base_logp=None, device="cuda", chunk=500, amp=True):
    """Exact held-out metrics (see the module docstring). base_logp: (n, V) log-probs of the base model, or None."""
    hist, human, mask = held
    out = {"legal_mass": 0.0, "top1_legal": 0.0, "human_match": 0.0, "human_nll": 0.0, "kl_to_base": 0.0}
    n = len(hist)
    for i in range(0, n, chunk):
        logp = F.log_softmax(next_token_logits(model, hist[i: i + chunk], device, amp), dim=-1)
        p, m, h = logp.exp(), mask[i: i + chunk].to(device), human[i: i + chunk].to(device)
        top = logp.argmax(-1)
        out["legal_mass"] += (p * m).sum().item()
        out["top1_legal"] += m.gather(1, top[:, None]).float().sum().item()
        out["human_match"] += (top == h).float().sum().item()
        out["human_nll"] += -logp.gather(1, h[:, None]).sum().item()
        if base_logp is not None:
            out["kl_to_base"] += (p * (logp - base_logp[i: i + chunk])).sum().item()
    return {k: v / n for k, v in out.items()}


@torch.no_grad()
def base_logprobs(base, held, device="cuda", chunk=500, amp=True):
    hist = held[0]
    return torch.cat([F.log_softmax(next_token_logits(base, hist[i: i + chunk], device, amp), dim=-1)
                      for i in range(0, len(hist), chunk)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="start checkpoint: run name under /workspace/ckpt, or a .pt path")
    p.add_argument("--name", required=True, help="output run name: /workspace/ckpt/NAME/final.pt")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch", type=int, default=256, help="positions per step")
    p.add_argument("--group", type=int, default=16, help="sampled moves per position")
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--beta", type=float, default=0.02, help="weight of the KL penalty to the base model")
    p.add_argument("--loss", default="grpo", choices=["grpo", "exact"], help="exact: -log legal mass (not RL, a comparison)")
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--eval_n", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(Path(__file__).parents[3] / "results/rl/bandit.csv"))
    p.add_argument("--no_save", action="store_true")
    a = p.parse_args()
    device = "cuda"
    torch.manual_seed(a.seed)
    model, base = load_model(a.ckpt), load_model(a.ckpt)
    for q in base.parameters():
        q.requires_grad_(False)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    shard, rng, gen = Shard(DATA, "train"), np.random.default_rng(a.seed + 1), torch.Generator(device).manual_seed(a.seed)
    held = heldout(a.eval_n, seed=123)
    blp = base_logprobs(base, held)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    t0, window = time.time(), []
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()

        def log(step):
            model.eval()
            m = evaluate(model, held, blp)
            model.train()
            row = {"name": a.name, "ckpt": a.ckpt, "loss": a.loss, "lr": a.lr, "beta": a.beta, "batch": a.batch, "group": a.group,
                   "seed": a.seed, "step": step, "positions_seen": step * a.batch,
                   "train_legal": round(float(np.mean(window)), 4) if window else "",
                   "seconds": round(time.time() - t0, 1), **{k: round(v, 5) for k, v in m.items()}}
            w.writerow(row)
            f.flush()
            print({k: row[k] for k in ("step", "train_legal", "legal_mass", "top1_legal", "human_match", "human_nll",
                                      "kl_to_base", "seconds")}, flush=True)
            window.clear()

        log(0)
        for step in range(1, a.steps + 1):
            hist, _, mask = make_batch(shard, rng.choice(shard.n, a.batch, replace=False), rng)
            r, _ = train_step(model, base, opt, hist, mask, a.group, a.beta, gen, device, loss_type=a.loss)
            window.append(r)
            if step % a.eval_every == 0 or step == a.steps:
                log(step)
    if not a.no_save:
        d = Path("/workspace/ckpt") / a.name
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"model_cfg": model.cfg.__dict__, "model": model.state_dict()}, d / "final.pt")
        print("saved", d / "final.pt")


if __name__ == "__main__":
    main()

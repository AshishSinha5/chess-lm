"""Offline RFT sweep (DECISIONS G35). Round 1 of RFT starts from the base model, so its rollouts do not depend on the
learning rate. This script reuses the saved round-1 rollouts of an RFT run (the kept games), fine-tunes the base model
on them with other settings, and measures the cost (human-move loss, KL to base). No games are played here.
Strength is measured afterwards with `chesslm.rl.evaluate` (see scripts/rl_rft_sweep.sh).

Usage: python3 -m chesslm.rl.rft_offline --ckpt size_m --rollouts results/rl/rft_m/round1_rollouts.jsonl \\
         --tag lr3e-5_ep2 --lr 3e-5 --epochs 2
Saves /workspace/ckpt/rft_sweep/TAG/final.pt and appends a row to results/rl/rft_sweep.csv."""
import argparse
import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from chesslm.eval.play import load_model
from chesslm.rl.bandit import base_logprobs, evaluate as heldout_metrics, heldout
from chesslm.rl.rft import finetune, save

FIELDS = ["tag", "ckpt", "rollouts", "lr", "epochs", "batch", "seed", "kept_games", "kept_tokens", "steps", "loss_first",
          "loss_last", "loss_mean_last10", "legal_mass", "top1_legal", "human_nll", "kl_to_base", "seconds"]


def load_kept(path):
    """Kept games of a rollouts file as light objects with the two fields the SFT loss needs."""
    rows = [json.loads(l) for l in open(path)]
    return [SimpleNamespace(tokens=r["tokens"], model_pos=r["model_pos"]) for r in rows if r["kept"]], len(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--rollouts", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--held_n", type=int, default=2000)
    p.add_argument("--out", default=str(Path(__file__).parents[3] / "results/rl/rft_sweep.csv"))
    a = p.parse_args()
    device = "cuda"
    torch.manual_seed(a.seed)
    t0 = time.time()
    kept, n_all = load_kept(a.rollouts)
    model, base = load_model(a.ckpt), load_model(a.ckpt)
    held = heldout(a.held_n, seed=123)
    blp = base_logprobs(base, held)
    del base
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    steps = finetune(model, opt, kept, a.epochs, a.batch, np.random.default_rng(a.seed), device)
    hm = heldout_metrics(model, held, blp)
    ls = [s["loss"] for s in steps]
    row = {"tag": a.tag, "ckpt": a.ckpt, "rollouts": a.rollouts, "lr": a.lr, "epochs": a.epochs, "batch": a.batch,
           "seed": a.seed, "kept_games": len(kept), "kept_tokens": sum(len(g.model_pos) for g in kept), "steps": len(steps),
           "loss_first": round(ls[0], 4), "loss_last": round(ls[-1], 4), "loss_mean_last10": round(float(np.mean(ls[-10:])), 4),
           **{k: round(hm[k], 5) for k in ("legal_mass", "top1_legal", "human_nll", "kl_to_base")},
           "seconds": round(time.time() - t0, 1)}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    save(model, Path("/workspace/ckpt/rft_sweep") / a.tag / "final.pt")
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()

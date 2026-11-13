"""Rollout speed test (docs/RL_PLAN.md, W4): games per second against Stockfish for engine counts and modes.
Writes results/rl/rollout_speed.csv (one row per config; rows are appended, so reruns add data).
Usage: python3 scripts/rollout_speed.py [--ckpt size_m] [--engines 8,11,16] [--modes sample,greedy_legal] [--games 120]
Keep --engines at or below the usable CPU count (cgroup cpu.max), or Stockfish gets weaker (see EnginePool)."""
import argparse
import csv
import os
import time
from datetime import date
from pathlib import Path

import torch

from chesslm.eval.play import load_model
from chesslm.rl.rollout import EnginePool, make_jobs, openings, play_games, summarize

OUT = Path(__file__).parents[1] / "results/rl/rollout_speed.csv"
FIELDS = ["date", "impl", "ckpt", "mode", "engines", "batch", "games", "seconds", "games_per_s", "plies_per_s", "mean_plies",
          "illegal_end_rate", "score", "cpu_quota", "sf_nodes_per_move"]


def cpu_quota():
    try:
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()
        return round(int(q) / int(p), 1)
    except Exception:
        return os.cpu_count()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="size_m")
    p.add_argument("--engines", default="8,11,16")
    p.add_argument("--modes", default="sample,greedy_legal")
    p.add_argument("--games", type=int, default=120, help="games per config (openings x 2 colors x opponents)")
    p.add_argument("--opponents", default="elo:1320,elo:1500,elo:1700")
    p.add_argument("--batch_per_engine", type=int, default=4)
    p.add_argument("--nodes", type=int, default=0, help="nodes per Stockfish move; 0: 50 ms per move")
    p.add_argument("--impl", default="async", help="label of the rollout loop in the CSV: sync_ticks (first version) or async")
    a = p.parse_args()
    model = load_model(a.ckpt)
    opps = a.opponents.split(",")
    n_open = max(1, a.games // (2 * len(opps)))
    jobs = make_jobs(openings(n_open, seed=7, split="val"), opps)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    new = not OUT.exists()
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        for mode in a.modes.split(","):
            for n in map(int, a.engines.split(",")):
                batch = a.batch_per_engine * n
                with EnginePool(n, nodes=a.nodes or None) as pool:
                    play_games(model, jobs[: 2 * n], pool, mode=mode, batch=batch)  # warm up: start the engines
                    torch.cuda.synchronize()
                    t0 = time.time()
                    games = play_games(model, jobs, pool, mode=mode, batch=batch, seed=1)
                    dt = time.time() - t0
                s = summarize(games)
                plies = sum(g.plies for g in games)
                row = {"date": date.today().isoformat(), "impl": a.impl, "ckpt": a.ckpt, "mode": mode, "engines": n, "batch": batch,
                       "games": len(games), "seconds": round(dt, 1), "games_per_s": round(len(games) / dt, 2),
                       "plies_per_s": round(plies / dt, 1), "mean_plies": round(plies / len(games), 1),
                       "illegal_end_rate": round(sum(g.illegal for g in games) / len(games), 3),
                       "score": round(sum(g.score for g in games) / len(games), 3), "cpu_quota": cpu_quota(),
                       "sf_nodes_per_move": round(s["sf_nodes_per_move"] or 0)}
                w.writerow(row)
                f.flush()
                print(row, {k: round(v["score"], 2) for k, v in s["per_opponent"].items()}, flush=True)


if __name__ == "__main__":
    main()

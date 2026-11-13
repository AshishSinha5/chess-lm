"""Batched evaluation against Stockfish with the rollout engine. Same rules and Elo method as eval/play.py (stage 1).
python3 -m chesslm.rl.evaluate --ckpt size_m --opponents elo:1320,elo:1500 --games 100 --split val --seed 0
Openings: the first 6 plies of `games` random games of `split` (seed + 1, as in stage 1). Each opening is
played with both colors against each opponent. Stage-2 final evals use --split test (docs/RL_PLAN.md).
Output: JSON with per-opponent scores, lenient and strict Elo, and one row per game."""
import argparse
import json
import time
from pathlib import Path

from chesslm.eval.play import load_model
from chesslm.rl.rollout import EnginePool, make_jobs, openings, play_games, summarize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="run name under /workspace/ckpt, or a .pt path")
    p.add_argument("--opponents", default="elo:1320,elo:1500")
    p.add_argument("--games", type=int, default=100, help="openings per opponent; each is played with both colors")
    p.add_argument("--split", default="val")
    p.add_argument("--mode", default="greedy_legal", choices=["greedy_legal", "sample"])
    p.add_argument("--engines", type=int, default=10, help="keep below the usable CPU count (11.9 on this pod)")
    p.add_argument("--batch", type=int, default=0, help="games in play at once (0: 4 x engines)")
    p.add_argument("--movetime", type=float, default=0.05, help="seconds per Stockfish move (used if --nodes is 0)")
    p.add_argument("--nodes", type=int, default=0, help="nodes per Stockfish move; 0: use --movetime (stage-1 rule)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    model = load_model(a.ckpt)
    jobs = make_jobs(openings(a.games, a.seed + 1, a.split), a.opponents.split(","))
    t0 = time.time()
    with EnginePool(a.engines, movetime=a.movetime, nodes=a.nodes or None) as pool:
        games = play_games(model, jobs, pool, mode=a.mode, batch=a.batch or 4 * a.engines, seed=a.seed)
    res = {"ckpt": a.ckpt, "mode": a.mode, "split": a.split, "seed": a.seed, "engines": a.engines,
           "movetime_s": a.movetime, "nodes_limit": a.nodes, "seconds": round(time.time() - t0, 1), **summarize(games)}
    print({k: (round(v["score"], 3), round(v["strict_score"], 3)) for k, v in res["per_opponent"].items()},
          "lenient", res["rating"], "strict", res["rating_strict"], f"{res['seconds']} s",
          f"Stockfish {res['sf_nodes_per_move']:.0f} nodes/move", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()

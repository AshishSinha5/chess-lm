"""Worked example for docs/PROBING.md: the true board and the probe board for one test game.
  python -m chesslm.probe.example --ckpt size_m --layer 8
Runs on the CPU from the cache of `run.py layers`. It refits the linear probe at one layer (trained on the 80% train games),
then picks the test game that is most typical, not the best: the game whose mean number of wrong squares per position
is closest to the mean over all test positions. Among ties it prefers games with 5 positions.
Output: results/probe/example_game.csv (one row per position and square) and example_game_moves.csv (the moves).
make_figures.py reads these files."""
import argparse
import csv

import numpy as np
import torch

from chesslm.probe import linear, run
from chesslm.probe.data import TOK, split_by_game

NAMES = ["empty"] + [c + p for c in "wb" for p in "PNBRQK"]
SQ = [f"{f}{r}" for r in range(1, 9) for f in "abcdefgh"]  # a1 = 0, h8 = 63


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="size_m")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--per_game", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4, help="keep this low while a GPU run trains")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    c = torch.load(run.CACHE / a.ckpt / f"cache_{a.games}x{a.per_game}_s{a.seed}.pt", weights_only=False)
    pos, H = c["pos"], c["hidden"]
    tr = split_by_game(pos)
    Y = pos["labels"].astype(np.int64)
    X = H[a.layer].float()
    pred = linear.predict(linear.fit(X[tr], Y[tr], device="cpu"), X[~tr], device="cpu")
    m = linear.metrics(pred, Y[~tr])
    print(f"check against probe_by_layer.csv: acc {m['acc']:.4f}, bal_acc {m['bal_acc']:.4f}, board_match {m['board_match']:.4f}")

    te = np.flatnonzero(~tr)
    wrong = (pred != Y[~tr]).sum(1)  # wrong squares per test position
    games = pos["game"][te]
    target = wrong.mean()
    best = None
    for g in np.unique(games):
        sel = games == g
        score = (abs(wrong[sel].mean() - target), -int(sel.sum()))
        if best is None or score < best[0]:
            best = (score, g)
    g = best[1]
    sel = np.flatnonzero(games == g)
    order = sel[np.argsort(pos["k"][te][sel])]
    print(f"test game {g}: {len(order)} positions, wrong squares {wrong[order].tolist()}, "
          f"test mean {target:.2f} per position")

    rows = []
    for i in order:
        k = int(pos["k"][te][i])
        for s in range(64):
            rows.append({"model": a.ckpt, "layer": a.layer, "game": int(g), "ply": k, "square": SQ[s],
                         "true": NAMES[Y[te][i][s]], "probe": NAMES[pred[i][s]]})
    run.write_rows("example_game.csv", ["model", "layer", "game", "ply", "square", "true", "probe"], rows, a.ckpt)
    moves = TOK.decode(pos["games"][g].tolist())
    kmax = int(pos["k"][te][order].max())
    path = run.OUT / "example_game_moves.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "ply", "uci"])
        w.writerows([[a.ckpt, i + 1, u] for i, u in enumerate(moves[:kmax])])
    print(f"{kmax} moves -> {path}")


if __name__ == "__main__":
    main()

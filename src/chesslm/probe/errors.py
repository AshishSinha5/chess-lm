"""Probe errors by piece type (docs/PROBING.md; PLAN, Probing). Runs on the CPU from the cache of `run.py layers`.
  python -m chesslm.probe.errors --ckpt size_m --layer 8
Refits the linear probe at one layer and the move-history control, then splits the test errors by true class.
Error kinds: wrong piece (a piece, but the wrong class), missed piece (predicts empty), ghost piece (a piece on an empty square).
Output: results/probe/errors_by_piece.csv, ghosts.csv and confusion.csv. Rows for the same checkpoint are replaced."""
import argparse

import numpy as np
import torch

from chesslm.probe import linear, run
from chesslm.probe.data import N_CLASS, UNTOUCHED, VACATED, split_by_game

NAMES = ["empty"] + [c + p for c in "wb" for p in "PNBRQK"]


def by_piece(src, layer, pred, Y):
    """One row per class: recall and the share of each error kind. For empty, 'ghost' is the share shown as a piece."""
    rows = []
    for c in range(N_CLASS):
        t = Y == c
        p = pred[t]
        rows.append({"source": src, "layer": layer, "piece": NAMES[c], "n": int(t.sum()), "recall": float(np.mean(p == c)),
                     "wrong_piece": float(np.mean((p != c) & (p != 0))) if c else 0.0,
                     "missed": float(np.mean(p == 0)) if c else 0.0,
                     "ghost": 0.0 if c else float(np.mean(p != 0)),
                     "precision": float(np.mean(Y[pred == c] == c)) if np.any(pred == c) else float("nan")})
    return rows


def ghosts(src, layer, pred, Y, hist):
    """Ghost rate on empty squares, split by move history: no move touched the square, or the last event left it empty."""
    rows = []
    for name, state in [("untouched", UNTOUCHED), ("vacated", VACATED)]:
        e = (Y == 0) & (hist == state)
        rows.append({"source": src, "layer": layer, "square": name, "n": int(e.sum()), "ghost": float(np.mean(pred[e] != 0))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="size_m")
    ap.add_argument("--layer", type=int, default=8, help="best layer of the linear probe (probe_by_layer.csv)")
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--per_game", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu keeps the GPU free for training runs")
    args = ap.parse_args()
    torch.set_num_threads(16)
    path = run.CACHE / args.ckpt / f"cache_{args.games}x{args.per_game}_s{args.seed}.pt"
    c = torch.load(path, weights_only=False)  # made by `run.py layers`; this script does not build it
    pos, H = c["pos"], c["hidden"]
    tr = split_by_game(pos)
    Y = pos["labels"].astype(np.int64)
    Yte, hist = Y[~tr], pos["hist_state"][~tr]
    d = args.device

    X = H[args.layer].float()
    pred = linear.predict(linear.fit(X[tr], Y[tr], device=d), X[~tr], device=d)
    F = linear.history_features(pos, device=d)
    t = torch.as_tensor(tr)
    hist_pred = linear.predict(linear.fit(F[t], Y[tr], standardize=False, device=d), F[~t], device=d)
    print(f"check against probe_by_layer.csv: linear bal {linear.metrics(pred, Yte)['bal_acc']:.4f}, "
          f"move_history bal {linear.metrics(hist_pred, Yte)['bal_acc']:.4f}")

    rows, grows = [], []
    for src, layer, p in [("linear", args.layer, pred), ("move_history", "", hist_pred)]:
        rows += by_piece(src, layer, p, Yte)
        grows += ghosts(src, layer, p, Yte, hist)
    for r in rows:
        print(f"{r['source']:12s} {r['piece']:5s} n {r['n']:7d} recall {r['recall']:.4f} wrong {r['wrong_piece']:.4f} "
              f"missed {r['missed']:.4f} ghost {r['ghost']:.4f}")
    for r in grows:
        print(f"{r['source']:12s} ghost on {r['square']:9s} n {r['n']:7d} {r['ghost']:.4f}")
    for r in rows + grows:
        r["model"] = args.ckpt
    run.write_rows("errors_by_piece.csv", ["model", "source", "layer", "piece", "n", "recall", "wrong_piece", "missed",
                                           "ghost", "precision"], rows, args.ckpt)
    run.write_rows("ghosts.csv", ["model", "source", "layer", "square", "n", "ghost"], grows, args.ckpt)

    conf = np.zeros((N_CLASS, N_CLASS), np.int64)  # rows: true class, columns: predicted class (linear probe)
    np.add.at(conf, (Yte.ravel(), pred.ravel()), 1)
    crow = [{"model": args.ckpt, "layer": args.layer, "true": NAMES[i], **{NAMES[j]: int(conf[i, j]) for j in range(N_CLASS)}}
            for i in range(N_CLASS)]
    run.write_rows("confusion.csv", ["model", "layer", "true"] + NAMES, crow, args.ckpt)


if __name__ == "__main__":
    main()

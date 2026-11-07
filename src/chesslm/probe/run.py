"""Probe a checkpoint for board state (PLAN, Probing; DECISIONS G21).
  python -m chesslm.probe.run layers --ckpt size_m            # probes per layer + all 5 controls
  python -m chesslm.probe.run intervene --ckpt size_m --layers 2,4,6 --seeds 0,1,2 --ranks 32,128
`layers` saves positions and hidden states in /workspace/probe_cache/CKPT/ (not in git); `intervene` reuses them.
Output: results/probe/probe_by_layer.csv, probe_by_ply.csv, controls.csv, intervention.csv.
Rows for the same checkpoint are replaced, rows for other checkpoints are kept."""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from chesslm.eval.play import load_model
from chesslm.model.gpt import GPT
from chesslm.probe import acts, linear
from chesslm.probe.data import relative_labels, sample_positions, split_by_game

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/probe"  # main() can change it with --out
CACHE = Path("/workspace/probe_cache")
PLY_BINS = [1, 11, 21, 41, 61, 81, 257]  # ply buckets for probe_by_ply.csv: 1-10, 11-20, 21-40, ...


def write_rows(name, cols, rows, model):
    """Replace this model's rows in results/probe/NAME, keep the others."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    old = [r for r in csv.DictReader(open(path))] if path.exists() else []
    keep = [r for r in old if r["model"] != model]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(keep + [{c: (round(r[c], 5) if isinstance(r[c], float) else r[c]) for c in cols} for r in rows])
    print(f"{len(rows)} rows -> {path}")


def load_cache(ckpt, args):
    path = CACHE / ckpt / f"cache_{args.games}x{args.per_game}_s{args.seed}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    t = time.time()
    pos = sample_positions(args.games, args.per_game, seed=args.seed)
    print(f"{len(pos['k'])} positions from {len(pos['games'])} games in {time.time() - t:.0f}s")
    model = load_model(ckpt)
    hidden, ce, top1 = acts.collect(model, pos)
    path.parent.mkdir(parents=True, exist_ok=True)
    cache = {"pos": pos, "hidden": hidden, "ce": ce, "top1": top1}
    torch.save(cache, path)
    return cache


def probe_layer(X, Y, tr, mlp=False, seed=0):
    p = linear.fit(X[tr], Y[tr], mlp=mlp, seed=seed)
    pred = linear.predict(p, X[~tr])
    return p, pred, linear.metrics(pred, Y[~tr])


def cmd_layers(args):
    ckpt = args.ckpt
    c = load_cache(ckpt, args)
    pos, H = c["pos"], c["hidden"]
    tr = torch.as_tensor(split_by_game(pos)).numpy()
    Y = pos["labels"].astype(np.int64)
    Yrel = relative_labels(Y, pos["white_to_move"]).astype(np.int64)
    rows, best, preds = [], (-1, None), {}
    for l in range(len(H)):
        for target, Yt in [("absolute", Y), ("relative", Yrel)]:
            for kind in ["linear", "mlp"] if (args.mlp and target == "absolute") else ["linear"]:
                _, pred, m = probe_layer(H[l], Yt, tr, mlp=kind == "mlp")
                rows.append({"model": ckpt, "target": target, "probe": kind, "layer": l, **m})
                print(f"layer {l} {target:8s} {kind:6s} acc {m['acc']:.4f} bal {m['bal_acc']:.4f} board {m['board_match']:.4f}")
                if target == "absolute":
                    preds[(kind, l)] = pred
                    if kind == "linear" and m["bal_acc"] > best[0]:
                        best = (m["bal_acc"], l)
    cols = ["model", "target", "probe", "layer", "acc", "bal_acc", "board_match"]
    write_rows("probe_by_layer.csv", cols, rows, ckpt)

    F = linear.history_features(pos)
    t = torch.as_tensor(tr, device=F.device)
    hist_pred = linear.predict(linear.fit(F[t], Y[tr], standardize=False), F[~t])
    del F

    # accuracy by ply: best layer (linear, and MLP if run) against the move-history control.
    # Shows where the model tracks the board better than bookkeeping, and links to the illegal-move analysis.
    lb = best[1]
    sources = [("linear", lb, preds[("linear", lb)])]
    if ("mlp", lb) in preds:
        sources.append(("mlp", lb, preds[("mlp", lb)]))
    sources.append(("move_history", "", hist_pred))
    ply = pos["k"][~tr]
    rows = []
    for src, layer, pred in sources:
        for a, b in zip(PLY_BINS[:-1], PLY_BINS[1:]):
            sel = (ply >= a) & (ply < b)
            if sel.any():
                rows.append({"model": ckpt, "source": src, "layer": layer, "ply_from": a, "ply_to": b - 1,
                             "n": int(sel.sum()), **linear.metrics(pred[sel], Y[~tr][sel])})
    write_rows("probe_by_ply.csv", ["model", "source", "layer", "ply_from", "ply_to", "n", "acc", "bal_acc", "board_match"],
               rows, ckpt)

    # controls (absolute target, linear probe)
    rows = [{"model": ckpt, "control": "majority_class", "layer": "", **linear.majority(Y[tr], Y[~tr])}]
    rows.append({"model": ckpt, "control": "embedding_only", "layer": 0, **probe_layer(H[0], Y, tr)[2]})
    rows.append({"model": ckpt, "control": "move_history", "layer": "", **linear.metrics(hist_pred, Y[~tr])})
    rng = np.random.default_rng(args.seed)
    Yshuf = Y.copy()
    Yshuf[tr] = Y[tr][rng.permutation(int(tr.sum()))]  # whole boards go to the wrong positions
    for l in range(len(H)):
        p = linear.fit(H[l][tr], Yshuf[tr])
        rows.append({"model": ckpt, "control": "shuffled_labels", "layer": l, **linear.metrics(linear.predict(p, H[l][~tr]), Y[~tr])})
    torch.manual_seed(args.seed)
    trained = load_model(ckpt)
    rand = GPT(trained.cfg).cuda().eval()
    del trained
    Hr, _, _ = acts.collect(rand, pos)
    for l in range(len(Hr)):
        rows.append({"model": ckpt, "control": "random_init", "layer": l, **probe_layer(Hr[l], Y, tr)[2]})
    for r in rows:
        print(f"{r['control']:16s} layer {str(r['layer']):2s} bal {r['bal_acc']:.4f} board {r['board_match']:.4f}")
    write_rows("controls.csv", ["model", "control", "layer", "acc", "bal_acc", "board_match"], rows, ckpt)


def projector_hook(Q, mu):
    """Forward hook: remove the span of the orthonormal columns of Q from the block output, around the mean mu."""
    def hook(_, __, out):
        return out - ((out.float() - mu) @ Q @ Q.T).to(out.dtype)
    return hook


def matched_directions(cov, Z, target):
    """Random orthonormal directions with the same variance as the board directions (DECISIONS G26).
    cov: (d, d) covariance of the hidden state. Z: (d, rank) standard normal. target: variance of the board directions,
    sum of q^T cov q over the rank directions. Draw Q = orth(cov^(t/2) Z). t = 0 is plain random directions.
    A larger t leans towards high-variance directions. Bisect t until the variance matches the target.
    If even t = 4 stays below the target, use t = 4 and report it."""
    evals, evecs = torch.linalg.eigh(cov)
    evals = evals.clamp(min=1e-8)

    def build(t):
        Q = torch.linalg.qr(evecs @ ((evals ** (t / 2))[:, None] * (evecs.T @ Z)))[0]
        return Q, float(torch.einsum("dr,de,er->", Q, cov, Q))

    lo, hi = 0.0, 4.0
    Q, v = build(lo)
    if v >= target:
        return Q, v, lo
    Q, v = build(hi)
    if v <= target:
        return Q, v, hi
    for _ in range(40):
        mid = (lo + hi) / 2
        Q, v = build(mid)
        lo, hi = (mid, hi) if v < target else (lo, mid)
    Q, v = build((lo + hi) / 2)
    return Q, v, (lo + hi) / 2


def cmd_intervene(args):
    ckpt = args.ckpt
    c = load_cache(ckpt, args)
    pos, H = c["pos"], c["hidden"]
    tr = split_by_game(pos)
    te = ~tr
    Y = pos["labels"].astype(np.int64)
    model = load_model(ckpt)
    last = model.cfg.n_layer
    test_pos = {k: (v if k == "games" else [v[i] for i in np.flatnonzero(te)] if k == "legal" else v[te])
                for k, v in pos.items()}
    down = linear.fit(H[last][tr], Y[tr])  # board probe at the last layer, trained on clean hidden states

    def evaluate(kind, layer, rank, seed, var_frac=""):
        h, ce, top1 = acts.collect(model, test_pos, layers=[last])
        legal, match = acts.legal_and_match(test_pos, top1)
        m = linear.metrics(linear.predict(down, h[0]), Y[te])
        r = {"model": ckpt, "kind": kind, "layer": layer, "rank": rank, "seed": seed, "ce": float(ce.mean()),
             "legal": legal, "match": match, "down_bal_acc": m["bal_acc"], "down_board_match": m["board_match"],
             "var_frac": var_frac}
        print(" ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()))
        return r

    rows = [evaluate("none", "", 0, "")]
    layers = [int(x) for x in args.layers.split(",")]
    for l in layers:
        assert 1 <= l <= last, "layer must be a block output (1..n_layer)"
        Hl = H[l][tr].float().cuda()
        mu = Hl.mean(0)
        cov = torch.cov((Hl - mu).T)
        total = float(torch.trace(cov))
        del Hl
        for seed in [int(s) for s in args.seeds.split(",")]:
            p = linear.fit(H[l][tr], Y[tr], seed=seed)
            V = torch.linalg.svd(p.raw_directions().float(), full_matrices=False).Vh  # rows: directions by weight
            g = torch.Generator().manual_seed(seed)
            for rank in [int(r) for r in args.ranks.split(",")]:
                Z = torch.randn(model.cfg.d_model, rank, generator=g).cuda()
                Qr = torch.linalg.qr(Z)[0]
                Qp = V[:rank].T.contiguous()
                vp = float(torch.einsum("dr,de,er->", Qp, cov, Qp))
                Qm, vm, t = matched_directions(cov, Z, vp)
                print(f"layer {l} rank {rank} seed {seed}: variance share probe {vp / total:.4f}, "
                      f"random {float(torch.einsum('dr,de,er->', Qr, cov, Qr)) / total:.4f}, matched {vm / total:.4f} (t {t:.2f})")
                for kind, Q, v in [("probe", Qp, vp), ("random", Qr, None), ("random_matched", Qm, vm)]:
                    if v is None:
                        v = float(torch.einsum("dr,de,er->", Q, cov, Q))
                    h = model.blocks[l - 1].register_forward_hook(projector_hook(Q, mu))
                    try:
                        rows.append(evaluate(kind, l, rank, seed, round(v / total, 5)))
                    finally:
                        h.remove()
    cols = ["model", "kind", "layer", "rank", "seed", "ce", "legal", "match", "down_bal_acc", "down_board_match", "var_frac"]
    write_rows("intervention.csv", cols, rows, ckpt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["layers", "intervene"])
    ap.add_argument("--ckpt", default="size_m")
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--per_game", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mlp", action="store_true", help="also train the MLP probe (absolute target)")
    ap.add_argument("--layers", default="2,4,6")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--ranks", default="32,128")
    ap.add_argument("--out", default=None, help="output folder (default results/probe)")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(16)  # the default (one per core) slows a training run on the same machine
    if args.out:
        global OUT
        OUT = Path(args.out)
    {"layers": cmd_layers, "intervene": cmd_intervene}[args.cmd](args)


if __name__ == "__main__":
    main()

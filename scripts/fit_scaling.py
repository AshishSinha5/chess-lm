"""Fit a scaling law to the size and token sweeps. Output: results/scaling_fit.csv and results/scaling_points.csv.
  python scripts/fit_scaling.py
Model: loss = E + A / N^alpha + B / D^beta, with N = parameters and D = training tokens.
Data: the 16 runs in results/token_sweep.csv (4 sizes x 4 token budgets).
Check: the fit never sees full_m and full_l (1.6B tokens). It must predict them.
Second fit: Elo against loss, for the 4 sizes at 1B tokens (results/eval/elo.csv)."""
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
ALPHAS = np.linspace(0.05, 1.5, 146)
BETAS = np.linspace(0.05, 1.5, 146)


def read(name):
    return list(csv.DictReader(open(RES / name)))


def load():
    params = {r["model"]: float(r["params"]) for r in read("runs.csv") if r["run"].startswith("size_")}
    pts = [(params[r["model"]], float(r["tokens_seen"]), float(r["val_loss"]), r["run"]) for r in read("token_sweep.csv")]
    runs = {r["run"]: r for r in read("runs.csv")}
    held = [(params[m], float(runs[f"full_{m}"]["tokens_done"]), float(runs[f"full_{m}"]["final_val_loss"]), f"full_{m}")
            for m in ("m", "l")]
    return pts, held


def fit(pts):
    """Grid over alpha and beta. For each pair, E, A and B come from least squares. Keep only E, A, B >= 0."""
    N, D, y = (np.array(c, dtype=float) for c in zip(*[(p[0], p[1], p[2]) for p in pts]))
    best = None
    for a in ALPHAS:
        for b in BETAS:
            X = np.column_stack([np.ones_like(N), (N / 1e6) ** -a, (D / 1e9) ** -b])
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            if (coef < 0).any():
                continue
            sse = float(((X @ coef - y) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, a, b, *coef)
    return best[1:]  # alpha, beta, E, A, B


def predict(p, N, D):
    a, b, E, A, B = p
    return E + A * (N / 1e6) ** -a + B * (D / 1e9) ** -b


def main():
    pts, held = load()
    p = fit(pts)
    pred = np.array([predict(p, q[0], q[1]) for q in pts])
    true = np.array([q[2] for q in pts])
    rmse = float(np.sqrt(((pred - true) ** 2).mean()))

    rng = np.random.default_rng(0)
    boots = np.array([fit([pts[i] for i in rng.integers(0, len(pts), len(pts))]) for _ in range(200)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)

    rows = [("alpha", p[0], lo[0], hi[0]), ("beta", p[1], lo[1], hi[1]), ("E", p[2], lo[2], hi[2]),
            ("A", p[3], lo[3], hi[3]), ("B", p[4], lo[4], hi[4]), ("rmse_fit", rmse, "", "")]
    with open(RES / "scaling_points.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "params", "tokens", "val_loss", "fit_loss", "split"])
        for q, pr in zip(pts, pred):
            w.writerow([q[3], int(q[0]), int(q[1]), q[2], round(float(pr), 4), "fit"])
        for q in held:
            pr = predict(p, q[0], q[1])
            w.writerow([q[3], int(q[0]), int(q[1]), q[2], round(float(pr), 4), "held_out"])
            rows.append((f"held_out_error_{q[3]}", float(pr) - q[2], "", ""))

    # Elo against loss: lenient and strict, 4 sizes at 1B tokens. A straight line, so the 4 points give 2 numbers.
    loss = {r["run"].replace("size_", ""): float(r["final_val_loss"]) for r in read("runs.csv") if r["run"].startswith("size_")}
    sv = read("eval/strict_vs_lenient.csv")
    for label, col in (("lenient_published", "elo_published"), ("lenient_rerun", "elo_lenient"), ("strict", "elo_strict")):
        x = np.array([loss[r["model"].replace("size_", "")] for r in sv])
        e = np.array([float(r[col]) for r in sv])
        slope, icpt = np.polyfit(x, e, 1)
        r2 = 1 - ((e - (slope * x + icpt)) ** 2).sum() / ((e - e.mean()) ** 2).sum()
        rows += [(f"elo_{label}_slope_per_loss", slope, "", ""), (f"elo_{label}_intercept", icpt, "", ""), (f"elo_{label}_r2", r2, "", "")]

    with open(RES / "scaling_fit.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "value", "ci_low", "ci_high"])
        for n, v, l, h in rows:
            w.writerow([n, f"{v:.4f}", f"{l:.4f}" if l != "" else "", f"{h:.4f}" if h != "" else ""])
    for n, v, l, h in rows:
        print(f"{n:38s} {v:9.4f}  {'' if l == '' else f'({l:.4f}, {h:.4f})'}")


if __name__ == "__main__":
    main()

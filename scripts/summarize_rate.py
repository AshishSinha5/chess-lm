"""Write results/rate_check.csv: val loss against total tokens seen, for each model and rate.
Rate runs are rate_<model>_<lr> and their branches rate_<model>_<lr>_<T>m (DECISIONS G20).
The base rate comes from the size sweep: size_<model> and its branches tok_<model>_<T>m (G16).
Safe to run again: same input gives same output. Usage: python scripts/summarize_rate.py [--ckpt /workspace/ckpt]"""
import argparse
import csv
import json
import re
from pathlib import Path

from summarize_tokens import final_of

ROOT = Path(__file__).resolve().parents[1]
COLS = ["model", "lr", "run", "kind", "tokens_seen", "val_loss"]


def point(d, model, kind):
    """One CSV row for run folder d, or None if the run has no final loss."""
    if not (d / "log.jsonl").exists() or not final_of(d / "log.jsonl"):
        return None
    t, v = final_of(d / "log.jsonl")
    lr = json.load(open(d / "config.json"))["lr"]
    return [model, lr, d.name, kind, t, round(v, 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/ckpt")
    ck = Path(ap.parse_args().ckpt)
    rows, models = [], set()
    for d in sorted(ck.glob("rate_*")):
        m = re.fullmatch(r"rate_([a-z]+)_([0-9.e-]+?)(_\d+m)?", d.name)
        if m and (r := point(d, m.group(1), "branch" if m.group(3) else "full")):
            rows.append(r)
            models.add(m.group(1))
    for model in models:
        for d in [ck / f"size_{model}", *sorted(ck.glob(f"tok_{model}_*m"))]:
            if r := point(d, model, "full" if d.name.startswith("size_") else "branch"):
                rows.append(r)
    if not rows:
        print("no rate check runs in", ck)
        return
    rows.sort(key=lambda r: (r[0], r[1], r[4]))
    out = ROOT / "results/rate_check.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        w.writerows(rows)
    print(f"{len(rows)} points -> {out}")


if __name__ == "__main__":
    main()

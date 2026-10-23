"""Write results/token_sweep.csv: val loss against total tokens seen, for each model.
Branch runs are tok_<model>_<T>m (DECISIONS G16). The full run size_<model> is the 1B point.
Safe to run again: same input gives same output. Usage: python scripts/summarize_tokens.py [--ckpt /workspace/ckpt]"""
import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLS = ["model", "run", "kind", "tokens_seen", "decay_start_tokens", "val_loss"]


def final_of(log):
    """Last line with a final val loss: (tokens, val_loss), or None."""
    last = None
    for line in open(log):
        d = json.loads(line)
        if d.get("final"):
            last = (d["tokens"], d["val_loss"])
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/ckpt")
    ck = Path(ap.parse_args().ckpt)
    rows = []
    for d in sorted(ck.glob("tok_*")):
        m = re.fullmatch(r"tok_(\w+?)_(\d+)m", d.name)
        if not m or not (d / "log.jsonl").exists():
            continue
        fin = final_of(d / "log.jsonl")
        if fin:
            cfg = json.load(open(d / "config.json"))
            rows.append([m.group(1), d.name, "branch", fin[0], round(cfg["max_tokens"] / 1.25), round(fin[1], 4)])
    for model in sorted({r[0] for r in rows}):
        full = ck / f"size_{model}" / "log.jsonl"
        if full.exists() and final_of(full):
            t, v = final_of(full)
            rows.append([model, f"size_{model}", "full", t, round(0.8 * t), round(v, 4)])
    if not rows:
        print("no token sweep runs in", ck)
        return
    rows.sort(key=lambda r: (r[0], r[3]))
    out = ROOT / "results/token_sweep.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        w.writerows(rows)
    print(f"{len(rows)} points -> {out}")


if __name__ == "__main__":
    main()

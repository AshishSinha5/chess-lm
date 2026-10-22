"""Collect every training run into small text files under results/.
Reads CKPT/<run>/config.json and log.jsonl. Writes results/runs.csv, curves/, configs/, data/dataset.json,
and the table between the RESULTS markers in README.md. Safe to run again: same input gives same output.
Usage: python scripts/summarize_runs.py [--ckpt /workspace/ckpt] [--data /workspace/data/lichess_1800]"""
import argparse
import csv
import json
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = [("overfit", "overfit"), ("lr_", "lr_sweep"), ("speed_", "speed"), ("size_", "size_sweep"), ("tok_", "token_sweep")]
RUN_COLS = ["run", "stage", "model", "params", "d_model", "n_layer", "lr", "seed", "max_tokens", "tokens_done",
            "final_val_loss", "best_val_loss", "median_tok_s", "median_mfu", "status", "git_hash", "date"]
CURVE_COLS = ["step", "tokens", "train_loss", "val_loss", "lr", "grad_norm", "tok_s"]
TRAIN_EVERY = 5  # keep every 5th train line in the curve file


def stage_of(run):
    for prefix, stage in STAGES:
        if run.startswith(prefix):
            return stage
    return "other"


def rnd(x, n=5):
    return "" if x is None else round(x, n)


def read_log(path):
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def curve_rows(log):
    out, k = [], 0
    for d in log:
        if "val_loss" in d:
            out.append([d["step"], d["tokens"], "", rnd(d["val_loss"], 6), "", "", ""])
        elif "loss" in d:
            if k % TRAIN_EVERY == 0:
                out.append([d["step"], d["tokens"], rnd(d["loss"], 6), "", rnd(d["lr"], 8),
                            rnd(d["grad_norm"], 4), round(d["tok_s"])])
            k += 1
    return out


def summarize(run_dir):
    run = run_dir.name
    cfg = json.loads((run_dir / "config.json").read_text())
    log = read_log(run_dir / "log.jsonl")
    train = [d for d in log if "loss" in d]
    vals = [d for d in log if "val_loss" in d]
    final = [d for d in vals if d.get("final")]
    mc = cfg.get("model_cfg", {})
    tok = [d["tok_s"] for d in train]
    mfu = [d["mfu"] for d in train]
    tokens_done = final[0]["tokens"] if final else (train[-1]["tokens"] if train else 0)
    date = datetime.fromtimestamp((run_dir / "config.json").stat().st_mtime, timezone.utc).strftime("%Y-%m-%d")
    row = {
        "run": run, "stage": stage_of(run), "model": cfg.get("model", ""), "params": cfg.get("n_params", ""),
        "d_model": mc.get("d_model", ""), "n_layer": mc.get("n_layer", ""), "lr": cfg.get("lr", ""),
        "seed": cfg.get("seed", ""), "max_tokens": cfg.get("max_tokens", ""), "tokens_done": tokens_done,
        "final_val_loss": rnd(final[0]["val_loss"], 4) if final else "",
        "best_val_loss": rnd(min(d["val_loss"] for d in vals), 4) if vals else "",
        "median_tok_s": round(statistics.median(tok)) if tok else "",
        "median_mfu": rnd(statistics.median(mfu), 4) if mfu else "",
        "status": "done" if final else "running", "git_hash": str(cfg.get("git", ""))[:7], "date": date,
    }
    return row, curve_rows(log)


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def table_md(rows):
    head = "| Stage | Run | Model | Params | lr | Seed | Tokens | Val loss | Status |\n|---|---|---|---|---|---|---|---|---|\n"
    lines = []
    for r in rows:
        val = r["final_val_loss"] if r["status"] == "done" else f"{r['best_val_loss']} (so far)"
        params = f"{float(r['params']) / 1e6:.1f}M" if r["params"] != "" else ""
        lines.append(f"| {r['stage']} | {r['run']} | {r['model']} | {params} | {r['lr']} | {r['seed']} | "
                     f"{float(r['tokens_done']) / 1e9:.3f}B | {val} | {r['status']} |")
    return head + "\n".join(lines)


def update_readme(text):
    path = ROOT / "README.md"
    a, b = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    s = path.read_text()
    if a not in s or b not in s:
        print("README.md has no RESULTS markers; skipped")
        return
    head, rest = s.split(a, 1)
    _, tail = rest.split(b, 1)
    path.write_text(f"{head}{a}\n{text}\n{b}{tail}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/workspace/ckpt")
    p.add_argument("--data", default="/workspace/data/lichess_1800")
    a = p.parse_args()
    out = ROOT / "results"
    rows = []
    for d in sorted(Path(a.ckpt).iterdir()):
        if not (d / "config.json").exists() or not (d / "log.jsonl").exists():
            continue
        row, curve = summarize(d)
        rows.append(row)
        write_csv(out / "curves" / f"{row['run']}.csv", CURVE_COLS, curve)
        (out / "configs").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(d / "config.json", out / "configs" / f"{row['run']}.json")
    order = {s: i for i, (_, s) in enumerate(STAGES)}
    sizes = {"xs": 0, "s": 1, "m": 2, "l": 3}
    rows.sort(key=lambda r: (order.get(r["stage"], 99), sizes.get(r["model"], 9), float(r["lr"] or 0), r["seed"], r["run"]))
    write_csv(out / "runs.csv", RUN_COLS, [[r[c] for c in RUN_COLS] for r in rows])
    meta = Path(a.data) / "meta.json"
    if meta.exists():
        (out / "data").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(meta, out / "data" / "dataset.json")
    update_readme(table_md(rows))
    print(f"{len(rows)} runs ({sum(r['status'] == 'running' for r in rows)} running) -> {out}")


if __name__ == "__main__":
    main()

"""Build results/rl/runs.csv: one row per RL run, from the per-run files under results/rl/.
Sources: bandit.csv (last row of each run), eval json files (base_check, calibration, temp_test, bandit_games),
rft.csv (last round of each run). Run it again after each stage-2 step; it rewrites the whole file.
Usage: python3 scripts/summarize_rl_runs.py"""
import csv
import datetime
import json
from pathlib import Path

RL = Path(__file__).resolve().parents[1] / "results" / "rl"
FIELDS = ["run", "method", "start_ckpt", "mode", "temperature", "lr", "beta", "seed", "steps_or_rounds", "games",
          "strict_elo", "strict_lo", "strict_hi", "lenient_elo", "lenient_lo", "lenient_hi", "legal_mass",
          "human_nll", "kl_to_base", "status", "source", "date"]


def day(p):
    return datetime.datetime.fromtimestamp(Path(p).stat().st_mtime).strftime("%Y-%m-%d")


def rel(p):
    return str(Path(p).relative_to(RL.parents[1]))


def bandit_rows():
    p = RL / "bandit.csv"
    if not p.exists():
        return []
    last = {}
    for r in csv.DictReader(open(p)):
        last[r["name"]] = r
    return [{"run": n, "method": "bandit_" + r["loss"], "start_ckpt": r["ckpt"], "lr": r["lr"], "beta": r["beta"],
             "seed": r["seed"], "steps_or_rounds": r["step"], "legal_mass": r["legal_mass"],
             "human_nll": r["human_nll"], "kl_to_base": r["kl_to_base"], "status": "done", "source": rel(p), "date": day(p)}
            for n, r in last.items()]


def eval_rows():
    rows = []
    for d in ("base_check", "calibration", "temp_test", "bandit_games", "rft_sweep"):
        for p in sorted((RL / d).glob("*.json")):
            try:
                r = json.load(open(p))
                ci, cs = r["rating"], r["rating_strict"]
            except Exception:
                continue
            rows.append({"run": f"{d}/{p.stem}", "method": "eval", "start_ckpt": r.get("ckpt", ""), "mode": r.get("mode", ""),
                         "temperature": r.get("temperature", "") if r.get("mode") == "sample" else "", "seed": r.get("seed", ""), "games": len(r.get("games", [])),
                         "strict_elo": round(cs["elo"], 1), "strict_lo": round(cs["ci95"][0], 1), "strict_hi": round(cs["ci95"][1], 1),
                         "lenient_elo": round(ci["elo"], 1), "lenient_lo": round(ci["ci95"][0], 1), "lenient_hi": round(ci["ci95"][1], 1),
                         "status": "done", "source": rel(p), "date": day(p)})
    return rows


def rft_rows(planned_rounds):
    p = RL / "rft.csv"
    if not p.exists():
        return []
    last, games = {}, {}
    for r in csv.DictReader(open(p)):
        last[r["name"]] = r
        games[r["name"]] = games.get(r["name"], 0) + int(r["games"])
    return [{"run": n, "method": "rft", "start_ckpt": r["ckpt"], "mode": "greedy_legal (eval)", "temperature": r["temperature"],
             "lr": r["lr"], "seed": "", "steps_or_rounds": r["round"], "games": games[n],
             "strict_elo": r["greedy_strict_elo"], "strict_lo": r["greedy_strict_lo"], "strict_hi": r["greedy_strict_hi"],
             "lenient_elo": r["greedy_lenient_elo"], "lenient_lo": r["greedy_lenient_lo"], "lenient_hi": r["greedy_lenient_hi"],
             "legal_mass": r["legal_mass"], "human_nll": r["human_nll"], "kl_to_base": r["kl_to_base"],
             "status": "done" if int(r["round"]) >= planned_rounds else f"partial ({r['round']}/{planned_rounds} rounds)",
             "source": rel(p), "date": day(p)} for n, r in last.items()]


def main():
    rows = bandit_rows() + eval_rows() + rft_rows(planned_rounds=4)
    with open(RL / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {RL / 'runs.csv'}")


if __name__ == "__main__":
    main()

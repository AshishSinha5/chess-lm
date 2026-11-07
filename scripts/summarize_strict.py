"""Strict and lenient results from the evaluation rerun (DECISIONS G28).
Reads results/eval/strict_rerun/NAME.json (made by `python -m chesslm.eval.play --out results/eval/strict_rerun/NAME.json`).
Writes results/eval/strict_vs_lenient.csv: one row per model.
Lenient rule: an illegal top move is skipped, the model plays its best legal move. Strict rule: a game with any illegal top move is a loss.
Both ratings come from the same games. Usage: python scripts/summarize_strict.py"""
import csv
import json
from pathlib import Path

EVAL = Path(__file__).resolve().parents[1] / "results/eval"
COLS = ["model", "legal_rate", "games", "games_with_illegal", "illegal_moves_per_game", "elo_lenient", "lenient_ci_low",
        "lenient_ci_high", "elo_strict", "strict_ci_low", "strict_ci_high", "elo_published"]


def main():
    published = {r["model"]: float(r["elo"]) for r in csv.DictReader(open(EVAL / "elo.csv")) if r["opponent"] == "all"}
    rows = []
    for name in ["size_xs", "size_s", "size_m", "size_l"]:
        p = EVAL / "strict_rerun" / f"{name}.json"
        if not p.exists():
            continue
        r = json.load(open(p))
        m, g = r["match"], r["match"]["games"]
        rows.append([name, round(r["legal"]["legal_rate"], 4), len(g), sum(1 for x in g if x["illegal_moves"]),
                     round(sum(x["illegal_moves"] for x in g) / len(g), 2),
                     round(m["rating"]["elo"]), round(m["rating"]["ci95"][0]), round(m["rating"]["ci95"][1]),
                     round(m["rating_strict"]["elo"]), round(m["rating_strict"]["ci95"][0]), round(m["rating_strict"]["ci95"][1]),
                     round(published[name])])
    with open(EVAL / "strict_vs_lenient.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        w.writerows(rows)
    print(f"{len(rows)} rows -> {EVAL / 'strict_vs_lenient.csv'}")


if __name__ == "__main__":
    main()

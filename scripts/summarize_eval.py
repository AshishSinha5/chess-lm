"""Collect evaluation results into small CSV files.
Reads results/eval/NAME.json (made by `python -m chesslm.eval.play`). Writes in results/eval/:
legal_moves.csv (one row per model), stockfish_games.csv (one row per game), elo.csv (one row per model and opponent).
Safe to run again: same input gives same output. Usage: python scripts/summarize_eval.py"""
import csv
import json
from pathlib import Path

EVAL = Path(__file__).resolve().parents[1] / "results/eval"
LEGAL_COLS = ["model", "positions", "legal_rate", "human_match_rate"]
GAME_COLS = ["model", "opponent", "opponent_elo", "model_is_white", "plies", "score"]
ELO_COLS = ["model", "opponent", "games", "wins", "draws", "losses", "score", "elo", "ci_low", "ci_high", "movetime_s"]


def write(path, cols, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)


def main():
    legal, games, elo = [], [], []
    for p in sorted(EVAL.glob("*.json")):
        r = json.load(open(p))
        name = p.stem
        L = r["legal"]
        legal.append([name, L["positions"], round(L["legal_rate"], 4), round(L["human_match_rate"], 4)])
        m = r.get("match")
        if not m:
            continue
        for g in m["games"]:
            games.append([name, g["opp"], g["elo"] if g["elo"] is not None else "", int(g["white"]), g["plies"], g["score"]])
        for opp, s in m["per_opponent"].items():
            elo.append([name, opp, s["games"], s["wins"], s["draws"], s["losses"], round(s["score"], 4), "", "", "", m["movetime_s"]])
        n = sum(s["games"] for s in m["per_opponent"].values())
        score = sum(g["score"] for g in m["games"]) / n
        rt = m["rating"]
        elo.append([name, "all", n, sum(s["wins"] for s in m["per_opponent"].values()),
                    sum(s["draws"] for s in m["per_opponent"].values()),
                    sum(s["losses"] for s in m["per_opponent"].values()), round(score, 4),
                    round(rt["elo"]) if rt else "", round(rt["ci95"][0]) if rt else "", round(rt["ci95"][1]) if rt else "",
                    m["movetime_s"]])
    if not legal:
        print("no evaluation files in", EVAL)
        return
    write(EVAL / "legal_moves.csv", LEGAL_COLS, legal)
    write(EVAL / "stockfish_games.csv", GAME_COLS, games)
    write(EVAL / "elo.csv", ELO_COLS, elo)
    print(f"{len(legal)} models, {len(games)} games -> {EVAL}")


if __name__ == "__main__":
    main()

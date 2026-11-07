"""Evaluate a checkpoint: legal-move rate, then games against Stockfish, then an Elo estimate.
python -m chesslm.eval.play --ckpt size_m --legal_games 2000 --opponents elo:1320,elo:1500 --games 100
Stockfish path: --stockfish or $STOCKFISH (default /workspace/bin/stockfish). Output: results/eval/NAME.json"""
import argparse
import json
import math
import os
import random
from multiprocessing import get_context
from pathlib import Path

import chess
import chess.engine
import numpy as np
import torch

from chesslm.data.dataset import Shard
from chesslm.model.gpt import GPT, GPTConfig
from chesslm.tokenizers.uci_move import SPECIALS, UCIMoveTokenizer

TOK = UCIMoveTokenizer.canonical()
MAX_PLIES = 256  # the longest game in the training data
OPENING_PLIES = 6
STOCKFISH = os.environ.get("STOCKFISH", "/workspace/bin/stockfish")


def load_model(name, device="cuda"):
    path = Path(name) if name.endswith(".pt") else Path("/workspace/ckpt") / name / "final.pt"
    st = torch.load(path, map_location=device)
    model = GPT(GPTConfig(**st["model_cfg"])).to(device)
    model.load_state_dict(st["model"])
    return model.eval()


@torch.no_grad()
def next_logits(model, histories, device="cuda"):
    """Logits for the next token after each move history (list of UCI lists)."""
    seqs = [[TOK.bos] + [TOK.stoi[m] for m in h] for h in histories]
    T = max(len(s) for s in seqs)
    x = torch.zeros(len(seqs), T, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.tensor(s)
    with torch.autocast(device, dtype=torch.bfloat16):
        logits, _ = model(x.to(device))
    last = torch.tensor([len(s) - 1 for s in seqs], device=device)
    return logits[torch.arange(len(seqs)), last].float()


def choose_move_ex(model, history, board, device="cuda"):
    """Best legal move by model score (greedy), and whether the model's top move over ALL tokens was legal."""
    logits = next_logits(model, [history], device)[0]
    legal = [m.uci() for m in board.legal_moves]
    ids = torch.tensor([TOK.stoi[u] for u in legal], device=device)
    return legal[int(logits[ids].argmax())], bool((ids == logits.argmax()).any())


def choose_move(model, history, board, device="cuda"):
    """Best legal move by model score (greedy)."""
    return choose_move_ex(model, history, board, device)[0]


def val_games(n, seed, split="val"):
    """n random games from the split, as UCI lists."""
    shard = Shard("/workspace/data/lichess_1800", split)
    rng = np.random.default_rng(seed)
    games = []
    for g in rng.choice(shard.n, n, replace=False):
        ids = shard.tokens[shard.offs[g]: shard.offs[g + 1]]
        games.append(TOK.decode([int(i) for i in ids if i != TOK.eos]))
    return games


def legal_rate(model, n_games, seed=0, batch=256, device="cuda"):
    """At one random position per game, the share of top-1 moves (no legal mask) that are legal, and that match the human move."""
    rng = random.Random(seed)
    cases = []
    for g in val_games(n_games, seed):
        k = rng.randrange(0, len(g))
        cases.append((g[:k], g[k]))
    legal = match = 0
    for i in range(0, len(cases), batch):
        part = cases[i: i + batch]
        top = next_logits(model, [h for h, _ in part], device).argmax(-1).tolist()
        for (h, human), t in zip(part, top):
            board = chess.Board()
            for m in h:
                board.push_uci(m)
            mv = TOK.itos[t] if t >= len(SPECIALS) else None  # a special token is an illegal answer
            legal += mv is not None and chess.Move.from_uci(mv) in board.legal_moves
            match += mv == human
    return {"positions": len(cases), "legal_rate": legal / len(cases), "human_match_rate": match / len(cases)}


def parse_opponent(spec):
    kind, val = spec.split(":")
    opts = {"UCI_LimitStrength": True, "UCI_Elo": int(val)} if kind == "elo" else {"Skill Level": int(val)}
    return {"spec": spec, "options": opts, "elo": int(val) if kind == "elo" else None}


_W = {}


def _init_worker(ckpt, movetime, stockfish):
    _W["model"] = load_model(ckpt)
    _W["movetime"] = movetime
    _W["engine"] = chess.engine.SimpleEngine.popen_uci(stockfish)
    _W["engine"].configure({"Threads": 1, "Hash": 16})


def _play(job):
    """Play one game. job = (opening, model_is_white, opponent). Returns the score for the model: 1, 0.5 or 0."""
    opening, white, opp = job
    engine, model = _W["engine"], _W["model"]
    engine.configure({"UCI_LimitStrength": False, "Skill Level": 20})
    engine.configure(opp["options"])
    board, hist = chess.Board(), []
    for m in opening:
        board.push_uci(m); hist.append(m)
    n_illegal, first_illegal = 0, None  # strict rule: the model's raw top move must be legal (DECISIONS F2, G28)
    while not board.is_game_over(claim_draw=True) and len(hist) < MAX_PLIES:
        if board.turn == white:
            mv, raw_ok = choose_move_ex(model, hist, board)
            if not raw_ok:
                n_illegal += 1
                first_illegal = len(hist) if first_illegal is None else first_illegal
        else:
            mv = engine.play(board, chess.engine.Limit(time=_W["movetime"])).move.uci()
        board.push_uci(mv); hist.append(mv)
    res = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "1/2-1/2"
    score = {"1-0": 1.0, "0-1": 0.0}.get(res, 0.5)
    score = score if white else 1 - score
    return {"opp": opp["spec"], "white": white, "plies": len(hist), "score": score, "illegal_moves": n_illegal,
            "first_illegal_ply": first_illegal, "strict_score": 0.0 if n_illegal else score}


def expected(r, opp_elo):
    return 1 / (1 + 10 ** ((opp_elo - r) / 400))


def fit_elo(scores, opp_elos, lo=-500, hi=4000):
    """Rating r where the expected total score equals the real total (monotonic, so bisect)."""
    s, o = np.array(scores), np.array(opp_elos, dtype=float)
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if (s - 1 / (1 + 10 ** ((o - mid) / 400))).sum() > 0 else (lo, mid)
    return (lo + hi) / 2


def elo_with_ci(games, n_boot=1000, seed=0):
    g = [x for x in games if x["elo"] is not None]
    if not g:
        return None
    s, o = np.array([x["score"] for x in g]), np.array([x["elo"] for x in g])
    rng = np.random.default_rng(seed)
    boots = [fit_elo(s[i], o[i]) for i in (rng.integers(0, len(g), len(g)) for _ in range(n_boot))]
    return {"elo": fit_elo(s, o), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "note": "Nominal Stockfish UCI_Elo scale; a bound is returned if the score is 0% or 100%."}


def play_match(ckpt, opponents, games, workers, movetime, stockfish, seed=0):
    opps = [parse_opponent(s) for s in opponents]
    rng = random.Random(seed)
    openings = [g[:OPENING_PLIES] for g in val_games(games, seed + 1)]
    jobs = [(op, white, o) for o in opps for op in openings for white in (True, False)]
    rng.shuffle(jobs)
    with get_context("spawn").Pool(workers, _init_worker, (ckpt, movetime, stockfish)) as pool:
        out = pool.map(_play, jobs, chunksize=2)
    elo_of = {o["spec"]: o["elo"] for o in opps}
    for r in out:
        r["elo"] = elo_of[r["opp"]]
    per = {}
    for o in opps:
        sc = [r["score"] for r in out if r["opp"] == o["spec"]]
        per[o["spec"]] = {"games": len(sc), "score": sum(sc) / len(sc), "wins": sc.count(1.0), "draws": sc.count(0.5),
                          "losses": sc.count(0.0)}
    for o in opps:
        per[o["spec"]]["strict_score"] = float(np.mean([r["strict_score"] for r in out if r["opp"] == o["spec"]]))
        per[o["spec"]]["games_with_illegal"] = sum(1 for r in out if r["opp"] == o["spec"] and r["illegal_moves"])
    strict = [dict(r, score=r["strict_score"]) for r in out]
    return {"movetime_s": movetime, "per_opponent": per, "rating": elo_with_ci(out), "rating_strict": elo_with_ci(strict),
            "games": out}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="run name under /workspace/ckpt, or a .pt path")
    p.add_argument("--legal_games", type=int, default=2000)
    p.add_argument("--opponents", default="", help="comma list: elo:1320,skill:0")
    p.add_argument("--games", type=int, default=100, help="openings per opponent; each is played with both colors")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--movetime", type=float, default=0.05, help="seconds per Stockfish move")
    p.add_argument("--stockfish", default=STOCKFISH)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    name = Path(a.ckpt).parent.name if a.ckpt.endswith(".pt") else a.ckpt
    res = {"ckpt": a.ckpt, "legal": legal_rate(load_model(a.ckpt), a.legal_games, a.seed)}
    print(res["legal"], flush=True)
    if a.opponents:
        res["match"] = play_match(a.ckpt, a.opponents.split(","), a.games, a.workers, a.movetime, a.stockfish, a.seed)
        print(res["match"]["per_opponent"], res["match"]["rating"], flush=True)
    out = Path(a.out or Path(__file__).parents[3] / "results/eval" / f"{name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()

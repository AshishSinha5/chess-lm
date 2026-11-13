"""Batched rollouts: many games at once against Stockfish or a random mover. Used for RL and for fast evals.

Each game is a small state machine: it waits for a model move, or for an engine move, or it is over.
  - The model moves of all games that are ready now go through one batched GPU forward pass.
  - Each engine move runs in a pool of Stockfish processes. A game goes on as soon as ITS engine answers, so the
    engines never wait for the slowest game (the first version moved all games in lock step, one ply per tick).
  - A finished game frees its slot for the next job (continuous batching).
Each game has its own random generator, seeded from (seed, job index). So the sampled moves do not depend on how
the games were batched. (Stockfish itself is not deterministic: its search depends on time.)

Modes:
  sample        Sample a token from the full softmax (no legal mask). An illegal token ends the game as a loss.
  greedy_legal  Best legal move by score: the stage-1 lenient rule (eval/play.py). Also counts raw top tokens
                that are illegal, so strict and lenient scores come from the same games (DECISIONS G28).

Sequence layout (same as training): tokens = [bos, ply 0, ply 1, ...]. The token at index t is ply t-1 and is
predicted from the logits at position t-1. The model plays white -> its plies are even -> its t are odd.
`model_pos` lists the t of every token the model produced. Opening plies and opponent moves are not in it.
"""
import random
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from multiprocessing import get_context

import chess
import torch
import torch.nn.functional as F

from chesslm.rl import engine_worker
from chesslm.eval.play import MAX_PLIES, STOCKFISH, TOK, elo_with_ci, val_games
from chesslm.eval.play import parse_opponent as _parse_stockfish
from chesslm.tokenizers.uci_move import SPECIALS


def parse_opponent(spec):
    """'elo:1500', 'skill:0' (Stockfish) or 'random' (uniform random legal move, no engine)."""
    if spec == "random":
        return {"spec": "random", "options": None, "elo": None}
    return _parse_stockfish(spec)


def make_jobs(openings, opponents, both_colors=True):
    """One job per (opening, color, opponent). Paired design: every opening is played with both colors."""
    opps = [parse_opponent(o) if isinstance(o, str) else o for o in opponents]
    colors = (True, False) if both_colors else (True,)
    return [(op, white, o) for o in opps for op in openings for white in colors]


def openings(n, seed, split="val", plies=6):
    """First `plies` moves of n random games from a data split. Train on 'train', evaluate on 'test'."""
    return [g[:plies] for g in val_games(n, seed, split)]


@dataclass
class Game:
    opp: dict
    white: bool  # True if the model plays white
    opening: list
    board: chess.Board
    tokens: list  # [bos] + one id per ply; the last id can be an illegal model token
    model_pos: list = field(default_factory=list)
    logp: list = field(default_factory=list)  # log-prob of tokens[t] under the rollout policy, for t in model_pos
    illegal: bool = False  # sample mode: the game ended on an illegal model token
    idx: int = 0  # job index
    rng: object = None  # torch generator for the model's sampling (sample mode)
    prng: random.Random = None  # for the random opponent
    sf_nodes: int = 0  # Stockfish nodes searched in this game, and the number of engine moves
    sf_moves: int = 0
    n_illegal_top: int = 0  # greedy_legal mode: model moves where the raw top token was illegal
    first_illegal_ply: int = None
    over: bool = False
    result: str = ""
    score: float = None  # model's score: 1, 0.5 or 0

    @property
    def plies(self):
        return len(self.tokens) - 1

    def record(self):
        strict = 0.0 if (self.illegal or self.n_illegal_top) else self.score
        return {"opp": self.opp["spec"], "elo": self.opp["elo"], "white": self.white, "plies": self.plies,
                "result": self.result, "score": self.score, "strict_score": strict, "illegal_end": self.illegal,
                "illegal_moves": self.n_illegal_top, "first_illegal_ply": self.first_illegal_ply,
                "sf_nodes": self.sf_nodes, "sf_moves": self.sf_moves,
                "model_moves": len(self.model_pos)}


class EnginePool:
    """n worker processes, each with one persistent Stockfish engine (rl/engine_worker.py). Keep n at or below
    the usable CPU count: with a fixed time per move, an oversubscribed CPU makes Stockfish weaker.
    submit() returns a future whose result is the engine move as a UCI string."""

    def __init__(self, n, stockfish=STOCKFISH, movetime=0.05, nodes=None):
        """nodes: stop each engine move after this many nodes (the opponent does not depend on the CPU load).
        With nodes=None the engine stops after `movetime` seconds, as in stage 1."""
        self.movetime, self.nodes = movetime, nodes
        self._ex = ProcessPoolExecutor(n, mp_context=get_context("spawn"), initializer=engine_worker.init,
                                       initargs=(stockfish, movetime, nodes))

    def submit(self, board, opp):
        return self._ex.submit(engine_worker.move, [m.uci() for m in board.move_stack], opp["spec"], opp["options"])

    def close(self):
        self._ex.shutdown(wait=True)  # a worker exit closes the engine's stdin, so Stockfish quits

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _autocast(device, amp):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and str(device).startswith("cuda"))


@torch.no_grad()
def last_logits(model, seqs, device="cuda", amp=True):
    """Logits after the last token of each id sequence (right padded; the model is causal, so padding is harmless)."""
    T = max(len(s) for s in seqs)
    x = torch.full((len(seqs), T), TOK.pad, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.tensor(s)
    with _autocast(device, amp):
        logits = model(x.to(device))[0]
    last = torch.tensor([len(s) - 1 for s in seqs], device=logits.device)
    return logits[torch.arange(len(seqs), device=logits.device), last].float()


def _new_game(job, idx, seed):
    opening, white, opp = job
    board = chess.Board()
    for m in opening:
        board.push_uci(m)
    key = seed * 1_000_003 + idx
    return Game(opp=opp, white=white, opening=list(opening), board=board,
                tokens=[TOK.bos] + [TOK.stoi[m] for m in opening], idx=idx,
                rng=torch.Generator().manual_seed(key), prng=random.Random(key))


def _push(g, uci):
    g.board.push_uci(uci)
    g.tokens.append(TOK.stoi[uci])


def _finish(g):
    if g.illegal:
        g.result, g.score = "illegal", 0.0
        return
    res = g.board.result(claim_draw=True) if g.board.is_game_over(claim_draw=True) else "1/2-1/2"  # ply cap: draw
    g.result = res if g.board.is_game_over(claim_draw=True) else "max_plies"
    s = {"1-0": 1.0, "0-1": 0.0}.get(res, 0.5)
    g.score = s if g.white else 1.0 - s


def _model_moves(model, games, mode, temperature, device, amp):
    logits = last_logits(model, [g.tokens for g in games], device, amp)
    logp_all = F.log_softmax(logits / temperature, dim=-1)
    if mode == "sample":
        # Gumbel-max: argmax(logits / T + Gumbel noise) is an exact sample from softmax(logits / T).
        # The noise comes from each game's own generator, so a game does not depend on its batch.
        u = torch.stack([torch.rand(logits.shape[1], generator=g.rng) for g in games]).clamp(1e-12, 1 - 1e-7)
        toks = (logits / temperature - torch.log(-torch.log(u)).to(logits.device)).argmax(-1)
    tops = logits.argmax(-1).tolist()
    for i, g in enumerate(games):
        if mode == "sample":
            t = int(toks[i])
            uci = TOK.itos[t] if t >= len(SPECIALS) else None  # a special token is an illegal answer
            move = chess.Move.from_uci(uci) if uci else None
            legal = move is not None and move in g.board.legal_moves
        else:  # greedy_legal: same rule as eval.play.choose_move_ex
            legal_ucis = [m.uci() for m in g.board.legal_moves]
            ids = [TOK.stoi[u] for u in legal_ucis]
            row = logits[i, ids]
            t = ids[int(row.argmax())]
            move, legal = chess.Move.from_uci(TOK.itos[t]), True
            if tops[i] not in ids:
                g.n_illegal_top += 1
                g.first_illegal_ply = g.plies if g.first_illegal_ply is None else g.first_illegal_ply
        g.model_pos.append(len(g.tokens))
        g.tokens.append(t)
        g.logp.append(float(logp_all[i, t]))
        if legal:
            g.board.push(move)
        else:
            g.illegal = g.over = True


def play_games(model, jobs, pool=None, mode="sample", batch=64, temperature=1.0, seed=0, device="cuda", amp=True,
               max_plies=MAX_PLIES):
    """Play all jobs (opening, model_is_white, opponent dict). Returns a list of Game, in job order.
    `pool` (EnginePool) is needed only for Stockfish opponents. `batch` is the number of games in play at once:
    use about 4 times the engine count, so every engine has a queued move while the GPU works."""
    assert mode in ("sample", "greedy_legal")
    assert pool is not None or all(j[2]["spec"] == "random" for j in jobs), "Stockfish opponents need a pool"
    out, active, pending, ready, nxt = [None] * len(jobs), {}, {}, [], 0

    def route(i):
        """Decide what game i needs next: a model move (ready), an engine move (pending), or it is over."""
        g = active[i]
        while True:
            if g.over or g.board.is_game_over(claim_draw=True) or g.plies >= max_plies:
                _finish(g)
                out[i] = active.pop(i)
                return
            if g.board.turn == g.white:
                ready.append(i)
                return
            if g.opp["spec"] == "random":
                _push(g, g.prng.choice(list(g.board.legal_moves)).uci())
                continue
            pending[pool.submit(g.board, g.opp)] = i
            return

    while nxt < len(jobs) or active:
        while len(active) < batch and nxt < len(jobs):
            active[nxt] = _new_game(jobs[nxt], nxt, seed)
            route(nxt)
            nxt += 1
        for f in [f for f in pending if f.done()]:  # engine moves that are ready
            i = pending.pop(f)
            uci, nodes = f.result()
            _push(active[i], uci)
            if nodes:
                active[i].sf_nodes += nodes
                active[i].sf_moves += 1
            route(i)
        if ready:
            idx, ready[:] = list(ready), []
            _model_moves(model, [active[i] for i in idx], mode, temperature, device, amp)
            for i in idx:
                route(i)
        elif pending:
            wait(pending, return_when=FIRST_COMPLETED)  # the next loop turn takes the finished moves
    return out


def sequence_logprobs(model, games, device="cuda", amp=True, temperature=1.0):
    """Teacher-forced log-probs of the model tokens of each game, in the order of game.logp.
    Returns a list of 1-D tensors. Gradients flow if the caller does not disable them (GRPO uses this)."""
    seqs = [g.tokens for g in games]
    T = max(len(s) for s in seqs)
    x = torch.full((len(seqs), T), TOK.pad, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.tensor(s)
    x = x.to(device)
    with _autocast(device, amp):
        logits = model(x)[0]
    lp = F.log_softmax(logits.float() / temperature, dim=-1)
    out = []
    for i, g in enumerate(games):
        pos = torch.tensor(g.model_pos, device=x.device)
        out.append(lp[i, pos - 1, x[i, pos]])
    return out


def nodes_per_move(recs):
    """Mean Stockfish nodes per engine move (None if there was no engine). A drop means a weaker opponent."""
    moves = sum(r["sf_moves"] for r in recs)
    return sum(r["sf_nodes"] for r in recs) / moves if moves else None


def summarize(games):
    """Per-opponent scores and the lenient and strict Elo (same method as stage 1)."""
    recs = [g.record() for g in games]
    per = {}
    for spec in sorted({r["opp"] for r in recs}):
        rs = [r for r in recs if r["opp"] == spec]
        per[spec] = {"games": len(rs), "score": sum(r["score"] for r in rs) / len(rs),
                     "strict_score": sum(r["strict_score"] for r in rs) / len(rs),
                     "illegal_end": sum(r["illegal_end"] for r in rs),
                     "games_with_illegal_top": sum(1 for r in rs if r["illegal_moves"]),
                     "sf_nodes_per_move": nodes_per_move(rs),
                     "mean_plies": sum(r["plies"] for r in rs) / len(rs)}
    strict = [dict(r, score=r["strict_score"]) for r in recs]
    return {"per_opponent": per, "sf_nodes_per_move": nodes_per_move(recs), "rating": elo_with_ci(recs),
            "rating_strict": elo_with_ci(strict), "games": recs}

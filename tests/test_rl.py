"""RL exit tests 1 to 3 (docs/RL_PLAN.md, W4). CPU only: a tiny random model and the random opponent, no Stockfish."""
import random
import time
from concurrent.futures import ThreadPoolExecutor

import chess
import torch
import torch.nn as nn

from chesslm.eval.play import TOK
from chesslm.model.gpt import GPT, GPTConfig
from chesslm.rl.rollout import make_jobs, play_games, sequence_logprobs

torch.set_num_threads(4)  # torch picks threads from nproc, not from the cgroup CPU quota

OPENINGS = [["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"], ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6"],
            ["c2c4", "g8f6"], []]


def tiny(seed=0):
    torch.manual_seed(seed)
    return GPT(GPTConfig(vocab_size=TOK.vocab_size, d_model=32, n_layer=1, n_head=2, block_size=272)).eval()


def run(model, mode, seed=0, opps=("random",)):
    return play_games(model, make_jobs(OPENINGS, list(opps)), mode=mode, batch=8, seed=seed, device="cpu", amp=False,
                      max_plies=40)  # short games keep the test fast; the rules are the same


def check_parity(g):
    """model_pos holds every model ply and only those; the game replays legally."""
    n_open = len(g.opening)
    assert all((t - 1) % 2 == (0 if g.white else 1) for t in g.model_pos)  # ply t-1: white plays even plies
    assert all(t - 1 >= n_open for t in g.model_pos)  # opening plies are never model moves
    mine = [p for p in range(n_open, g.plies) if (p % 2 == 0) == g.white]
    assert [t - 1 for t in g.model_pos] == mine  # every model ply, and only those
    assert len(g.logp) == len(g.model_pos)
    board = chess.Board()
    for t in g.tokens[1:]:
        board.push_uci(TOK.itos[t])  # raises if a move is illegal
    assert not g.illegal and g.score in (0.0, 0.5, 1.0)


def test_1_mask_and_ply_parity():
    """Only model moves are in model_pos, at the right parity for both colors."""
    games = run(tiny(), "greedy_legal")
    assert {g.white for g in games} == {True, False}
    for g in games:
        check_parity(g)


def test_2_rollout_logprob_equals_training_logprob():
    """The log-prob stored during the rollout equals the teacher-forced log-prob of the training forward pass."""
    model = tiny(1)
    for mode in ("greedy_legal", "sample"):
        games = [g for g in run(model, mode, seed=3) if g.model_pos]
        assert games
        with torch.no_grad():
            lps = sequence_logprobs(model, games, device="cpu", amp=False)
        for g, lp in zip(games, lps):
            assert torch.allclose(lp, torch.tensor(g.logp), atol=1e-4), (mode, (lp - torch.tensor(g.logp)).abs().max())


class AlwaysEos(nn.Module):
    """A model whose top token is always <eos>: an illegal answer in any position."""

    def forward(self, idx):
        out = torch.zeros(*idx.shape, TOK.vocab_size)
        out[..., TOK.eos] = 50.0
        return out, None


def test_3_illegal_token_ends_game_as_loss():
    games = run(AlwaysEos(), "sample")
    for g in games:
        assert g.illegal and g.result == "illegal" and g.score == 0.0
        assert g.record()["strict_score"] == 0.0
        assert len(g.model_pos) == 1 and g.tokens[-1] == TOK.eos  # the game stops at the first model move
        assert g.model_pos[0] == len(g.tokens) - 1  # the illegal token is kept, so RL can push it down


def test_greedy_legal_counts_illegal_top_moves():
    """Lenient rule: an illegal top token does not end the game, but it is counted, and the strict score is 0."""
    games = run(AlwaysEos(), "greedy_legal")
    for g in games:
        assert not g.illegal and g.n_illegal_top == len(g.model_pos) > 0
        assert g.record()["strict_score"] == 0.0


class FakePool:
    """Stands in for EnginePool: a random legal move after a random delay, so the answers come out of order."""

    def __init__(self, seed=0):
        self.rng, self.ex = random.Random(seed), ThreadPoolExecutor(8)

    def submit(self, board, opp):
        mv = self.rng.choice([m.uci() for m in board.legal_moves])
        delay = self.rng.random() * 0.01

        def work():
            time.sleep(delay)
            return mv, 1000  # (move, nodes), like the real engine worker

        return self.ex.submit(work)


def test_async_engine_answers_out_of_order():
    """Games go on as their own engine answers. Order of answers is random, so every game must stay consistent."""
    jobs = make_jobs(OPENINGS * 3, ["elo:1320"])
    games = play_games(tiny(2), jobs, FakePool(), mode="greedy_legal", batch=10, device="cpu", amp=False, max_plies=30)
    assert len(games) == len(jobs) and all(g is not None for g in games)
    assert [g.opening for g in games] == [j[0] for j in jobs]  # results are in job order
    for g in games:
        check_parity(g)
        assert g.plies >= len(g.opening) and g.score in (0.0, 0.5, 1.0)


def test_seed_decides_the_games():
    """Same seed: same games, whatever the batch size. Other seed: other games."""
    def toks(seed, batch):
        jobs = make_jobs(OPENINGS * 2, ["random"])
        g = play_games(tiny(3), jobs, mode="sample", batch=batch, seed=seed, device="cpu", amp=False, max_plies=40)
        return [x.tokens for x in g]
    assert toks(5, 3) == toks(5, 16)
    assert toks(5, 16) != toks(6, 16)


class TwoMoves(nn.Module):
    """Logits that give e2e4 probability 0.7 and d2d4 probability 0.3 (nothing else)."""

    def forward(self, idx):
        out = torch.full((*idx.shape, TOK.vocab_size), -1e9)
        out[..., TOK.stoi["e2e4"]] = torch.log(torch.tensor(0.7))
        out[..., TOK.stoi["d2d4"]] = torch.log(torch.tensor(0.3))
        return out, None


def test_sampling_follows_the_softmax():
    """The Gumbel-max draw is an exact sample from softmax(logits): about 70% e2e4 over 600 games."""
    jobs = make_jobs([[]] * 600, ["random"], both_colors=False)
    games = play_games(TwoMoves(), jobs, mode="sample", batch=64, seed=11, device="cpu", amp=False, max_plies=1)
    first = [TOK.itos[g.tokens[1]] for g in games]
    assert set(first) == {"e2e4", "d2d4"}
    assert abs(first.count("e2e4") / 600 - 0.7) < 0.07  # 3 standard errors
    assert all(abs(g.logp[0] - (-0.3567 if TOK.itos[g.tokens[1]] == "e2e4" else -1.2040)) < 1e-3 for g in games)

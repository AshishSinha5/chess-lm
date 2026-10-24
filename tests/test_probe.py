import chess
import numpy as np
import torch

from chesslm.probe import linear
from chesslm.probe.data import board_labels, relative_labels


def test_start_labels():
    y = board_labels(chess.Board())
    assert y[chess.E1] == chess.KING and y[chess.D8] == chess.QUEEN + 6
    assert (y[16:48] == 0).all() and (y[8:16] == 1).all() and (y[48:56] == 7).all()


def test_relative_mirrors_for_black():
    b = chess.Board()
    b.push_uci("e2e4")  # black to move
    rel = relative_labels(board_labels(b)[None], [False])[0]
    assert rel[chess.E1] == chess.KING  # black king, seen as 'my' king on rank 1
    assert rel[chess.E5] == 7  # white pawn on e4, seen as 'their' pawn, mirrored to e5
    same = relative_labels(board_labels(b)[None], [True])[0]
    assert (same == board_labels(b)).all()


def test_metrics_and_majority():
    y = np.zeros((4, 64), np.int64)
    y[:, 0] = [1, 1, 2, 2]
    p = y.copy()
    p[0, 0] = 2
    m = linear.metrics(p, y)
    assert m["bal_acc"] == 0.75 and m["board_match"] == 0.75
    assert linear.majority(y, y)["board_match"] == 0.5


def test_fit_learns_linear_rule():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(2000, 16, generator=g)
    Y = torch.zeros(2000, 64, dtype=torch.long)
    Y[:, 5] = (X[:, 0] > 0).long() + 1
    p = linear.fit(X, Y, min_steps=500, batch=256, device="cpu")
    assert linear.metrics(linear.predict(p, X, device="cpu"), Y)["acc"] > 0.99

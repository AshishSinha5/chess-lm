"""Probe data: positions from held-out games, board labels and move-history features (PLAN, Probing).
A position is the board after k moves. In the token sequence [bos, m1, ..., mn, eos] it sits at index k:
there the model reads move k and predicts move k+1. Games come from the test split, which no run trains or evaluates on."""
import chess
import numpy as np

from chesslm.data.dataset import Shard
from chesslm.tokenizers.uci_move import UCIMoveTokenizer

TOK = UCIMoveTokenizer.canonical()
DATA = "/workspace/data/lichess_1800"
N_CLASS = 13  # 0 empty, 1-6 white P N B R Q K, 7-12 black P N B R Q K
UNTOUCHED, VACATED, WHITE_IN, BLACK_IN = 0, 1, 2, 3  # move-history state of a square


def board_labels(board):
    """Piece class on each of the 64 squares (a1 = 0, h8 = 63)."""
    out = np.zeros(64, np.int8)
    for sq, p in board.piece_map().items():
        out[sq] = p.piece_type + (0 if p.color == chess.WHITE else 6)
    return out


def relative_labels(labels, white_to_move):
    """Player-relative variant: 1-6 'mine' (side to move), 7-12 'theirs'.
    When black is to move, mirror the ranks, so the side to move always starts on rank 1. labels: (N, 64)."""
    labels = np.asarray(labels)
    out = labels.copy()
    black = ~np.asarray(white_to_move, bool)
    mirror = np.arange(64) ^ 56  # a1 <-> a8, ...
    b = labels[black][:, mirror]
    out[black] = np.where(b == 0, 0, np.where(b <= 6, b + 6, b - 6))
    return out


def sample_positions(n_games, per_game, seed=0, split="test", min_moves=10):
    """Random positions from n_games games, up to per_game per game. Each position has a human next move (k < n).
    Returns a dict of arrays; 'games' holds the token ids of each game, 'game' the game of each position."""
    shard = Shard(DATA, split)
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ["game", "k", "labels", "white_to_move", "next_tok", "legal", "hist_state", "hist_from"]}
    games = []
    for g in rng.permutation(shard.n):
        if len(games) == n_games:
            break
        ids = shard.tokens[shard.offs[g]: shard.offs[g + 1]].astype(np.int64)
        moves = TOK.decode(ids.tolist())
        if len(moves) < min_moves:
            continue
        ks = set(rng.choice(np.arange(1, len(moves)), min(per_game, len(moves) - 1), replace=False).tolist())
        gi = len(games)
        games.append(ids)
        board = chess.Board()
        state = np.full(64, UNTOUCHED, np.int8)
        src = np.full(64, -1, np.int8)  # from-square of the last move that landed on each square
        for k, u in enumerate(moves, 1):
            m = chess.Move.from_uci(u)
            state[m.from_square], src[m.from_square] = VACATED, -1
            state[m.to_square] = WHITE_IN if board.turn == chess.WHITE else BLACK_IN
            src[m.to_square] = m.from_square
            board.push(m)
            if k in ks:
                out["game"].append(gi)
                out["k"].append(k)
                out["labels"].append(board_labels(board))
                out["white_to_move"].append(board.turn == chess.WHITE)
                out["next_tok"].append(TOK.stoi[moves[k]])
                out["legal"].append(np.array([TOK.stoi[x.uci()] for x in board.legal_moves], np.int16))
                out["hist_state"].append(state.copy())
                out["hist_from"].append(src.copy())
    res = {k: (v if k == "legal" else np.array(v)) for k, v in out.items()}
    res["games"] = games
    return res


def split_by_game(pos, test_frac=0.2):
    """Boolean mask over positions: True = probe training set. The split is by game, so no game is in both sets."""
    n_games = len(pos["games"])
    return pos["game"] < round(n_games * (1 - test_frac))

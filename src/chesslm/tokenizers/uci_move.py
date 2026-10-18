"""Scheme A: one token per UCI move. Vocab is built from moves seen in data."""
import chess

SPECIALS = ["<pad>", "<bos>", "<eos>"]


class UCIMoveTokenizer:
    def __init__(self, moves):
        self.itos = SPECIALS + sorted(set(moves))
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad, self.bos, self.eos = 0, 1, 2

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, ucis):
        return [self.bos] + [self.stoi[m] for m in ucis] + [self.eos]

    def decode(self, ids):
        return [self.itos[i] for i in ids if i >= len(SPECIALS)]


def replay_legal(ucis):
    """True if every UCI move is legal when replayed from the start position."""
    board = chess.Board()
    for u in ucis:
        m = chess.Move.from_uci(u)
        if m not in board.legal_moves:
            return False
        board.push(m)
    return True


def san_game_to_ucis(game):
    """python-chess Game -> list of UCI strings (promotions as e7e8q)."""
    return [m.uci() for m in game.mainline_moves()]

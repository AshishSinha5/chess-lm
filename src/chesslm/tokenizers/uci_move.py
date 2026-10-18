"""Scheme A: one token per UCI move. Vocab is built from moves seen in data."""
import chess

SPECIALS = ["<pad>", "<bos>", "<eos>"]


class UCIMoveTokenizer:
    def __init__(self, moves):
        self.itos = SPECIALS + sorted(set(moves))
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad, self.bos, self.eos = 0, 1, 2

    @classmethod
    def canonical(cls):
        """Fixed vocab: every move that is possible by geometry (queen, knight, promotion)."""
        return cls(all_uci_moves())

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, ucis):
        return [self.bos] + [self.stoi[m] for m in ucis] + [self.eos]

    def decode(self, ids):
        return [self.itos[i] for i in ids if i >= len(SPECIALS)]


def all_uci_moves():
    """All UCI strings a legal game can contain, ignoring the position."""
    sq = lambda f, r: "abcdefgh"[f] + str(r + 1)
    deltas = [(df, dr) for df in range(-2, 3) for dr in range(-2, 3) if sorted((abs(df), abs(dr))) == [1, 2]]
    moves = set()
    for f in range(8):
        for r in range(8):
            for k in range(1, 8):
                for df, dr in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    tf, tr = f + df * k, r + dr * k
                    if 0 <= tf < 8 and 0 <= tr < 8:
                        moves.add(sq(f, r) + sq(tf, tr))
            for df, dr in deltas:
                tf, tr = f + df, r + dr
                if 0 <= tf < 8 and 0 <= tr < 8:
                    moves.add(sq(f, r) + sq(tf, tr))
            if r in (1, 6):
                tr = 7 if r == 6 else 0
                for df in (-1, 0, 1):
                    if 0 <= f + df < 8:
                        moves.update(sq(f, r) + sq(f + df, tr) + p for p in "qrbn")
    return sorted(moves)


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

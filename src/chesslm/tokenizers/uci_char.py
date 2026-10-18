"""Scheme B: character-level over UCI. Each move is its chars plus a space separator."""
SPECIALS = ["<pad>", "<bos>", "<eos>"]
CHARS = list("abcdefgh12345678qrbn ")


class UCICharTokenizer:
    def __init__(self):
        self.itos = SPECIALS + CHARS
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad, self.bos, self.eos = 0, 1, 2

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, ucis):
        return [self.bos] + [self.stoi[c] for c in " ".join(ucis)] + [self.eos]

    def decode(self, ids):
        s = "".join(self.itos[i] for i in ids if i >= len(SPECIALS))
        return s.split()

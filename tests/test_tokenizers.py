import chess
import random
from chesslm.tokenizers.uci_move import UCIMoveTokenizer, replay_legal
from chesslm.tokenizers.uci_char import UCICharTokenizer


def random_game(rng, max_plies=120):
    b, out = chess.Board(), []
    while not b.is_game_over() and len(out) < max_plies:
        m = rng.choice(list(b.legal_moves))
        out.append(m.uci())
        b.push(m)
    return out


def games(n=50):
    rng = random.Random(0)
    return [random_game(rng) for _ in range(n)]


def test_move_roundtrip_and_legal():
    gs = games()
    tok = UCIMoveTokenizer([m for g in gs for m in g])
    for g in gs:
        ids = tok.encode(g)
        assert ids[0] == tok.bos and ids[-1] == tok.eos
        assert tok.decode(ids) == g
        assert replay_legal(tok.decode(ids))


def test_char_roundtrip_and_legal():
    tok = UCICharTokenizer()
    for g in games():
        assert tok.decode(tok.encode(g)) == g
        assert replay_legal(tok.decode(tok.encode(g)))


def test_promotion_and_castling_encoded():
    tok = UCICharTokenizer()
    g = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "e1g1"]
    assert tok.decode(tok.encode(g)) == g and replay_legal(g)
    assert tok.decode(tok.encode(["e7e8q"])) == ["e7e8q"]


def test_canonical_vocab_covers_games():
    from chesslm.tokenizers.uci_move import all_uci_moves
    tok = UCIMoveTokenizer.canonical()
    assert 1800 < len(all_uci_moves()) < 2300
    for g in games():
        assert tok.decode(tok.encode(g)) == g
    for u in ["e1g1", "e1c1", "e7e8q", "a2b1n", "g1f3"]:
        assert u in tok.stoi

"""PGN text -> list of UCI strings."""
import io

import chess.pgn

MIN_PLIES, MAX_PLIES = 10, 256


def pgn_to_ucis(raw):
    """Return UCI moves, or None if the game is too short or cannot be parsed."""
    game = chess.pgn.read_game(io.StringIO(raw))
    if game is None or game.errors:
        return None
    ucis = [m.uci() for m in game.mainline_moves()]
    if len(ucis) < MIN_PLIES:
        return None
    return ucis[:MAX_PLIES]

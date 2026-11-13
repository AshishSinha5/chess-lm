"""Worker process for EnginePool: one Stockfish engine per process. Imports only python-chess, so it starts fast.

Why processes and not threads: python-chess parses every Stockfish output line in Python. With 11 engine threads
in the main process, the model loop got little interpreter time (GIL): one forward pass took about 190 ms instead
of 8 ms, and the engine count did not change throughput (results/EXPERIMENT_LOG.md, step 26)."""
from multiprocessing import util

import chess
import chess.engine

_S = {}


def init(stockfish, movetime):
    e = chess.engine.SimpleEngine.popen_uci(stockfish)
    e.configure({"Threads": 1, "Hash": 16})  # same settings as the stage-1 eval
    _S.update(engine=e, movetime=movetime, spec=None)
    # Quit the engine when the worker exits. Else the worker waits forever on python-chess's engine thread.
    util.Finalize(None, e.quit, exitpriority=10)


def move(ucis, spec, options):
    """Stockfish's move for the position after `ucis` (from the start position). Returns (UCI string, nodes searched).
    The node count shows how strong the engine was at the fixed time per move: a busy CPU gives fewer nodes."""
    e = _S["engine"]
    if _S["spec"] != spec:  # stage 1 reset to full strength, then set the opponent options
        e.configure({"UCI_LimitStrength": False, "Skill Level": 20})
        e.configure(options)
        _S["spec"] = spec
    board = chess.Board()
    for u in ucis:
        board.push_uci(u)
    r = e.play(board, chess.engine.Limit(time=_S["movetime"]), info=chess.engine.INFO_BASIC)
    return r.move.uci(), r.info.get("nodes")

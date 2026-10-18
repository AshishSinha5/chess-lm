"""Benchmark: stream, filter and replay games. Usage: bench_pgn.py MONTH N_GAMES WORKERS"""
import sys
import time
from collections import Counter
from multiprocessing import Pool

from chesslm.data.download import stream_games
from chesslm.data.filter import headers, keep
from chesslm.data.tokenize import pgn_to_ucis


def work(raw):
    return pgn_to_ucis(raw)


if __name__ == "__main__":
    month, n, workers = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    t0 = time.time()
    raws, seen, elo = [], 0, Counter()
    for raw in stream_games(month, n):
        seen += 1
        h = headers(raw)
        elo[min(int(h.get("WhiteElo", 0) or 0), int(h.get("BlackElo", 0) or 0)) // 200 * 200] += 1
        if keep(h):
            raws.append(raw)
    t1 = time.time()
    with Pool(workers) as p:
        res = p.map(work, raws, chunksize=64)
    t2 = time.time()
    ok = [r for r in res if r]
    plies = sum(len(r) for r in ok)
    print(f"read+filter: {seen} games in {t1-t0:.1f}s ({seen/(t1-t0):.0f}/s); kept {len(raws)} ({len(raws)/seen:.1%})")
    print(f"replay: {len(ok)} games, {plies} plies in {t2-t1:.1f}s with {workers} workers "
          f"-> {len(ok)/(t2-t1):.0f} games/s, {plies/(t2-t1):.0f} tokens/s")
    print(f"avg plies {plies/max(len(ok),1):.1f}")
    print("min-rating histogram (all games):", sorted(elo.items()))

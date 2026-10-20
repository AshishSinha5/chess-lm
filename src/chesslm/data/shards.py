"""Stream months -> filter -> replay -> uint16 token shards, split by game-ID hash.
Usage: python -m chesslm.data.shards OUT_DIR WORKERS MONTH [MONTH ...] [--max-chunks N]
Output per split: tokens.bin (uint16, <bos> moves <eos> per game) and offsets.npy (game starts, + total)."""
import json
import re
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from chesslm.data.download import stream_chunks
from chesslm.data.filter import headers, keep
from chesslm.data.tokenize import pgn_to_ucis
from chesslm.tokenizers.uci_move import UCIMoveTokenizer

TOK = UCIMoveTokenizer.canonical()
SPLITS = ("train", "val", "test")
SITE_ID = re.compile(r"lichess\.org/(\w+)")


def split_of(game_id):
    h = zlib.crc32(game_id.encode()) % 100
    return "train" if h < 98 else ("val" if h == 98 else "test")


def process(chunk):
    out = {s: ([], []) for s in SPLITS}  # token arrays, lengths
    stats = {"games": 0, "kept": 0, "bad": 0}
    for raw in chunk.decode("utf-8", "replace").split("\n[Event "):
        if not raw.strip():
            continue
        raw = raw if raw.startswith("[Event ") else "[Event " + raw
        stats["games"] += 1
        h = headers(raw)
        if not keep(h):
            continue
        ucis = pgn_to_ucis(raw)
        m = SITE_ID.search(h.get("Site", ""))
        if ucis is None or m is None or any(u not in TOK.stoi for u in ucis):
            stats["bad"] += 1
            continue
        ids = np.array(TOK.encode(ucis), dtype=np.uint16)
        arrs, lens = out[split_of(m.group(1))]
        arrs.append(ids)
        lens.append(len(ids))
        stats["kept"] += 1
    return {s: (np.concatenate(a) if a else np.zeros(0, np.uint16), l) for s, (a, l) in out.items()}, stats


def main(out_dir, workers, months, max_chunks=None):
    out_dir = Path(out_dir)
    files, lens, totals = {}, {s: [] for s in SPLITS}, {"games": 0, "kept": 0, "bad": 0}
    for s in SPLITS:
        (out_dir / s).mkdir(parents=True, exist_ok=True)
        files[s] = open(out_dir / s / "tokens.bin", "wb")
    with Pool(workers) as pool:
        for month in months:
            n = max_chunks or None
            chunks = stream_chunks(month)
            for i, (res, stats) in enumerate(pool.imap(process, chunks, chunksize=1)):
                for s in SPLITS:
                    files[s].write(res[s][0].tobytes())
                    lens[s].extend(res[s][1])
                for k in totals:
                    totals[k] += stats[k]
                if i % 10 == 0:
                    print(month, "chunk", i, totals, {s: int(sum(lens[s])) for s in SPLITS}, flush=True)
                if n and i + 1 >= n:
                    break
    for s in SPLITS:
        files[s].close()
        offs = np.concatenate([[0], np.cumsum(lens[s])]).astype(np.uint64)
        np.save(out_dir / s / "offsets.npy", offs)
    meta = {"months": months, "filter": "both>=1800, tc>=3+0, normal, 10-256 plies", "vocab_size": TOK.vocab_size,
            **totals, **{f"{s}_tokens": int(sum(lens[s])) for s in SPLITS}, **{f"{s}_games": len(lens[s]) for s in SPLITS}}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print("DONE", meta, flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    mc = None
    if "--max-chunks" in args:
        i = args.index("--max-chunks")
        mc = int(args[i + 1])
        del args[i:i + 2]
    main(args[0], int(args[1]), args[2:], mc)

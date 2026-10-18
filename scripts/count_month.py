"""Count games and estimated tokens for a whole month at several rating thresholds.
Usage: count_month.py MONTH WORKERS. Plies are estimated from the last move number."""
import json
import re
import sys
import time
import urllib.request
from multiprocessing import Pool

import zstandard

from chesslm.data.download import BASE
from chesslm.data.filter import headers, keep

THRESHOLDS = (1600, 1800, 2000)
LAST_MOVE = re.compile(r"(\d+)\. (?:\S+)(?: (?:\{[^}]*\} )?(?!\d+\.)(\S+))?\s*(?:1-0|0-1|1/2-1/2|\*)?\s*$")
NUM = re.compile(r"(\d+)\.")


def plies(raw):
    movetext = raw.split("\n\n", 1)[1] if "\n\n" in raw else ""
    nums = NUM.findall(movetext)
    if not nums:
        return 0
    n = int(nums[-1])
    tail = movetext[movetext.rfind(f"{n}."):]
    words = [w for w in re.sub(r"\{[^}]*\}", "", tail).split() if w not in ("1-0", "0-1", "1/2-1/2", "*")]
    return 2 * n if len(words) >= 3 else 2 * n - 1


def count(chunk):
    out = {"all": 0, **{t: [0, 0] for t in THRESHOLDS}}
    for raw in chunk.decode("utf-8", "replace").split("\n[Event "):
        if not raw.strip():
            continue
        raw = raw if raw.startswith("[Event ") else "[Event " + raw
        out["all"] += 1
        h = headers(raw)
        for t in THRESHOLDS:
            if keep(h, t):
                p = plies(raw)
                if p >= 10:
                    out[t][0] += 1
                    out[t][1] += min(p, 256)
    return out


def chunks(month, size=32 << 20):
    resp = urllib.request.urlopen(BASE.format(month=month))
    reader = zstandard.ZstdDecompressor().stream_reader(resp)
    rest = b""
    while True:
        data = reader.read(size)
        if not data:
            break
        data = rest + data
        i = data.rfind(b"\n[Event ")
        if i < 0:
            rest = data
            continue
        yield data[:i]
        rest = data[i:]
    if rest:
        yield rest


if __name__ == "__main__":
    month, workers = sys.argv[1], int(sys.argv[2])
    total = {"all": 0, **{t: [0, 0] for t in THRESHOLDS}}
    t0 = time.time()
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap(count, chunks(month), chunksize=1)):
            total["all"] += r["all"]
            for t in THRESHOLDS:
                total[t][0] += r[t][0]
                total[t][1] += r[t][1]
            if i % 10 == 0:
                print(f"{time.time()-t0:.0f}s chunk {i} games {total['all']:,} "
                      + " ".join(f"{t}:{total[t][0]:,}g/{total[t][1]/1e9:.3f}Btok" for t in THRESHOLDS), flush=True)
    print("FINAL", json.dumps({str(k): v for k, v in total.items()}), flush=True)

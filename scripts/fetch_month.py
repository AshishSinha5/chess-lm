"""Download a Lichess month with parallel range requests. Resumable.
Usage: fetch_month.py MONTH OUT_DIR [CONNECTIONS]. Writes OUT_DIR/lichess_db_standard_rated_MONTH.pgn.zst"""
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

PART = 64 << 20
BASE = "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"


def main(month, out_dir, conns=16):
    url = BASE.format(month=month)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / url.rsplit("/", 1)[1]
    done_file = Path(str(dest) + ".done")
    size = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD")).headers["Content-Length"])
    parts = [(i, i * PART, min((i + 1) * PART, size) - 1) for i in range((size + PART - 1) // PART)]
    done = set(int(x) for x in done_file.read_text().split()) if done_file.exists() else set()
    fd = os.open(dest, os.O_RDWR | os.O_CREAT)
    os.ftruncate(fd, size)
    todo = [p for p in parts if p[0] not in done]
    lock, got = threading.Lock(), [sum(p[2] - p[1] + 1 for p in parts if p[0] in done)]

    def worker():
        while True:
            with lock:
                if not todo:
                    return
                i, a, b = todo.pop(0)
            for attempt in range(8):
                try:
                    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
                    with urllib.request.urlopen(req, timeout=60) as r:
                        pos = a
                        while pos <= b:
                            buf = r.read(1 << 20)
                            if not buf:
                                raise IOError("short read")
                            os.pwrite(fd, buf, pos)
                            pos += len(buf)
                            with lock:
                                got[0] += len(buf)
                    with lock:
                        with open(done_file, "a") as f:
                            f.write(f"{i}\n")
                    break
                except Exception as e:
                    with lock:
                        got[0] -= max(0, pos - a) if "pos" in dir() else 0
                    time.sleep(2 * (attempt + 1))
            else:
                raise SystemExit(f"part {i} failed")

    ts = [threading.Thread(target=worker, daemon=True) for _ in range(conns)]
    t0 = time.time()
    [t.start() for t in ts]
    while any(t.is_alive() for t in ts):
        time.sleep(10)
        print(f"{month}: {got[0]/1e9:.2f}/{size/1e9:.2f} GB, {got[0]/(time.time()-t0)/1e6:.1f} MB/s avg", flush=True)
    os.close(fd)
    assert dest.stat().st_size == size and len(set(int(x) for x in done_file.read_text().split())) == len(parts)
    Path(str(dest) + ".complete").touch()
    print("FETCHED", dest, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 16)

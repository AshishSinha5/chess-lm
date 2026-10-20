"""Stream raw games from a Lichess .pgn.zst URL. The file is never stored."""
import io
import os
import time
import urllib.request

import zstandard

BASE = "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"


def stream_games(month, max_games=None):
    """Yield one raw PGN game (headers + movetext) at a time."""
    resp = urllib.request.urlopen(BASE.format(month=month))
    reader = zstandard.ZstdDecompressor().stream_reader(resp)
    text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    buf, n = [], 0
    for line in text:
        if line.startswith("[Event ") and buf:
            yield "".join(buf)
            n += 1
            if max_games and n >= max_games:
                return
            buf = []
        buf.append(line)
    if buf:
        yield "".join(buf)


RAW_DIR = "/workspace/raw"


def local_path(month, raw_dir=RAW_DIR):
    return os.path.join(raw_dir, BASE.format(month=month).rsplit("/", 1)[1])


def stream_chunks(month, size=32 << 20):
    """Yield ~32 MB blocks of raw PGN bytes, cut on game boundaries.
    Reads the local file from scripts/fetch_month.py when it exists (waits until it is complete).
    Falls back to a direct stream from Lichess."""
    path = local_path(month)
    if os.path.exists(path):
        while not os.path.exists(path + ".complete"):
            time.sleep(10)
        src = open(path, "rb")
    else:
        src = urllib.request.urlopen(BASE.format(month=month))
    reader = zstandard.ZstdDecompressor().stream_reader(src)
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

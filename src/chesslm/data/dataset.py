"""Read token shards and make length-bucketed batches.
Games go to a bucket by padded length (multiple of 16). Each bucket makes batches of a fixed size
batch_tokens // T, so there are few distinct tensor shapes (good for torch.compile)."""
import math
import queue
import threading
from pathlib import Path

import numpy as np
import torch

PAD = 0
IGNORE = -100
MULT = 16


class Shard:
    def __init__(self, root, split, max_games=None):
        d = Path(root) / split
        self.tokens = np.memmap(d / "tokens.bin", dtype=np.uint16, mode="r")
        offs = np.load(d / "offsets.npy").astype(np.int64)
        self.offs = offs
        self.n = len(offs) - 1 if not max_games else min(len(offs) - 1, max_games)
        self.lens = np.diff(offs)[: self.n]

    def pad_len(self, length):
        """Input length T for a game of `length` tokens (inputs = tokens[:-1])."""
        return math.ceil((length - 1) / MULT) * MULT

    def collate(self, idx, T):
        arr = np.full((len(idx), T + 1), PAD, dtype=np.int64)
        for r, g in enumerate(idx):
            a, b = self.offs[g], self.offs[g + 1]
            arr[r, : b - a] = self.tokens[a:b]
        x, y = arr[:, :-1], arr[:, 1:].copy()
        y[y == PAD] = IGNORE
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(y)


def batch_indices(shard, batch_tokens, seed, shuffle=True):
    """Yield (game indices, T). Deterministic for a given seed, so training can resume."""
    order = np.random.default_rng(seed).permutation(shard.n) if shuffle else np.arange(shard.n)
    pending = {}
    for s in range(0, shard.n, 100_000):
        idx = order[s : s + 100_000]
        for g, T in zip(idx.tolist(), [shard.pad_len(l) for l in shard.lens[idx]]):
            lst = pending.setdefault(T, [])
            lst.append(g)
            if len(lst) == max(1, batch_tokens // T):
                yield np.array(lst), T
                pending[T] = []


def loader(shard, batch_tokens, seed=0, skip=0, epochs=None, prefetch=8):
    """Yield (x, y, epoch, batch_in_epoch). Runs forever unless epochs is set."""
    q = queue.Queue(prefetch)

    def work():
        ep = 0
        while epochs is None or ep < epochs:
            for i, (idx, T) in enumerate(batch_indices(shard, batch_tokens, seed + ep)):
                if ep == skip[0] and i < skip[1]:
                    continue
                x, y = shard.collate(idx, T)
                q.put((x, y, ep, i + 1))
            ep += 1
        q.put(None)

    skip = skip if isinstance(skip, tuple) else (0, skip)
    threading.Thread(target=work, daemon=True).start()
    while (item := q.get()) is not None:
        yield item

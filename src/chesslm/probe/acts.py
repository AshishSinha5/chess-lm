"""Run a model over whole games and read the hidden states at the probe positions.
Hidden state 0 is the embedding output; hidden state l (1..n_layer) is the output of block l."""
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def collect(model, pos, layers=None, batch=64, device="cuda"):
    """Returns (hidden, ce, top1).
    hidden: float16 tensor (len(layers), N, d) on the CPU. ce: loss on the human next move per position. top1: argmax token."""
    layers = list(range(model.cfg.n_layer + 1)) if layers is None else list(layers)
    games, game, k = pos["games"], pos["game"], pos["k"]
    N, d = len(game), model.cfg.d_model
    hidden = torch.empty(len(layers), N, d, dtype=torch.float16)
    ce = torch.empty(N)
    top1 = torch.empty(N, dtype=torch.long)
    next_tok = torch.as_tensor(pos["next_tok"])
    by_game = [[] for _ in games]
    for i, g in enumerate(game.tolist()):
        by_game[g].append(i)
    order = sorted((g for g in range(len(games)) if by_game[g]), key=lambda g: len(games[g]))  # similar lengths per batch
    for s in range(0, len(order), batch):
        gs = order[s: s + batch]
        T = max(len(games[g]) for g in gs)
        x = torch.zeros(len(gs), T, dtype=torch.long)
        for r, g in enumerate(gs):
            x[r, : len(games[g])] = torch.from_numpy(games[g])
        with torch.autocast(device, dtype=torch.bfloat16):
            logits, _, hid = model(x.to(device), return_hidden=True)
        rows = torch.tensor([r for r, g in enumerate(gs) for _ in by_game[g]], device=device)
        idx = torch.tensor([i for g in gs for i in by_game[g]])
        cols = torch.as_tensor(k[idx.numpy()], device=device)
        for j, l in enumerate(layers):
            hidden[j, idx] = hid[l][rows, cols].half().cpu()
        lg = logits[rows, cols].float()
        ce[idx] = F.cross_entropy(lg, next_tok[idx].to(device), reduction="none").cpu()
        top1[idx] = lg.argmax(-1).cpu()
    return hidden, ce, top1


def legal_and_match(pos, top1, mask=None):
    """Share of top-1 moves (no legal mask) that are legal, and share that equal the human move."""
    idx = np.arange(len(top1)) if mask is None else np.flatnonzero(mask)
    t = top1.numpy()
    legal = np.mean([t[i] in pos["legal"][i] for i in idx])
    match = np.mean(t[idx] == pos["next_tok"][idx])
    return float(legal), float(match)

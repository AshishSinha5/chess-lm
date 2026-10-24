"""Board probes: predict the piece class on all 64 squares from one feature vector.
Linear probe (main result) or a small MLP (comparison). Features are standardized with the training-set mean and std."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from chesslm.probe.data import N_CLASS


class Probe(nn.Module):
    def __init__(self, d, mu, sd, mlp=False, hidden=1024):
        super().__init__()
        self.register_buffer("mu", mu)
        self.register_buffer("sd", sd)
        out = 64 * N_CLASS
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, out)) if mlp else nn.Linear(d, out)

    def forward(self, x):
        return self.net((x.float() - self.mu) / self.sd).view(-1, 64, N_CLASS)

    def raw_directions(self):
        """Linear probe weights in the raw (not standardized) feature space: (64 * N_CLASS, d)."""
        return self.net.weight / self.sd


def fit(X, Y, mlp=False, epochs=20, min_steps=3000, lr=None, wd=1e-4, batch=1024, seed=0, standardize=True, device="cuda"):
    """X: (N, d) features, Y: (N, 64) labels. Trains for epochs, or longer to reach min_steps. Returns a Probe in eval mode."""
    torch.manual_seed(seed)
    X = torch.as_tensor(X).to(device)
    Y = torch.as_tensor(Y, dtype=torch.long).to(device)
    Xf = X.float()
    mu = Xf.mean(0) if standardize else torch.zeros(X.shape[1], device=device)
    sd = Xf.std(0).clamp(min=1e-4) if standardize else torch.ones(X.shape[1], device=device)
    probe = Probe(X.shape[1], mu, sd, mlp).to(device)
    lr = lr or (1e-3 if mlp else 1e-2)  # the MLP trained well at 1e-3 in the first test; 1e-2 is not tested on it
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    per_epoch = (len(X) + batch - 1) // batch
    epochs = max(epochs, -(-min_steps // per_epoch))
    steps = epochs * per_epoch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator(device=device).manual_seed(seed)
    for _ in range(epochs):
        for idx in torch.randperm(len(X), device=device, generator=g).split(batch):
            loss = F.cross_entropy(probe(X[idx]).reshape(-1, N_CLASS), Y[idx].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    return probe.eval()


@torch.no_grad()
def predict(probe, X, batch=8192, device="cuda"):
    X = torch.as_tensor(X)
    return torch.cat([probe(X[i: i + batch].to(device)).argmax(-1).cpu() for i in range(0, len(X), batch)]).numpy()


def metrics(pred, Y):
    """acc: all squares. bal_acc: mean recall over the 12 piece classes, on non-empty squares only.
    board_match: share of positions with all 64 squares right."""
    pred, Y = np.asarray(pred), np.asarray(Y)
    recalls = [np.mean(pred[Y == c] == c) for c in range(1, N_CLASS) if np.any(Y == c)]
    return {"acc": float(np.mean(pred == Y)), "bal_acc": float(np.mean(recalls)),
            "board_match": float(np.mean((pred == Y).all(1)))}


def majority(Ytrain, Ytest):
    """Majority-class control: for each square, always predict its most common class in the training set."""
    Ytrain = np.asarray(Ytrain)
    mode = np.array([np.bincount(Ytrain[:, s], minlength=N_CLASS).argmax() for s in range(64)])
    return metrics(np.broadcast_to(mode, np.asarray(Ytest).shape), Ytest)


def history_features(pos, device="cuda"):
    """Move-history control: per square, its last event (untouched, vacated, white arrived, black arrived)
    and the from-square of the last move that landed on it, plus the side to move. No board rules are used."""
    st = torch.as_tensor(pos["hist_state"], dtype=torch.long, device=device)
    src = torch.as_tensor(pos["hist_from"], dtype=torch.long, device=device) + 1  # 0 = none
    stm = torch.as_tensor(pos["white_to_move"], dtype=torch.float, device=device)[:, None]
    return torch.cat([F.one_hot(st, 4).flatten(1).float(), F.one_hot(src, 65)[..., 1:].flatten(1).float(), stm], 1)

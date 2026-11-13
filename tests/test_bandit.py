"""Legality bandit tests (docs/RL_PLAN.md, W4 step 4). CPU only, tiny model."""
import copy

import chess
import torch
import torch.nn.functional as F

from chesslm.eval.play import TOK
from chesslm.model.gpt import GPT, GPTConfig
from chesslm.rl.bandit import group_advantage, legal_mask, next_token_logits, train_step

torch.set_num_threads(4)
HIST = [[], ["e2e4"], ["e2e4", "e7e5", "g1f3"], ["d2d4", "d7d5", "c2c4", "e7e6"]]


def board_of(h):
    b = chess.Board()
    for m in h:
        b.push_uci(m)
    return b


def test_legal_mask_start_position():
    m = legal_mask(chess.Board())
    assert m.sum().item() == 20 and m[TOK.stoi["e2e4"]] and m[TOK.stoi["g1f3"]]
    assert not m[TOK.stoi["e2e5"]] and not m[: len(["<pad>", "<bos>", "<eos>"])].any()  # special tokens are illegal


def test_group_advantage():
    r = torch.tensor([[1., 1., 1., 1.], [1., 1., 1., 0.], [0., 0., 0., 1.]])
    a = group_advantage(r)
    assert torch.allclose(a[0], torch.zeros(4))  # all equal: no signal
    assert torch.allclose(a.mean(1), torch.zeros(3), atol=1e-6)  # centered per group
    assert a[1, 3] < 0 < a[1, 0] and a[2, 3] > 0 > a[2, 0]  # the rare outcome gets the large, signed advantage
    assert abs(a[1, 3]) > abs(a[1, 0])


def test_bandit_raises_legal_mass_on_fixed_positions():
    """The update pushes probability from illegal tokens to legal ones. A random model starts near 20/1971 = 1%."""
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=TOK.vocab_size, d_model=32, n_layer=1, n_head=2, block_size=272))
    base = copy.deepcopy(model).eval()
    for q in base.parameters():
        q.requires_grad_(False)
    mask = torch.stack([legal_mask(board_of(h)) for h in HIST])

    def mass():
        with torch.no_grad():
            p = F.softmax(next_token_logits(model, HIST, "cpu", False), dim=-1)
        return (p * mask).sum(-1).mean().item()

    m0 = mass()
    opt, gen = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0), torch.Generator().manual_seed(0)
    for _ in range(150):
        r, kl = train_step(model, base, opt, HIST, mask, K=128, beta=0.0, gen=gen, device="cpu", amp=False)
    m1 = mass()
    assert m0 < 0.05 and m1 > 5 * m0 and m1 > 0.2, (m0, m1)


def test_kl_penalty_keeps_policy_near_base():
    """With a large beta the policy stays closer to the base model than with beta 0 (same data, same seed)."""
    def run(beta):
        torch.manual_seed(1)
        model = GPT(GPTConfig(vocab_size=TOK.vocab_size, d_model=32, n_layer=1, n_head=2, block_size=272))
        base = copy.deepcopy(model).eval()
        for q in base.parameters():
            q.requires_grad_(False)
        mask = torch.stack([legal_mask(board_of(h)) for h in HIST])
        opt, gen = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0), torch.Generator().manual_seed(2)
        for _ in range(60):
            train_step(model, base, opt, HIST, mask, K=128, beta=beta, gen=gen, device="cpu", amp=False)
        with torch.no_grad():
            lp, lb = (F.log_softmax(next_token_logits(m, HIST, "cpu", False), -1) for m in (model, base))
        return (lp.exp() * (lp - lb)).sum(-1).mean().item()
    assert run(5.0) < run(0.0)


def test_exact_loss_raises_legal_mass_and_keeps_legal_ratios():
    """-log(legal mass) lifts the legal mass. Among legal moves, the probabilities keep their ratios (more than the sampled loss does)."""
    torch.manual_seed(3)
    model = GPT(GPTConfig(vocab_size=TOK.vocab_size, d_model=32, n_layer=1, n_head=2, block_size=272))
    base = copy.deepcopy(model).eval()
    for q in base.parameters():
        q.requires_grad_(False)
    mask = torch.stack([legal_mask(board_of(h)) for h in HIST])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    with torch.no_grad():
        p0 = F.softmax(next_token_logits(model, HIST, "cpu", False), -1)
    r0 = (p0 * mask).sum(-1).mean().item()
    for _ in range(80):
        r, _ = train_step(model, base, opt, HIST, mask, K=1, beta=0.0, gen=None, device="cpu", amp=False, loss_type="exact")
    assert r > 5 * r0 and r > 0.5

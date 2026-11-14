"""RFT tests (docs/RL_PLAN.md, W4 step 5b). CPU only, tiny model."""
import chess
import torch

from chesslm.eval.play import TOK
from chesslm.model.gpt import GPT, GPTConfig
from chesslm.rl.rft import build_batch, finetune, sft_loss
from chesslm.rl.rollout import Game

torch.set_num_threads(4)


def fake_game(ucis, model_is_white):
    """A game where the model plays `model_is_white` color after a 2-ply opening."""
    toks = [TOK.bos] + [TOK.stoi[u] for u in ucis]
    pos = [t for t in range(1, len(toks)) if (t % 2 == 1) == model_is_white]  # white plies are the odd t
    pos = [t for t in pos if t > 2]  # not the opening
    g = Game(opp={}, white=model_is_white, opening=ucis[:2], board=chess.Board(), tokens=toks, idx=0)
    g.model_pos = pos
    return g


def test_mask_marks_only_model_moves():
    ucis = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]
    for white in (True, False):
        g = fake_game(ucis, white)
        x, y, m = build_batch([g], "cpu")
        idx = [i for i in range(m.shape[1]) if m[0, i] == 1]
        assert idx == [t - 1 for t in g.model_pos]
        for i in idx:  # the target at a masked-in index is a model token
            assert y[0, i].item() == g.tokens[i + 1] and x[0, i].item() == g.tokens[i]
        assert m.sum().item() == len(g.model_pos)
        assert not m[0, :2].any()  # opening plies are never trained on (white: t=1,2; black: t=1,2)


def test_padding_is_ignored_and_loss_matches_manual():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=TOK.vocab_size, n_layer=1, n_head=2, d_model=32, block_size=64)).eval()
    a = fake_game(["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"], True)
    b = fake_game(["d2d4", "d7d5", "c2c4", "e7e6"], False)
    both = sft_loss(model, [a, b], "cpu", amp=False).item()
    la, lb = sft_loss(model, [a], "cpu", amp=False).item(), sft_loss(model, [b], "cpu", amp=False).item()
    na, nb = len(a.model_pos), len(b.model_pos)
    assert abs(both - (la * na + lb * nb) / (na + nb)) < 1e-4  # padding does not change the mean over model tokens


def test_finetune_lowers_loss_on_kept_games():
    import numpy as np
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=TOK.vocab_size, n_layer=1, n_head=2, d_model=32, block_size=64))
    kept = [fake_game(["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"], True)] * 4
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    before = sft_loss(model, kept, "cpu", amp=False).item()
    finetune(model, opt, kept, epochs=30, batch=4, rng=np.random.default_rng(0), device="cpu")
    assert sft_loss(model, kept, "cpu", amp=False).item() < 0.5 * before


def test_finetune_logs_lr_tokens_and_speed():
    import numpy as np
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=TOK.vocab_size, n_layer=1, n_head=2, d_model=32, block_size=64))
    kept = [fake_game(["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"], True)] * 5
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    steps = finetune(model, opt, kept, epochs=2, batch=2, rng=np.random.default_rng(0), device="cpu")
    assert len(steps) == 6 and steps[0]["lr"] == 2e-3 and steps[0]["tokens"] == 2 * len(kept[0].model_pos)
    assert all(s["tok_s"] > 0 and s["grad_norm"] >= 0 for s in steps)
    from chesslm.rl.rft import STEP_FIELDS
    assert set(STEP_FIELDS) - {"name", "round"} == set(steps[0])


def test_config_and_resume_state_roundtrip(tmp_path):
    import argparse
    import numpy as np
    from chesslm.rl.rft import RESUME_FREE, save_state, write_config
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=TOK.vocab_size, n_layer=1, n_head=2, d_model=32, block_size=64))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sft_loss(model, [fake_game(["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"], True)], "cpu", amp=False).backward()
    opt.step()
    rng = np.random.default_rng(3)
    rng.permutation(10)
    cfg = write_config(tmp_path / "config.json", argparse.Namespace(lr=1e-5, rounds=4), model, "size_m")
    assert cfg["args"]["lr"] == 1e-5 and cfg["n_params"] > 0 and len(cfg["git"]) == 40 and "rounds" in RESUME_FREE
    save_state(model, opt, rng, 2, tmp_path / "last.pt")
    st = torch.load(tmp_path / "last.pt")
    rng2 = np.random.default_rng(0)
    rng2.bit_generator.state = st["rng"]
    assert st["round"] == 2 and (rng2.permutation(10) == rng.permutation(10)).all()  # same next random draw
    assert "exp_avg" in next(iter(st["opt"]["state"].values()))

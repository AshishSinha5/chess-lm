import numpy as np
import torch
from chesslm.model.gpt import GPT, GPTConfig
from chesslm.data.dataset import Shard, batch_indices, loader


def small():
    return GPT(GPTConfig(vocab_size=50, d_model=64, n_layer=2, n_head=2, block_size=32))


def test_shapes_and_loss():
    m = small()
    x = torch.randint(3, 50, (4, 16))
    logits, loss = m(x, x)
    assert logits.shape == (4, 16, 50) and loss.item() > 0


def test_causal():
    m = small().eval()
    x = torch.randint(3, 50, (1, 16))
    a, _ = m(x)
    x2 = x.clone(); x2[0, 10:] = 7
    b, _ = m(x2)
    assert torch.allclose(a[0, :10], b[0, :10], atol=1e-5)
    assert not torch.allclose(a[0, 12], b[0, 12], atol=1e-5)


def test_param_counts():
    from chesslm.model.gpt import GPT, GPTConfig
    for d, l, h in [(256, 6, 4), (384, 6, 6), (512, 8, 8), (640, 10, 10)]:
        n = GPT(GPTConfig(d_model=d, n_layer=l, n_head=h)).n_params()
        print(d, n)
        assert 3e6 < n < 6e7


def make_shard(tmp_path, lens):
    toks, offs = [], [0]
    for i, n in enumerate(lens):
        toks += [1] + [3 + (i + j) % 40 for j in range(n - 2)] + [2]
        offs.append(len(toks))
    d = tmp_path / "train"; d.mkdir()
    np.array(toks, np.uint16).tofile(d / "tokens.bin")
    np.save(d / "offsets.npy", np.array(offs, np.uint64))
    return Shard(tmp_path, "train")


def test_batches(tmp_path):
    sh = make_shard(tmp_path, [10, 20, 33, 40, 18, 25, 70, 12] * 20)
    seen = 0
    for idx, T in batch_indices(sh, 256, seed=1):
        assert T % 16 == 0 and len(idx) == max(1, 256 // T)
        assert (sh.lens[idx] - 1 <= T).all()
        seen += len(idx)
    assert 0 < seen <= sh.n
    x, y, ep, i = next(loader(sh, 256, 0, epochs=1))
    assert x.shape == y.shape and (y[x == 0] == -100).all()
    assert (x[:, 0] == 1).all()

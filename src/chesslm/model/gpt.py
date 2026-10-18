"""Decoder-only transformer: pre-norm, RoPE, GELU MLP, tied embeddings, no biases, no dropout."""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 1971
    d_model: int = 256
    n_layer: int = 6
    n_head: int = 4
    block_size: int = 272  # 258 real tokens max, padded up to a multiple of 16

    @property
    def head_dim(self):
        return self.d_model // self.n_head


def rope_cache(block_size, head_dim, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    ang = torch.outer(torch.arange(block_size).float(), inv)  # (T, hd/2)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """x: (B, H, T, hd). Rotate each pair of features by an angle set by position."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    T = x.shape[2]
    c, s = cos[:T].to(x.dtype), sin[:T].to(x.dtype)
    out = torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, C // self.n_head).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        return x + self.proj(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        cos, sin = rope_cache(cfg.block_size, cfg.head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        for b in self.blocks:  # scale residual outputs so variance stays flat with depth
            for p in (b.attn.proj, b.proj):
                nn.init.normal_(p.weight, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())  # tied head is counted once

    def forward(self, idx, targets=None, return_hidden=False):
        x = self.wte(idx)
        hidden = [x]
        for b in self.blocks:
            x = b(x, self.cos, self.sin)
            hidden.append(x)
        logits = F.linear(self.ln_f(x), self.wte.weight)  # tied output head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return (logits, loss, hidden) if return_hidden else (logits, loss)

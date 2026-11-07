---
license: apache-2.0
tags:
  - chess
  - gpt
  - game-playing
  - safetensors
library_name: pytorch
---

# Chess LM: small GPT models that play chess

Four small GPT-style models (XS, S, M, L). Each model reads the moves of a chess game and predicts the next move.
We trained them from scratch on human games from Lichess. They know no rules and use no search.

Code, training logs and write-ups: <https://github.com/AshishSinha5/chess-lm>

## Models

| Folder | Parameters | Layers | Width | Loss per move | Legal moves | Elo vs Stockfish |
|---|---|---|---|---|---|---|
| `xs/` | 5.2M | 6 | 256 | 1.789 | 98.4% | 1182 |
| `s/` | 11.4M | 6 | 384 | 1.643 | 99.4% | 1296 |
| `m/` | 26.2M | 8 | 512 | 1.519 | 99.6% | 1410 |
| `l/` | 50.4M | 10 | 640 | 1.465 | 99.5% | 1479 |

- **Loss per move:** the cross-entropy on held-out games, in nats, per move token (lower is better).
- **Legal moves:** how often the top-scoring move is legal, with no legal-move mask. One random position from each of 5,000 held-out games.
- **Elo vs Stockfish:** the Elo estimate from games against Stockfish 16 at limited `UCI_Elo`. It uses Stockfish's nominal scale. It is **not** a Lichess or FIDE rating.

All four files are the 1B-token runs with seed 0. Each folder holds `model.safetensors` (weights only) and `config.json`.

Strength rises with size. The step from M to L is small. Extra data does not close it: with 1.6B tokens the L lead over M is 0.051 in loss, against 0.054 at 1B tokens.

## How the models work

- **Input:** one token per move in UCI notation, for example `e2e4`. The vocabulary has 1,971 tokens: every possible UCI move plus `<pad>`, `<bos>` and `<eos>`.
- **Sequence:** `<bos>`, then the moves played so far. The model scores every token for the next move.
- **Architecture:** a decoder-only transformer with RoPE, pre-norm, GELU, tied embeddings, no biases and no dropout. The maximum length is 272 tokens.
- **Rules:** the model does not know the rules. It can name an illegal move. In the example below, the code keeps only legal moves and picks the highest score.

## Use

Clone the code, install it, then load a model from this repo.

```bash
git clone https://github.com/AshishSinha5/chess-lm && cd chess-lm
pip install -e . safetensors huggingface_hub python-chess
```

```python
import json
import chess
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from chesslm.model.gpt import GPT, GPTConfig
from chesslm.tokenizers.uci_move import UCIMoveTokenizer

size = "l"  # xs, s, m or l
path = snapshot_download("sinha/chess-lm", allow_patterns=[f"{size}/*"])
cfg = json.load(open(f"{path}/{size}/config.json"))["model_cfg"]
model = GPT(GPTConfig(**cfg)).eval()
model.load_state_dict(load_file(f"{path}/{size}/model.safetensors"))
tok = UCIMoveTokenizer.canonical()

board = chess.Board()
history = []  # UCI moves played so far, for example ["e2e4", "e7e5"]
ids = torch.tensor([[tok.bos] + [tok.stoi[m] for m in history]])
with torch.no_grad():
    logits = model(ids)[0][0, -1]
legal = [m.uci() for m in board.legal_moves]
print(max(legal, key=lambda u: logits[tok.stoi[u]].item()))
```

## License

Apache 2.0, for the weights and the code. The training games come from the Lichess open database (CC0).

## Training data

- **Source:** the Lichess open game database (CC0), months 2026-08 and 2026-09.
- **Filter:** both players rated 1800 or more, time control 3+0 or slower, normal game end.
- **Size:** 22.4M training games, 1.64B tokens. Each run here saw 1B tokens, one pass, no repeated games.
- **Recipe:** AdamW, weight decay 0.1, 65,536 tokens per batch, 2% warmup, then a flat phase and a final decay. The learning rate falls with width: 3e-3 (XS), 2e-3 (S), 1.5e-3 (M), 1.2e-3 (L).

## Limits

- The models predict what a 1800+ human plays next. They do not search, so they play weak chess by engine standards.
- The Elo numbers hold for the evaluation setup in the code repo (greedy legal move, openings copied from the first 6 plies of held-out games, limited Stockfish). They do not transfer to other settings.
- About 0.5% to 1.6% of top-scoring moves are illegal. The example above filters them.
- One seed per size. The seed spread in loss is about 0.003 for M and L.

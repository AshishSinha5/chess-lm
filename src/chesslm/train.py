"""Train a GPT on token shards. Example:
python -m chesslm.train --model xs --name xs_test --data /workspace/data/lichess_1800 --max_tokens 1e9
WANDB_MODE defaults to offline. Checkpoints go to /workspace/ckpt/NAME and resume on restart."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import torch
import yaml

os.environ.setdefault("WANDB_MODE", "offline")

import torch._dynamo  # noqa: E402

torch._dynamo.config.automatic_dynamic_shapes = False  # shapes are fixed per length bucket
torch._dynamo.config.cache_size_limit = 64  # one compiled graph per bucket

from chesslm.data.dataset import Shard, loader  # noqa: E402
from chesslm.model.gpt import GPT, GPTConfig  # noqa: E402

A100_BF16_PEAK = 312e12


def lr_at(frac, a):
    """Warmup, constant, then linear decay (WSD). frac = tokens seen / max_tokens."""
    if frac < a.warmup_frac:
        return a.lr * frac / a.warmup_frac
    if frac > 1 - a.decay_frac:
        r = (1 - frac) / a.decay_frac
        return a.lr * (a.min_lr_ratio + (1 - a.min_lr_ratio) * max(r, 0))
    return a.lr


@torch.no_grad()
def evaluate(model, shard, a, device):
    model.eval()
    tot, n = 0.0, 0
    for i, (x, y, _, _) in enumerate(loader(shard, a.batch_tokens, seed=0, epochs=1)):
        if i >= a.eval_batches:
            break
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x.to(device), y.to(device))
        k = (y != -100).sum().item()
        tot += loss.item() * k
        n += k
    model.train()
    return tot / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="xs")
    p.add_argument("--name", required=True)
    p.add_argument("--data", default="/workspace/data/lichess_1800")
    p.add_argument("--max_tokens", type=float, default=1e9)
    p.add_argument("--batch_tokens", type=int, default=65536)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--warmup_frac", type=float, default=0.02)
    p.add_argument("--decay_frac", type=float, default=0.2)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_games", type=int, default=None)
    p.add_argument("--eval_split", default="val")
    p.add_argument("--eval_batches", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--ckpt_every", type=int, default=1000)
    p.add_argument("--save_at", type=str, default="", help="comma list of token counts for extra checkpoints")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--no_compile", action="store_true")
    a = p.parse_args()
    a.max_tokens = int(a.max_tokens)
    save_at = sorted(int(float(s)) for s in a.save_at.split(",") if s)

    device = "cuda"
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    mcfg = yaml.safe_load(open(Path(__file__).parents[2] / "configs/model" / f"{a.model}.yaml"))
    mcfg.pop("name")
    meta = json.load(open(Path(a.data) / "meta.json")) if (Path(a.data) / "meta.json").exists() else {}
    cfg = GPTConfig(vocab_size=meta.get("vocab_size", 1971), **mcfg)
    model = GPT(cfg).to(device)
    n_params = model.n_params()
    decay = [q for q in model.parameters() if q.dim() >= 2]
    no_decay = [q for q in model.parameters() if q.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), fused=True)
    fwd = model if a.no_compile else torch.compile(model)

    ckdir = Path("/workspace/ckpt") / a.name
    ckdir.mkdir(parents=True, exist_ok=True)
    tokens_seen, step, epoch, bie = 0, 0, 0, 0
    last = ckdir / "last.pt"
    if last.exists():
        st = torch.load(last, map_location=device)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        tokens_seen, step, epoch, bie = st["tokens"], st["step"], st["epoch"], st["bie"]
        print(f"resumed at step {step}, tokens {tokens_seen:,}", flush=True)
    else:
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        json.dump({**vars(a), "model_cfg": cfg.__dict__, "n_params": n_params, "git": git,
                   "torch": torch.__version__}, open(ckdir / "config.json", "w"), indent=1)

    train = Shard(a.data, "train", a.max_games)
    evals = Shard(a.data, a.eval_split, a.max_games if a.eval_split == "train" else None)
    print(f"{a.model}: {n_params/1e6:.2f}M params, {train.n:,} train games", flush=True)

    wb = None
    try:
        import wandb
        wb = wandb.init(project="chess-lm", name=a.name, config=vars(a), dir="/workspace")
    except Exception as e:
        print("wandb off:", e)
    logf = open(ckdir / "log.jsonl", "a")

    def log(d):
        logf.write(json.dumps(d) + "\n"); logf.flush()
        if wb:
            wb.log(d, step=d["step"])

    def save(path):
        tmp = path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "tokens": tokens_seen,
                    "step": step, "epoch": epoch, "bie": bie, "model_cfg": cfg.__dict__}, tmp)
        tmp.rename(path)

    model.train()
    t0, win_tok = time.time(), 0
    pending_save = [s for s in save_at if s > tokens_seen]
    for x, y, epoch, bie in loader(train, a.batch_tokens, a.seed, skip=(epoch, bie)):
        frac = tokens_seen / a.max_tokens
        if frac >= 1:
            break
        lr = lr_at(frac, a)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = fwd(x, y)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        k = (y != -100).sum().item()
        tokens_seen += k
        win_tok += k
        step += 1
        if step % a.log_every == 0:
            torch.cuda.synchronize()
            dt = time.time() - t0
            tps = win_tok / dt
            d = {"step": step, "tokens": tokens_seen, "loss": loss.item(), "lr": lr, "grad_norm": gn.item(),
                 "tok_s": tps, "tflops": 6 * n_params * tps / 1e12, "mfu": 6 * n_params * tps / A100_BF16_PEAK}
            log(d)
            print({k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in d.items()}, flush=True)
            t0, win_tok = time.time(), 0
        if step % a.eval_every == 0:
            v = evaluate(model, evals, a, device)
            log({"step": step, "tokens": tokens_seen, "val_loss": v})
            print(f"  eval loss/move {v:.4f}", flush=True)
        if step % a.ckpt_every == 0:
            save(last)
        while pending_save and tokens_seen >= pending_save[0]:
            save(ckdir / f"tok_{pending_save.pop(0)}.pt")
    v = evaluate(model, evals, a, device)
    log({"step": step, "tokens": tokens_seen, "val_loss": v, "final": True})
    save(last)
    save(ckdir / "final.pt")
    print(f"done. final eval loss/move {v:.4f}", flush=True)


if __name__ == "__main__":
    main()

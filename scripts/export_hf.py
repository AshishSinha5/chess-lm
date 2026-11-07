"""Stage the 1B-token size-sweep models for Hugging Face: weights only, as safetensors.
python scripts/export_hf.py [--out /workspace/hf_stage] [--upload --repo sinha/chess-lm]
Layout: xs/ s/ m/ l/, each with model.safetensors and config.json. The top-level README.md is the model card.
The script checks that each file loads and gives the same logits as the .pt checkpoint (CPU, small input)."""
import argparse
import csv
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from chesslm.model.gpt import GPT, GPTConfig

SIZES = {"xs": "size_xs", "s": "size_s", "m": "size_m", "l": "size_l"}
ELO = {"xs": 1182, "s": 1296, "m": 1410, "l": 1479}  # nominal scale, README step 8
CKPT = Path("/workspace/ckpt")
ROOT = Path(__file__).resolve().parent.parent


def stage(out):
    runs = {r["run"]: r for r in csv.DictReader(open(ROOT / "results" / "runs.csv"))}
    rows = []
    for size, run in SIZES.items():
        st = torch.load(CKPT / run / "final.pt", map_location="cpu")
        d = out / size
        d.mkdir(parents=True, exist_ok=True)
        weights = {k: v.contiguous() for k, v in st["model"].items()}
        save_file(weights, d / "model.safetensors", metadata={"format": "pt"})
        cfg = dict(st["model_cfg"])
        meta = {"model_cfg": cfg, "run": run, "tokens": st["tokens"], "seed": 0}
        json.dump(meta, open(d / "config.json", "w"), indent=1)
        # check: same logits from the safetensors file as from the checkpoint
        a = GPT(GPTConfig(**cfg)).eval()
        a.load_state_dict(st["model"])
        b = GPT(GPTConfig(**cfg)).eval()
        b.load_state_dict(load_file(d / "model.safetensors"))
        x = torch.randint(0, cfg["vocab_size"], (2, 16))
        with torch.no_grad():
            same = torch.equal(a(x)[0], b(x)[0])
        assert same, f"{size}: logits differ"
        r = runs[run]
        rows.append((size, r["params"], r["final_val_loss"], ELO[size], (d / "model.safetensors").stat().st_size / 1e6))
        print(f"{size}: ok, {rows[-1][4]:.0f} MB, loss {r['final_val_loss']}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/hf_stage")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--repo", default="sinha/chess-lm")
    a = ap.parse_args()
    out = Path(a.out)
    stage(out)
    if a.upload:
        from huggingface_hub import HfApi

        HfApi().upload_folder(folder_path=str(out), repo_id=a.repo, repo_type="model",
                              allow_patterns=["*/model.safetensors", "*/config.json"],
                              commit_message="Add 1B-token size-sweep weights (XS, S, M, L)")
        print("uploaded to", a.repo)

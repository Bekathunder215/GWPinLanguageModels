"""
Inference / prompting script (Tiny Shakespeare, char-level).
Students will integrate sustainability tracking themselves.

Source: https://github.com/karpathy/nanoGPT
"""

# ----------------------------
# Edit these
# ----------------------------
import argparse
import os
import pickle
from pathlib import Path

import torch
import yaml

from helpers import training_to_gpt_config
from model import GPT, GPTConfig

parser = argparse.ArgumentParser()
parser.add_argument(
    "--exp",
    type=str,
    default="",
    help="experiment config file name (without .yaml)",
)
args = parser.parse_args()
cfg_raw = yaml.safe_load(Path("configs/defaults.yaml").read_text())
scenario = cfg_raw["scenarios"][args.exp]

OUT_DIR = "out"
CKPT_PATH = os.path.join(OUT_DIR, scenario["SAVE_CHECKPOINT_NAME"])
print(f"ckpt path is {Path(CKPT_PATH)}")

PROMPT = "To be, or not to be"
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.5
TOP_K = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------


def load_meta(data_dir: str):
    meta_path = os.path.join(data_dir, "meta.pkl")
    with open(meta_path, "rb") as f:
        return pickle.load(f)


def main():
    ckpt = torch.load(str(CKPT_PATH), map_location=DEVICE)

    # train.py should store config with model parameters and data_dir
    data_dir = ckpt["config"]["data_dir"]
    model_cfg = ckpt["config"]["model"]
    # print(model_cfg)

    meta = load_meta(data_dir)
    stoi = meta["stoi"]  # char to index mapping
    itos = meta["itos"]  # index to char mapping

    def encode(s: str):
        # map unknown chars to a safe fallback if needed
        return [stoi.get(ch, stoi[" "]) for ch in s]

    def decode(tokens):
        return "".join([itos[t] for t in tokens])

    config = GPTConfig(**training_to_gpt_config(model_cfg, 65))
    model = GPT(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(model)
    print(model.get_num_params())

    idx = torch.tensor([encode(PROMPT)], dtype=torch.long, device=DEVICE)

    out = model.generate(
        idx, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=TOP_K
    )

    for i in range(15):
        print(
            f"Token: '{itos[out[0, i].item()]}' | Probability: {out[0, i].item():.4f}"
        )
    print(f"tokens are: {len(out[0])}")
    print(decode(out[0].tolist()))


if __name__ == "__main__":
    main()

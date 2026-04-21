"""
Inference / prompting script (Tiny Shakespeare, char-level).

Prompting FU: generation of 200 characters from a fixed seed prompt,
using the Scenario 1 checkpoint, on fixed hardware in Denmark.
"""

import argparse
import os
import pickle
from pathlib import Path

import torch
import yaml
from codecarbon import OfflineEmissionsTracker

from helpers import training_to_gpt_config
from model import GPT, GPTConfig

parser = argparse.ArgumentParser()
parser.add_argument(
    "--exp",
    type=str,
    required=True,
    help="prompting scenario name (e.g. four_short, five_mid)",
)
args = parser.parse_args()

cfg_raw = yaml.safe_load(Path("configs/defaults.yaml").read_text())
scenario = cfg_raw["prompting_scenarios"][args.exp]

OUT_DIR = "out"
CKPT_PATH = os.path.join(OUT_DIR, scenario["CHECKPOINT_NAME"])
PROMPT = scenario["PROMPT"]
MAX_NEW_TOKENS = scenario["MAX_NEW_TOKENS"]
TEMPERATURE = scenario["TEMPERATURE"]
TOP_K = scenario["TOP_K"]
EMISSIONS_DIR = scenario["EMISSIONS_DIR"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_meta(data_dir: str):
    meta_path = os.path.join(data_dir, "meta.pkl")
    with open(meta_path, "rb") as f:
        return pickle.load(f)


def main():
    ckpt = torch.load(str(CKPT_PATH), map_location=DEVICE)

    data_dir = ckpt["config"]["data_dir"]
    model_cfg = ckpt["config"]["model"]

    meta = load_meta(data_dir)
    stoi = meta["stoi"]
    itos = meta["itos"]

    def encode(s: str):
        return [stoi.get(ch, stoi[" "]) for ch in s]

    def decode(tokens):
        return "".join([itos[t] for t in tokens])

    config = GPTConfig(**training_to_gpt_config(model_cfg, 65))
    model = GPT(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    idx = torch.tensor([encode(PROMPT)], dtype=torch.long, device=DEVICE)

    tracker = OfflineEmissionsTracker(
        output_dir=Path(f"./data/{EMISSIONS_DIR}/"),
        output_file="emissions.csv",
        measure_power_secs=1,
        save_to_file=True,
        cloud_provider="gcp",
        cloud_region="europe-west1",
        on_csv_write="append",
    )

    tracker.start()
    out = model.generate(
        idx, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=TOP_K
    )
    emissions_kg = tracker.stop()

    generated_text = decode(out[0].tolist())
    generated_chars = len(generated_text) - len(PROMPT)

    print(decode(out[0].tolist()))

    total_emissions_g = emissions_kg * 1000
    print("\n--- Prompting Functional Unit Report ---")
    print(f"Scenario: {args.exp}")
    print(f"Model: {scenario['CHECKPOINT_NAME']} | Temp: {TEMPERATURE} | Max tokens: {MAX_NEW_TOKENS}")
    print(f"Characters generated: {generated_chars}")
    print(f"Total Emissions: {total_emissions_g:.6f} gCO2eq")


if __name__ == "__main__":
    main()

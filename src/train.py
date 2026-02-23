"""
Course training script (simplified from nanoGPT).

Focus:
- Train a small GPT-style model from scratch on a tiny dataset.
- Students will integrate sustainability tracking themselves.

Source: https://github.com/karpathy/nanoGPT
"""

import os
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

# -----------------------------------------------------------------------------
# Experiment configuration
import yaml
from codecarbon import OfflineEmissionsTracker

from model import GPT, GPTConfig


@dataclass
class TrainingConfig:
    OUT_DIR: str
    DATA_DIR: str
    EVAL_INTERVAL: int
    EVAL_ITERS: int
    LOG_INTERVAL: int
    SAVE_CHECKPOINT: bool
    N_LAYER: int
    N_HEAD: int
    N_EMBD: int
    DROPOUT: float
    BIAS: bool
    SEED: int
    DEVICE: str
    DTYPE: str
    BATCH_SIZE: int
    BLOCK_SIZE: int
    MAX_ITERS: int
    LEARNING_RATE: float
    WEIGHT_DECAY: float
    GRAD_CLIP: float
    EMISSIONS_DIR: str


cfg_raw = yaml.safe_load(Path("configs/defaults.yaml").read_text())
scenario = cfg_raw["scenarios"]["two"]
cfg = TrainingConfig(**scenario)

print(f"cofg is {cfg}")

# I/O
OUT_DIR = cfg.OUT_DIR
DATA_DIR = os.path.join(cfg.DATA_DIR)
EVAL_INTERVAL = cfg.EVAL_INTERVAL
EVAL_ITERS = cfg.EVAL_ITERS
LOG_INTERVAL = cfg.LOG_INTERVAL
SAVE_CHECKPOINT = cfg.SAVE_CHECKPOINT

# Model (main tunables)
N_LAYER = cfg.N_LAYER
N_HEAD = cfg.N_HEAD
N_EMBD = cfg.N_EMBD
DROPOUT = cfg.DROPOUT
BIAS = cfg.BIAS

# Training (main parameters you can also experiment with)
SEED = cfg.SEED
DEVICE = torch.device(
    cfg.DEVICE
)  # If you can, try also seeing consumption when using gpu (change this to 'cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = cfg.DTYPE
BATCH_SIZE = cfg.BATCH_SIZE  # Number of sequences processed in parallel.
BLOCK_SIZE = cfg.BLOCK_SIZE  # Maximum context length for predictions (e.g. 128 or 256). The longer the block size, the more memory and compute it requires, but it can also lead to better performance.
MAX_ITERS = cfg.MAX_ITERS  # Total number of training iterations. The more iterations, the better the model can perform, but it also takes more time and energy to train.
LEARNING_RATE = float(cfg.LEARNING_RATE)
WEIGHT_DECAY = cfg.WEIGHT_DECAY  # L2 Regularization
GRAD_CLIP = cfg.GRAD_CLIP  # To prevent exploding gradients

# ----------------------------------------------------


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_meta(data_dir: str):
    meta_path = os.path.join(data_dir, "meta.pkl")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "rb") as f:
        return pickle.load(f)


def get_batch(split: str, data_dir: str, block_size: int, batch_size: int, device: str):
    # simple, robust memmap loader
    bin_path = os.path.join(data_dir, f"{split}.bin")
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")

    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )

    x = x.to(device)
    y = y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(
    model: GPT,
    data_dir: str,
    block_size: int,
    batch_size: int,
    device: str,
    eval_iters: int,
):
    model.eval()
    losses = {}
    for split in ["train", "val"]:
        split_losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            x, y = get_batch(split, data_dir, block_size, batch_size, device)
            _, loss = model(x, y)
            split_losses[k] = loss
        losses[split] = split_losses.mean().item()
    model.train()
    return losses


def save_checkpoint(
    out_dir: str,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    iter_num: int,
    config: dict,
):
    os.makedirs(out_dir, exist_ok=True)
    ckpt = {
        "iter_num": iter_num,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "config": config,
    }
    torch.save(ckpt, os.path.join(out_dir, "ckpt.pt"))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(SEED)
    tracker = OfflineEmissionsTracker(
        output_dir=Path(f"./data/{cfg.EMISSIONS_DIR}/"),
        output_file="emissions.csv",
        measure_power_secs=10,
        save_to_file=True,
        cloud_provider="gcp",
        cloud_region="europe-west1",
    )

    meta = load_meta(DATA_DIR)
    print(meta)
    vocab_size = meta["vocab_size"] if meta and "vocab_size" in meta else 50304

    GPTcfg = GPTConfig(
        block_size=BLOCK_SIZE,
        vocab_size=vocab_size,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        dropout=DROPOUT,
        bias=BIAS,
    )

    # create the model and move it to the device
    model = GPT(GPTcfg).to(DEVICE)

    # create the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    # (optional) uncomment this for printing model size once
    # print(f"Device: {DEVICE}")
    # print(f"Model parameters: {model.get_num_params():,}")
    # print(f"Training for {MAX_ITERS} iterations | batch={BATCH_SIZE} | block={BLOCK_SIZE}")

    tracker.start()

    t0 = time.time()
    for it in range(MAX_ITERS + 1):
        # periodic evaluation
        if it % EVAL_INTERVAL == 0:
            losses = estimate_loss(
                model, DATA_DIR, BLOCK_SIZE, BATCH_SIZE, DEVICE, EVAL_ITERS
            )
            dt = time.time() - t0
            print(
                f"iter {it:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | elapsed {dt:.1f}s"
            )
            tracker.flush()

            if SAVE_CHECKPOINT and it > 0:
                config_dump = {
                    "data_dir": DATA_DIR,
                    "train": {
                        "batch_size": BATCH_SIZE,
                        "block_size": BLOCK_SIZE,
                        "max_iters": MAX_ITERS,
                        "learning_rate": LEARNING_RATE,
                        "weight_decay": WEIGHT_DECAY,
                        "grad_clip": GRAD_CLIP,
                        "dtype": DTYPE,
                        "device": DEVICE,
                    },
                    "model": asdict(cfg),
                }
                save_checkpoint(OUT_DIR, model, optimizer, it, config_dump)

        # training step
        x, y = get_batch("train", DATA_DIR, BLOCK_SIZE, BATCH_SIZE, DEVICE)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if GRAD_CLIP and GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()

        if it % LOG_INTERVAL == 0:
            print(f"iter {it:5d} | loss {loss.item():.4f}", end="\n")

    tracker.stop()
    print("Training completed.")

    # Save final checkpoint
    if SAVE_CHECKPOINT:
        config_dump = {
            "data_dir": DATA_DIR,
            "train": {
                "batch_size": BATCH_SIZE,
                "block_size": BLOCK_SIZE,
                "max_iters": MAX_ITERS,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "grad_clip": GRAD_CLIP,
                "dtype": DTYPE,
                "device": DEVICE,
            },
            "model": asdict(cfg),
        }
        save_checkpoint(OUT_DIR, model, optimizer, MAX_ITERS, config_dump)


if __name__ == "__main__":
    main()

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Educational case study for **1210X Quantitative Methods to Assess Sustainability** at DTU. Trains a small GPT-style decoder-only transformer on Tiny Shakespeare and quantitatively assesses sustainability across environmental, economic, and social dimensions from a life-cycle perspective. Full task description: `docs/SLMs_Case_Description.md`.

## Current Task: Task II — Sustainability Assessment

Three dimensions required:

**Environmental** (primary codebase focus):
- CodeCarbon measurements for training (done) and prompting (needs HPC runs)
- Comparative analysis across training scenarios (1/2/3) and prompting scenarios (4/5)
- Grandfathering approach: allocate environmental boundary to FU based on IT sector's historical ~2% share of global carbon budget; quantify distance per scenario
- Optional: economic allocation comparison

**Economic** (no additional code needed — use existing outputs):
- Estimate cloud cost from measured training durations + GPU type (V100 equivalent)
- Simplified LCC: compute time cost, hardware, energy
- Discuss cost scaling with model size, duration, usage intensity

**Social** (qualitative — no code):
- Stakeholder groups affected by energy/resource use and privacy concerns
- Human health risks per group

**Task III** (later):
- Executive summary + infographic
- EcoLogits benchmark: scale SLM results to compare against a production LLM
- Rebound effects, Circular Economy (Narrowing/Closing/Slowing)

## Functional Units

**Training FU**: One complete training run on Tiny Shakespeare — 3000 gradient update steps, batch size 64, block size 256, DTU HPC (V100), Denmark electricity grid. Metric: gCO₂eq per training run.

**Prompting FU**: Generation of 200 characters from a fixed seed prompt (`"To be, or not to be"`), using the Scenario 1 checkpoint as the fixed baseline model, DTU HPC (V100), Denmark electricity grid. Metric: gCO₂eq per generation. The trained model is a fixed system parameter, not part of the FU boundary.

## Scenarios

**Training** — single-parameter architecture sensitivity, baseline = Scenario 1 (`configs/defaults.yaml` → `scenarios`):
- `one`: 4L, 4H, 128D, ~834K params (baseline)
- `two`: 8L, 8H, 128D, ~1.6M params (deeper + more heads)
- `three`: 4L, 4H, 256D (wider embedding)

**Prompting** — single-parameter inference sensitivity, all use `one.pt` (`configs/defaults.yaml` → `prompting_scenarios`):
- Scenario 4 — output length: `four_short` (50 tokens), `four_mid` (200), `four_long` (400)
- Scenario 5 — temperature: `five_low` (0.1), `five_mid` (0.5), `five_high` (1.5)

**Known result:** Scenario 5 (temperature) will show near-zero emissions variance — temperature only rescales logits and does not affect compute per token. This is a valid and discussable finding.

## Environment Setup

```bash
conda env create -f env_requirements.yaml
conda activate slm-sustainability
```

## Common Commands

**Data preparation (run once):**
```bash
python data/prepare.py
```

**Training (submit on DTU HPC):**
```bash
bsub < script.sh    # scenario one
bsub < script2.sh   # scenario two
bsub < script3.sh   # scenario three
```

**Prompting (submit on DTU HPC):**
```bash
bsub < scriptPrompt4short.sh
bsub < scriptPrompt4mid.sh
bsub < scriptPrompt4long.sh
bsub < scriptPrompt5low.sh
bsub < scriptPrompt5mid.sh
bsub < scriptPrompt5high.sh
```

**Check job status on HPC:**
```bash
bjobs
```

## Architecture

**Model** (`src/model.py`): Decoder-only GPT transformer. `GPTConfig` controls `n_layer`, `n_head`, `n_embd`, `block_size`, `dropout`, `bias`. Weight tying between input embedding and output projection.

**Training** (`src/train.py`): Loads from `scenarios` block in `defaults.yaml`. CodeCarbon wraps the full training loop. Reports total gCO₂eq per training run. Saves checkpoints to `out/`, emissions CSV to `data/{EMISSIONS_DIR}/emissions.csv`.

**Prompting** (`src/prompt.py`): Loads from `prompting_scenarios` block in `defaults.yaml`. CodeCarbon wraps only `model.generate()`. Reports gCO₂eq for the generation. Emissions CSV saved to `data/{EMISSIONS_DIR}/emissions.csv`.

**Config** (`configs/defaults.yaml`): Two top-level keys — `scenarios` (training) and `prompting_scenarios` (inference). Each scenario varies exactly one parameter from the baseline.

**Data** (`data/prepare.py`): Downloads Tiny Shakespeare, builds character-level vocab (65 chars), saves `train.bin`, `val.bin`, `meta.pkl`.

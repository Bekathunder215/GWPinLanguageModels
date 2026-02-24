# 2. Object of assessment (SLM + functional unit)

## Scenario 1

- **Object assessed**: a Small Language Model using a decoder-only GPT-style Transformer
  (character-level autoregressive LM).
- **Architecture**: token embedding + positional embedding, masked multi-head self-attention, MLP,
  residual connections, LayerNorm, linear LM head with tied weights.
- **Configured size**: 4 layers, 4 heads, embedding size 128, context length 256, dropout 0.1.
- **Approx. parameter count**: ≈ 834,432 ≈834,432 parameters ($\approx 0.83$M, with vocab size 65
  from Tiny Shakespeare).
- **Hardware used**: Apple M2 on macOS; CodeCarbon logs show CPU execution (GPU utilization reported
  as 0%).
- **Training dataset**: Tiny Shakespeare, char-level, 1,115,394, 1,115,394 tokens total, split into
  1,003,854, 1,003,854 train and, 111,540 validation (90/10).
- **Generated tokens (inference FU)**: 200 new tokens per prompt (plus prompt tokens as context).
- **Training workload**: 500 iterations with batch 32 and block 256, about
  $(32 \times 256 \times 501 = 4.10)$M token positions processed.


## Scenario 2

- **Object assessed**: a Small Language Model using a decoder-only GPT-style Transformer
  (character-level autoregressive LM).
- **Architecture**: token embedding + positional embedding, masked multi-head self-attention, MLP,
  residual connections, LayerNorm, linear LM head with tied weights.
- **Configured size**: 8 layers, 8 heads, embedding size 128, context length 256, dropout 0.2.
- **Approx. parameter count**: (\approx 1{,}627{,}520) parameters ((\approx 1.63)M, with vocab size
  65 from Tiny Shakespeare).
- **Hardware used**: Apple M2 on macOS; CodeCarbon logs show CPU execution (GPU utilization reported
  as 0%).
- **Training dataset**: Tiny Shakespeare, char-level, (1{,}115{,}394) tokens total; split into
  (1{,}003{,}854) train and (111{,}540) validation (90/10).
- **Generated tokens** (inference FU): 200 new tokens per prompt (plus prompt tokens as context).
- **Training workload**: 500 iterations with batch 64 and block 256 (\Rightarrow) about (64 \times
  256 \times 501 = 8.21)M token positions processed.


# 3. System boundaries (with exclusions and justification)

### Included stages:

- dataset preparation
- model training compute
- inference/prompt compute
- electricity-to-emissions conversion (via CodeCarbon logs)

### Excluded for limited relevance:

- end-user display/device use after text generation
- network transfer (local run)
- model serving infrastructure not used in this local setup

### Excluded for simplification:

- hardware manufacturing/transport/end-of-life
- embodied impacts of Apple M2
- full software stack lifecycle (OS/framework production)

### Important distinction sentence:

“Exclusions due to limited relevance are omitted because they contribute marginally to this
case-study FU, while simplification exclusions are potentially relevant but removed to keep the
assessment tractable with available data.” If you want, I can now turn this into a polished report
subsection (ready to paste), including a one-figure system-boundary diagram in Mermaid.

# Scenarios

### Scenario 1

```bash
    # I/O
    OUT_DIR: "out"
    DATA_DIR: "data"
    EVAL_INTERVAL: 200
    EVAL_ITERS: 50
    LOG_INTERVAL: 50
    SAVE_CHECKPOINT: True
    EMISSIONS_DIR: "two"

    # Model (main tunables)
    N_LAYER: 8
    N_HEAD: 8
    N_EMBD: 128
    DROPOUT: 0.2
    BIAS: True

    # Training (main parameters you can also experiment with)
    SEED: 1
    DEVICE: "cpu"
    DTYPE: "float32"
    BATCH_SIZE: 64
    BLOCK_SIZE: 256
    MAX_ITERS: 500
    LEARNING_RATE: 2e-4
    WEIGHT_DECAY: 0.1
    GRAD_CLIP: 1.0
```

### Scenario 2

```bash
    # I/O
    OUT_DIR: "out"
    DATA_DIR: "data"
    EVAL_INTERVAL: 200
    EVAL_ITERS: 50
    LOG_INTERVAL: 50
    SAVE_CHECKPOINT: True
    EMISSIONS_DIR: "one"

    # Model (main tunables)
    N_LAYER: 4
    N_HEAD: 4
    N_EMBD: 128
    DROPOUT: 0.1
    BIAS: True

    # Training (main parameters you can also experiment with)
    SEED: 1
    DEVICE: "cpu"
    DTYPE: "float32"
    BATCH_SIZE: 32
    BLOCK_SIZE: 256
    MAX_ITERS: 500
    LEARNING_RATE: 3e-4
    WEIGHT_DECAY: 0.1
    GRAD_CLIP: 1.0

```

# 1. Defining the Training Functional Unit

**Functional Unit**: "The processing of 4,096 Million training tokens during the optimization
phase."

Why 1 Million? In your scenario 1 (Batch 32, Block 256), a single iteration only processes 8,192
tokens. Scaling to 4,096M tokens provides a more readable "carbon price tag" for the experiments.

What it measures: It specifically isolates the impact of your model architecture (N_LAYER, N_EMBD)
and hardware efficiency (DEVICE).

# 2. Implementation Strategy

**Tokens per Iteration**: BATCH_SIZE×BLOCK_SIZE.

**Total Tokens in Run**: (Iterations Completed)×(Tokens per Iteration).

**Efficiency Metric**: $( \frac{\text{Total Tokens}}{\text{Total Emission}})×1,000,000$.

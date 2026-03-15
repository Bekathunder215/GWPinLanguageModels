Project Document: Functional Unit Definition

# 1. Core Definition

Functional Unit: "The generation of 1,000 tokens of text with a context window of 256 characters,
using a vocabulary size of 69."

Purpose: This unit serves as the reference point to compare the energy efficiency and carbon
intensity of different model architectures (varying layers, heads, or embedding sizes) across
different training scenarios.

# 2. Technical Mapping

**Vocabulary (V)**: Approximately 69 characters (standard alphanumeric + punctuation).

**Context Window (L)**: 256 tokens (**BLOCK_SIZE**). (This defines the complexity of the attention
mechanism $(O(L^2))$ being measured.)

**Model Capacity**: Defined by **N_LAYER, N_HEAD, and N_EMBD** TrainingConfig.

# 3. Calculation Methodology (LCA Integration)

To translate the training run into this functional unit, we will use the data from emissions.csv and
the following formula:

## Total Tokens Processed ($T_total}):

$$
T_{total} = MAX_ITERS x BATCH_SIZE x BLOCK_SIZE
$$

## Carbon Intensity per Unit ($CI_{unit}$):

$$
CI_{\text{unit}} = \left(\frac{\text{Total Emissions (gCO}_2\text{e)}}{T_{\text{total}}}\right) \times 1000
$$

# 4. Why this Functional Unit?

**Invariance**: It focuses on the output (text generation) rather than the process (training time),
allowing you to compare a fast, power-hungry GPU against a slow, efficient CPU.

**Comparability**: By fixing the BLOCK_SIZE at 256, you ensure that "memory effort" is normalized
across different tests.

**Sustainability Insight**: It answers the specific question: "How much carbon does it cost to
'write' a page of text with this specific model configuration?"

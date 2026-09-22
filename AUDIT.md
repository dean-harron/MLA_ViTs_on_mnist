# Original Notebook Audit

This document records the important findings from the original `visual_transformer_MLA.ipynb` I used an AI to assist in debugging and refactoring parts of the code to make it easier to transfer the code from one giant notebook file to separate and well structured python modules.

## Issues in the jupyter notebook version

### 1. MLA was called with the wrong interface

`MultiHeadLatentAttention.forward()` accepts a single `hidden_states` tensor plus optional cache/mask arguments. The transformer block instead called the module with the normalized tensor three times, matching a conventional Q/K/V attention API.

That mismatch produced the notebook's recorded runtime error during training:

```text
RuntimeError: Tensors must have same number of dimensions: got 2 and 3
```

The refactored transformer block now calls MLA with one tensor.

### 2. Causal masking was incompatible with the ViT encoder

The original MLA implementation unconditionally created a causal attention mask. The model prepended a CLS token and then classified from that token.

For an image encoder, the CLS token needs access to all patch tokens. A causal mask allows position 0 to see only position 0, so the classification token cannot directly aggregate later patch information.

The refactored implementation defaults to `causal_attention=False`.

### 3. `forward_absorbed()` had an invalid sequence contraction

The original content-score computation did not retain a separate key-token dimension. The corrected contraction explicitly computes query-token × key-token scores:

```text
q_latent: (B, Tq, H, R)
c_kv:     (B, Tk, R)
scores:   (B, H, Tq, Tk)
```

### 4. The absorbed path was not behaviorally symmetric with the standard path

The standard path could apply attention dropout while the absorbed path did not. The two paths now use the same dropout behavior.

### 5. Configuration was duplicated and inconsistent

Notebook-level global variables and `MLAConfig` overlapped. The dataclass also had `v_head_dim=96` as its default even though the actual notebook model supplied `v_head_dim=16`.

The repository has one source of truth in `ModelConfig`.

### 6. The test split was called validation data

MNIST's standard `train=False` split is the test set. The repository uses `test_loader` and "test accuracy" terminology to avoid implying a separate validation split exists.

### 7. Notebook-specific and redundant code was removed

The package removes:

- shell installation commands,
- repeated imports,
- repeated evaluation loops,
- exploratory shape-printing cells,
- duplicate global configuration,
- the notebook-only save/load sequence.

Equivalent functionality is available as reusable modules and scripts.

### 8. Notebook execution state was inconsistent

The uploaded notebook records a training-cell runtime error in its stored output, but later cells still contain evaluation/checkpoint outputs such as `86.31%` accuracy. This means the notebook's displayed execution history is not a reliable single successful run and should not be treated as reproducible evidence.

## Validation performed on the refactored implementation

- Python bytecode compilation: passed.
- Pytest suite: passed (4 tests).
- Forward + backward smoke test: passed with finite gradients.
- Synthetic train/evaluate smoke test: passed.
- Checkpoint save/reload: passed with matching outputs.
- Standard MLA vs. absorbed MLA: matched within floating-point tolerance; the observed maximum absolute difference was approximately `5.96e-08`.

## Scope

The refactor preserves the notebook's compact architecture rather than turning it into a large production framework. The implementation is intended for learning, experimentation, and GitHub presentation.

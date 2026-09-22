# MNIST Vision Transformer with Multi-Head Latent Attention

A clean, modular PyTorch implementation of **Multi-Head Latent Attention (MLA)** used inside a small Vision Transformer (ViT) for MNIST digit classification.

The original implementation was developed as a Jupyter notebook. This repository separates the implementation into reusable modules, adds validation/tests, and provides command-line training and evaluation entry points. Please note that this implementation is an experiment for MLA ViT, traditional CNNs have outperformed ViTs on the MNIST Dataset in our experiments for the same epochs. It should rather be treated as an experimental VLM prototype of MLA.

## What this repository demonstrates

- Patch embedding for 28×28 MNIST images.
- A small pre-norm Vision Transformer encoder.
- Multi-Head Latent Attention with:
  - low-rank query compression,
  - low-rank compressed KV state,
  - a decoupled RoPE component,
  - RMSNorm on the compressed Q/KV latents,
  - a standard attention path,
  - a weight-absorbed path that avoids materializing per-head K/V from the cached latent.
- Reproducible training and evaluation.
- Checkpoint saving/loading.
- Prediction visualization.
- Unit tests for tensor shapes, masking, and equivalence between the standard and absorbed MLA paths.

## Important implementation note

Once again, this is an **educational MLA implementation**, not a drop-in reproduction of a specific production LLM implementation. The code preserves the core latent-attention ideas from the notebook while adapting them to an encoder-only vision model.

### Why the vision model does not use causal attention

The original attention class always created a causal mask. That is appropriate for autoregressive decoder-style attention, but it is not appropriate for a standard ViT encoder.

With a sequence `[CLS, patch_1, patch_2, ...]`, a causal mask would prevent the CLS token at position 0 from attending to the image patches at later positions. The refactored model therefore defaults to **bidirectional attention** (`causal=False`).

Causal masking remains available in the MLA module for testing and decoder-style experiments.

## Architecture

```text
MNIST image (1 × 28 × 28)
        │
PatchEmbedding
Conv2d(kernel=7, stride=7)
        │
16 image patches × hidden_size
        │
        ├── prepend learned [CLS] token
        ├── add learned positional embeddings
TransformerEncoderBlock × N
        │
        ├── RMSNorm
        ├── Multi-Head Latent Attention
        │     ├── Q compression
        │     ├── KV compression
        │     ├── decoupled RoPE
        │     └── latent attention
        ├── residual
        ├── RMSNorm
        ├── MLP
        └── residual
        │
[CLS] representation
        │
RMSNorm + Linear(10)
        │
MNIST logits
```

## Repository layout

```text
mnist-mla-vit/
├── .github/
│   └── workflows/
│       └── tests.yml
├── notebooks/
│   └── README.md
├── scripts/
│   ├── evaluate.py
│   ├── inspect_model.py
│   └── train.py
├── src/
│   └── mla_mnist/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── evaluation.py
│       ├── training.py
│       ├── utils.py
│       └── models/
│           ├── __init__.py
│           ├── attention.py
│           └── vit.py
├── tests/
│   ├── test_attention.py
│   └── test_model.py
├── .gitignore
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Installation

Python 3.10+ is supported.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install the project:

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Train

The default configuration is intentionally small so it is practical for experimentation:

```bash
python scripts/train.py
```

Or use the installed entry point:

```bash
mla-train
```

Useful overrides:

```bash
python scripts/train.py --epochs 5 --batch-size 64 --lr 0.001
python scripts/train.py --device cpu
python scripts/train.py --seed 42
```

MNIST is downloaded automatically into `./data/` if it is not already present.

Training writes generated files under:

```text
artifacts/
├── checkpoints/
│   ├── best.pt
│   └── last.pt
└── figures/
    └── predictions.png
```

## Evaluate

Evaluate a saved checkpoint:

```bash
python scripts/evaluate.py --checkpoint artifacts/checkpoints/best.pt
```

You can also save a prediction grid:

```bash
python scripts/evaluate.py \
    --checkpoint artifacts/checkpoints/best.pt \
    --save-figure artifacts/figures/evaluation_predictions.png
```

## Run the tests

```bash
pytest
```

The attention tests include a direct numerical check that the standard MLA implementation and the weight-absorbed implementation produce matching outputs within floating-point tolerance.

## Inspect the model

```bash
python scripts/inspect_model.py
```

This prints the model configuration, parameter count, and a sample forward-pass tensor shape.

## Configuration

All model and training defaults live in:

```text
src/mla_mnist/config.py
```

The main `ModelConfig` mirrors the dimensions used in the original notebook while removing duplicated global variables.

Typical MLA parameters are:

```python
q_lora_rank = 16
kv_lora_rank = 8
qk_nope_head_dim = 16
qk_rope_head_dim = 8
v_head_dim = 16
```

## Refactor and bug-fix summary
I thought I should include this for whoever might find it useful or if you encounter(ed) similar bugs in your implementation.
The notebook contained several issues that were corrected in the modular version:

1. **Incorrect attention call signature**
   - The transformer block passed the same tensor three times, as if MLA accepted separate Q/K/V inputs.
   - The MLA implementation actually accepts a single hidden-state tensor because it performs its own Q/K/V projections.
   - The refactored block passes the normalized hidden states once.

2. **Incorrect causal masking for ViT**
   - The original MLA path always applied a causal mask.
   - This prevents `[CLS]` from attending to later image patches.
   - The encoder now uses bidirectional attention by default.

3. **Broken `forward_absorbed()` score contraction**
   - The original content-score einsum omitted the key-sequence dimension.
   - It was replaced with the correct query-token × key-token contraction.

4. **Missing dropout parity in the absorbed path**
   - The standard path applied attention dropout during training.
   - The absorbed path now follows the same behavior.

5. **Duplicated configuration**
   - Hyperparameters were defined both as notebook globals and inside `MLAConfig`.
   - Configuration is now centralized.

6. **Inconsistent `v_head_dim` defaults**
   - The notebook dataclass default used `96`, while the actual model used `16`.
   - The repository uses one consistent default.

7. **Notebook installation code removed**
   - `!pip install torchvision` is not part of the implementation.
   - Dependencies belong in `pyproject.toml` / `requirements.txt`.

8. **Ad-hoc evaluation and shape-debug cells removed**
   - Repeated evaluation logic is now reusable Python code.
   - Shape inspection is available through `scripts/inspect_model.py`.

9. **Reproducibility added**
   - A seed utility controls Python, NumPy, and PyTorch random generators.
   - DataLoader behavior is configured through the training pipeline.

10. **Checkpointing made explicit**
    - Best and last checkpoints are saved with configuration and training metadata.

## Notes on MLA and KV-cache reduction

A central motivation of MLA is to cache a compact latent representation instead of storing a full independently projected key/value tensor for every head.

For this implementation, the cache is represented by:

```text
(c_kv, k_pe)
```

where `c_kv` is the compressed KV latent and `k_pe` is the decoupled RoPE key component.

The `forward_absorbed()` path demonstrates how the key up-projection can be algebraically absorbed into the query path and how the value up-projection can be applied after mixing in latent space.

Because MNIST classification is an encoder task, this repository primarily uses the standard forward path during training. The absorbed path is retained and tested so the implementation can be studied and extended toward autoregressive decoding experiments.

## Source notebook

The repository intentionally does **not** copy the original monolithic notebook into the main source tree. The notebook's logic has been reorganized into importable modules.

The `notebooks/` directory is reserved for future demonstrations built on top of the package.

## License

MIT.

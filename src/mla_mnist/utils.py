"""Training utilities."""

from __future__ import annotations

import random
from pathlib import Path
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed common random number generators for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device: str) -> torch.device:
    """Resolve 'auto' or an explicit PyTorch device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def count_parameters(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ensure_parent(path: str | Path) -> Path:
    """Create parent directories for a file path and return the Path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

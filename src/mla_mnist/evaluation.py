"""Evaluation utilities and command-line entry point."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch
from torch import nn

from .config import ModelConfig
from .data import get_mnist_loaders
from .models.vit import VisionTransformer
from .utils import resolve_device


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,) -> tuple[float, float]:
    """Return mean loss and accuracy over a dataset."""
    model.eval()
    criterion = criterion or nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


def load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,) -> tuple[VisionTransformer, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = ModelConfig(**checkpoint["model_config"])
    model = VisionTransformer(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def save_prediction_grid(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    path: str | Path,
    num_images: int = 20,) -> None:
    """Save a grid of predictions from one test batch."""
    model.eval()
    images, labels = next(iter(loader))
    images = images[:num_images]
    labels = labels[:num_images]

    logits = model(images.to(device))
    predictions = logits.argmax(dim=1).cpu()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = 4
    cols = (num_images + rows - 1) // rows
    figure = plt.figure(figsize=(3 * cols, 3 * rows))

    for index in range(num_images):
        axis = figure.add_subplot(rows, cols, index + 1)
        axis.imshow(images[index].squeeze(0), cmap="gray")
        axis.set_title(
            f"Pred: {predictions[index].item()} | True: {labels[index].item()}"
        )
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved MNIST MLA-ViT checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-figure", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)

    _, test_loader = get_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    loss, accuracy = evaluate_model(model, test_loader, device)

    print(f"Device: {device}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Validation/test loss: {loss:.4f}")
    print(f"Validation/test accuracy: {accuracy:.2%}")

    if args.save_figure:
        save_prediction_grid(model, test_loader, device, args.save_figure)
        print(f"Saved prediction grid: {args.save_figure}")

if __name__ == "__main__":
    main()

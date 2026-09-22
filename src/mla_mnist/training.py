"""Training loop and command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW

from .config import ModelConfig, TrainConfig
from .data import get_mnist_loaders
from .evaluation import evaluate_model, save_prediction_grid
from .models.vit import VisionTransformer
from .utils import count_parameters, ensure_parent, resolve_device, seed_everything


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    log_interval: int,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

        if batch_idx == 1 or batch_idx % log_interval == 0:
            print(
                f"  batch {batch_idx:4d}/{len(loader):4d} "
                f"loss={loss.item():.4f}"
            )

    return running_loss / total, correct / total


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    model_config: ModelConfig,
    train_config: TrainConfig,
    val_accuracy: float,
) -> None:
    path = ensure_parent(path)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model_config.to_dict(),
            "train_config": train_config.__dict__,
            "val_accuracy": val_accuracy,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MNIST MLA-ViT model.")
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--data-dir", type=str, default=TrainConfig.data_dir)
    parser.add_argument("--output-dir", type=str, default=TrainConfig.output_dir)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--device", type=str, default=TrainConfig.device)
    parser.add_argument("--log-interval", type=int, default=TrainConfig.log_interval)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_config = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        device=args.device,
        log_interval=args.log_interval,
    )
    model_config = ModelConfig()

    seed_everything(train_config.seed)
    device = resolve_device(train_config.device)

    train_loader, test_loader = get_mnist_loaders(
        data_dir=train_config.data_dir,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
    )

    model = VisionTransformer(model_config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    print(f"Device: {device}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    output_dir = Path(train_config.output_dir)
    best_path = output_dir / "checkpoints" / "best.pt"
    last_path = output_dir / "checkpoints" / "last.pt"

    best_accuracy = -1.0

    for epoch in range(1, train_config.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            train_config.log_interval,
        )
        val_loss, val_accuracy = evaluate_model(
            model,
            test_loader,
            device,
            criterion,
        )

        print(
            f"Epoch {epoch:02d}/{train_config.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.2%} | "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.2%}"
        )

        save_checkpoint(
            last_path,
            model,
            optimizer,
            epoch,
            model_config,
            train_config,
            val_accuracy,
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                model_config,
                train_config,
                val_accuracy,
            )

    # Save the final model's configuration separately for easy inspection.
    config_path = output_dir / "config.json"
    ensure_parent(config_path)
    config_path.write_text(
        json.dumps(
            {
                "model": model_config.to_dict(),
                "training": train_config.__dict__,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    prediction_path = output_dir / "figures" / "predictions.png"
    save_prediction_grid(model, test_loader, device, prediction_path)

    print(f"Best validation accuracy: {best_accuracy:.2%}")
    print(f"Best checkpoint: {best_path}")
    print(f"Prediction grid: {prediction_path}")


if __name__ == "__main__":
    main()

"""Central configuration for the MNIST MLA-ViT experiment."""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ModelConfig:
    """Model and attention hyperparameters."""
    image_size: int = 28
    patch_size: int = 7
    num_channels: int = 1
    num_classes: int = 10

    hidden_size: int = 16
    num_heads: int = 1
    mlp_dim: int = 16
    num_layers: int = 1

    q_lora_rank: Optional[int] = 16
    kv_lora_rank: int = 8
    qk_nope_head_dim: int = 16
    qk_rope_head_dim: int = 8
    v_head_dim: int = 16

    rope_theta: float = 10_000.0
    max_position_embeddings: int = 2048
    attention_dropout: float = 0.0
    causal_attention: bool = False

    def __post_init__(self) -> None:
        if self.image_size <= 0 or self.patch_size <= 0:
            raise ValueError("image_size and patch_size must be positive.")
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        if self.num_channels <= 0 or self.num_classes <= 0:
            raise ValueError("num_channels and num_classes must be positive.")
        if self.hidden_size <= 0 or self.num_heads <= 0:
            raise ValueError("hidden_size and num_heads must be positive.")
        if self.mlp_dim <= 0 or self.num_layers <= 0:
            raise ValueError("mlp_dim and num_layers must be positive.")
        if self.q_lora_rank is not None and self.q_lora_rank <= 0:
            raise ValueError("q_lora_rank must be positive or None.")
        if self.kv_lora_rank <= 0:
            raise ValueError("kv_lora_rank must be positive.")
        if self.qk_nope_head_dim <= 0 or self.qk_rope_head_dim <= 0:
            raise ValueError("attention head dimensions must be positive.")
        if self.v_head_dim <= 0:
            raise ValueError("v_head_dim must be positive.")
        if self.qk_rope_head_dim % 2 != 0:
            raise ValueError("qk_rope_head_dim must be even for rotary embeddings.")
        if self.max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive.")
        if self.sequence_length > self.max_position_embeddings:
            raise ValueError(
                "sequence_length must not exceed max_position_embeddings.")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1).")

    @property
    def num_patches(self) -> int:
        patches_per_side = self.image_size // self.patch_size
        return patches_per_side**2

    @property
    def sequence_length(self) -> int:
        return self.num_patches + 1  # + [CLS]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainConfig:
    """Training and data-loader configuration."""

    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    data_dir: str = "data"
    output_dir: str = "artifacts"
    num_workers: int = 0
    device: str = "auto"
    log_interval: int = 100

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive.")

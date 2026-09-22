"""Vision Transformer built around Multi-Head Latent Attention."""

from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig
from .attention import MultiHeadLatentAttention, RMSNorm


class PatchEmbedding(nn.Module):
    """Split an image into non-overlapping patches and project them."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_patches = config.num_patches

        self.projection = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected image tensor with 4 dimensions, got {x.ndim}.")
        if x.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"Expected images of size {self.image_size}x{self.image_size}, "
                f"got {tuple(x.shape[-2:])}.")

        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerEncoderBlock(nn.Module):
    """Pre-norm transformer block with MLA and a feed-forward network."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size)
        self.attention = MultiHeadLatentAttention(config)
        self.norm2 = RMSNorm(config.hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.mlp_dim),
            nn.GELU(),
            nn.Linear(config.mlp_dim, config.hidden_size),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Small Vision Transformer classifier for MNIST."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()

        self.patch_embedding = PatchEmbedding(self.config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.config.hidden_size))
        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                self.config.sequence_length,
                self.config.hidden_size,
            )
        )

        self.transformer_layers = nn.ModuleList(
            [TransformerEncoderBlock(self.config) for _ in range(self.config.num_layers)])

        self.head = nn.Sequential(
            RMSNorm(self.config.hidden_size),
            nn.Linear(self.config.hidden_size, self.config.num_classes),)

        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Parameter):
            nn.init.trunc_normal_(module, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x)
        batch_size = x.shape[0]

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed

        for block in self.transformer_layers:
            x = block(x)

        return self.head(x[:, 0])

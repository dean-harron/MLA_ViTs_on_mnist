import torch

from src.mla_mnist.config import ModelConfig
from src.mla_mnist.models.vit import PatchEmbedding, VisionTransformer
from src.mla_mnist.utils import count_parameters


def test_patch_embedding_shape():
    config = ModelConfig()
    embedder = PatchEmbedding(config)
    images = torch.randn(4, 1, 28, 28)

    tokens = embedder(images)

    assert tokens.shape == (4, config.num_patches, config.hidden_size)


def test_vit_forward_shape():
    config = ModelConfig()
    model = VisionTransformer(config).eval()
    images = torch.randn(3, 1, 28, 28)

    logits = model(images)

    assert logits.shape == (3, config.num_classes)
    assert count_parameters(model) > 0


def test_model_has_cls_token_and_position_embedding():
    config = ModelConfig()
    model = VisionTransformer(config)

    assert model.cls_token.shape == (1, 1, config.hidden_size)
    assert model.pos_embed.shape == (1, config.sequence_length, config.hidden_size)

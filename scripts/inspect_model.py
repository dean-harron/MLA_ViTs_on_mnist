import torch

from mla_mnist.config import ModelConfig
from mla_mnist.models.vit import VisionTransformer
from mla_mnist.utils import count_parameters


config = ModelConfig()
model = VisionTransformer(config)

dummy_images = torch.randn(2, config.num_channels, config.image_size, config.image_size)
logits = model(dummy_images)

print("Model configuration:")
for key, value in config.to_dict().items():
    print(f"  {key}: {value}")

print(f"\nSequence length: {config.sequence_length}")
print(f"Trainable parameters: {count_parameters(model):,}")
print(f"Sample output shape: {tuple(logits.shape)}")

# Notebooks

The original implementation was a monolithic notebook. The GitHub-ready version moves the actual implementation into `src/mla_mnist/`.

A future notebook can import the package rather than redefining model classes and configuration in separate cells, for example:

```python
from mla_mnist.config import ModelConfig
from mla_mnist.models.vit import VisionTransformer
```

This keeps notebooks focused on experiments and visualization instead of duplicating source code.

# spatial-attention Documentation

Welcome to the spatial-attention documentation!

## Overview

Spatial grouping attention using Rotary Spatial Embeddings (RoSE)

## Table of Contents

- [Installation](installation.md)
- [Quick Start](quickstart.md)
- [API Reference](api.md)
- [Examples](examples.md)
- [Contributing](contributing.md)

## Getting Started

Inspired by the spatial grouping layer in Native Segmentation Vision Transformers (https://arxiv.org/abs/2505.16993), implemented in PyTorch with a modified rotary position embedding generalized to N-dimensions and incorporating real-world pixel spacing.

### Installation

```bash
pip install spatial-attention
```

### Quick Example

```python
import spatial_attention

# Your example code here
obj = spatial_attention.SpatialAttention(param1="example")
result = obj.process()
print(result)
```

## Support

If you encounter any issues or have questions, please:

1. Check the [documentation](https://github.com/rhoadesScholar/spatial-attention)
2. Search [existing issues](https://github.com/rhoadesScholar/spatial-attention/issues)
3. Create a [new issue](https://github.com/rhoadesScholar/spatial-attention/issues/new)

## License

This project is licensed under the BSD 3-Clause License License - see the [LICENSE](../LICENSE) file for details.

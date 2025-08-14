# spatial-grouping-attention Documentation

Welcome to the spatial-grouping-attention documentation!

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
pip install spatial-grouping-attention
```

### Quick Example

```python
import spatial_grouping_attention

# Create your spatial attention object
obj = spatial_grouping_attention.SpatialGroupingAttention(param1="example")
```

## Support

If you encounter any issues or have questions, please:

1. Check the [documentation](https://github.com/rhoadesScholar/spatial-grouping-attention)
2. Search [existing issues](https://github.com/rhoadesScholar/spatial-grouping-attention/issues)
3. Create a [new issue](https://github.com/rhoadesScholar/spatial-grouping-attention/issues/new)

## License

This project is licensed under the BSD 3-Clause License License - see the [LICENSE](../LICENSE) file for details.

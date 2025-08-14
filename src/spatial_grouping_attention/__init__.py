"""spatial_grouping_attention: Spatial grouping attention
using Rotary Spatial Embeddings (RoSE)

Inspired by the spatial grouping layer in Native Segmentation
Vision Transformers (https://arxiv.org/abs/2505.16993),
implemented in PyTorch with a modified rotary position embedding
generalized to N-dimensions and incorporating real-world pixel spacing.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("spatial-grouping-attention")
except PackageNotFoundError:
    __version__ = "0.1.0"

__author__ = "Jeff Rhoades"
__email__ = "rhoadesj@hhmi.org"

from . import utils
from .mlp import MLP
from .spatial_grouping_attention import (
    DenseSpatialGroupingAttention,
    SparseSpatialGroupingAttention,
    SpatialGroupingAttention,
)

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "SpatialGroupingAttention",
    "SparseSpatialGroupingAttention",
    "DenseSpatialGroupingAttention",
    "MLP",
    "utils",
]

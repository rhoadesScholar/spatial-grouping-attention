# spatial-grouping-attention

## PyTorch Implementation of Spatial Grouping Attention

![GitHub - License](https://img.shields.io/github/license/rhoadesScholar/spatial-grouping-attention)
[![CI/CD Pipeline](https://github.com/rhoadesScholar/spatial-grouping-attention/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/rhoadesScholar/spatial-grouping-attention/actions/workflows/ci-cd.yml)
[![codecov](https://codecov.io/github/rhoadesScholar/spatial-grouping-attention/graph/badge.svg)](https://codecov.io/github/rhoadesScholar/spatial-grouping-attention)
![PyPI - Version](https://img.shields.io/pypi/v/spatial-grouping-attention)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/spatial-grouping-attention)

Inspired by the spatial grouping layer in Native Segmentation Vision Transformers (https://arxiv.org/abs/2505.16993), implemented in PyTorch with a modified rotary position embedding generalized to N-dimensions and incorporating real-world pixel spacing.

## Installation

### From PyPI

You will first need to install PyTorch separately, as it is required for building one of our dependencies (natten). We recommend installing within a virtual environment, such as `venv` or `mamba`:

```bash
# create a virtual environment
mamba create -n spatial-attention -y python=3.11 pytorch ninja cmake

# activate the virtual environment
mamba activate spatial-attention

# install the package(s)
pip install spatial-grouping-attention
pip install natten==0.17.5 # requires python 3.11
```

### From source

To install the latest development version directly from GitHub, follow the creation and activation of a virtual environment, as above, then run:

```bash
pip install git+https://github.com/rhoadesScholar/spatial-grouping-attention.git
```

## Usage

The spatial grouping attention mechanism automatically computes query grid parameters (`q_spacing` and `q_grid_shape`) from the input key grid and convolution parameters. This makes it easy to use - you only need to specify the input resolution and the algorithm handles the spatial downsampling.

### Basic 2D Dense Attention

```python
import torch
from spatial_grouping_attention import DenseSpatialGroupingAttention

# Create attention module with 7x7 grouping kernel, stride=1 (no downsampling)
attention = DenseSpatialGroupingAttention(
    feature_dims=128,
    spatial_dims=2,
    kernel_size=7,        # 7x7 spatial grouping
    stride=1,            # No spatial downsampling
    num_heads=8,
    mlp_ratio=4
)

# Input: 32x32 image with 128 features per pixel
batch_size, height, width = 2, 32, 32
x = torch.randn(batch_size, height * width, 128)

# Only specify input (key) grid - query grid computed automatically
input_spacing = (0.5, 0.5)      # 0.5 microns per pixel
input_grid_shape = (32, 32)     # Input resolution

output = attention(x=x, input_spacing=input_spacing, input_grid_shape=input_grid_shape)
# Auto-computed: q_spacing = (0.5, 0.5) * 1 = (0.5, 0.5)
# Auto-computed: q_grid_shape = (32, 32) (no downsampling with stride=1)
print(f"Output shape: {output['x_out'].shape}")  # (2, 1024, 128)
```

### 2D Attention with Spatial Downsampling

```python
# Create attention with spatial downsampling for efficiency
downsampling_attention = DenseSpatialGroupingAttention(
    feature_dims=256,
    spatial_dims=2,
    kernel_size=5,        # 5x5 grouping kernel
    stride=2,            # 2x spatial downsampling
    padding=2,           # Maintain spatial coverage
    num_heads=16
)

# High resolution input: 128x128 image
x_hires = torch.randn(1, 128*128, 256)
input_spacing_hires = (0.1, 0.1)        # 0.1 mm per pixel (input)
input_grid_shape_hires = (128, 128)     # High-res input grid

output_hires = downsampling_attention(
    x=x_hires,
    input_spacing=input_spacing_hires,
    input_grid_shape=input_grid_shape_hires
)
# Auto-computed: q_spacing = (0.1, 0.1) * 2 = (0.2, 0.2)
# Auto-computed: q_grid_shape = (128+2*2-5)//2+1 = (64, 64)
print(f"Downsampled output: {output_hires['x_out'].shape}")  # (1, 4096, 256)
print(f"Compression ratio: {128*128 / (64*64)}x")        # 4x fewer points
```

### 3D Sparse Attention (GPU Required)

```python
# 3D sparse attention for volumetric data (requires CUDA + natten)
try:
    from spatial_grouping_attention import SparseSpatialGroupingAttention

    sparse_3d = SparseSpatialGroupingAttention(
        feature_dims=128,
        spatial_dims=3,
        kernel_size=(3, 5, 5),    # Anisotropic: 3x5x5 grouping
        stride=(1, 2, 2),         # Downsample only in x,y
        num_heads=8,
        neighborhood_kernel=9     # Local attention window
    )

    # 3D volume: 16x64x64 voxels
    depth, height, width = 16, 64, 64
    x_3d = torch.randn(1, depth*height*width, 128).cuda()

    # Anisotropic spacing (e.g., confocal microscopy)
    input_spacing_3d = (0.5, 0.1, 0.1)     # z, y, x spacing in microns
    input_grid_shape_3d = (16, 64, 64)

    output_3d = sparse_3d(
        x=x_3d,
        input_spacing=input_spacing_3d,
        input_grid_shape=input_grid_shape_3d
    )
    # Auto-computed: q_spacing = (0.5*1, 0.1*2, 0.1*2) = (0.5, 0.2, 0.2)
    # Auto-computed: q_grid_shape = (16, 32, 32) - downsampled in x,y only
    print(f"3D sparse output: {output_3d['x_out'].shape}")  # (1, 16384, 128)

except ImportError:
    print("SparseSpatialGroupingAttention requires CUDA and natten package")
```

### Multi-Scale Processing

```python
# Process same input at multiple scales efficiently
multiscale_attention = DenseSpatialGroupingAttention(
    feature_dims=64,
    spatial_dims=2,
    kernel_size=9,
    stride=4,            # 4x downsampling for global context
    num_heads=4,
    iters=3              # Multiple attention iterations
)

# Input image
x_input = torch.randn(1, 64*64, 64)
input_spacing = (1.0, 1.0)              # 1 micron per pixel
input_grid_shape = (64, 64)

# Global context via 4x downsampling
global_output = multiscale_attention(
    x=x_input,
    input_spacing=input_spacing,
    input_grid_shape=input_grid_shape
)
# Auto-computed: q_spacing = (4.0, 4.0), q_grid_shape = (16, 16)
print(f"Global context: {global_output['x_out'].shape}")  # (1, 256, 64)

# Fine-scale processing with stride=1
fine_attention = DenseSpatialGroupingAttention(
    feature_dims=64,
    spatial_dims=2,
    kernel_size=5,
    stride=1,            # Full resolution
    num_heads=4
)

fine_output = fine_attention(
    x=x_input,
    input_spacing=input_spacing,
    input_grid_shape=input_grid_shape
)
# Auto-computed: q_spacing = (1.0, 1.0), q_grid_shape = (64, 64)
print(f"Fine details: {fine_output['x_out'].shape}")      # (1, 4096, 64)
```

### Integration with Neural Networks

```python
class HierarchicalSpatialNet(torch.nn.Module):
    """Multi-scale spatial processing network"""

    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()

        # Input embedding
        self.embed = torch.nn.Linear(input_channels, 128)

        # Coarse-scale attention (4x downsampling)
        self.coarse_attention = DenseSpatialGroupingAttention(
            feature_dims=128,
            spatial_dims=2,
            kernel_size=7,
            stride=4,         # 4x spatial compression
            num_heads=8
        )

        # Fine-scale attention (2x downsampling)
        self.fine_attention = DenseSpatialGroupingAttention(
            feature_dims=128,
            spatial_dims=2,
            kernel_size=5,
            stride=2,         # 2x spatial compression
            num_heads=8
        )

        # Cross-scale fusion
        self.fusion = torch.nn.Linear(256, 128)
        self.classifier = torch.nn.Linear(128, num_classes)

    def forward(self, images, pixel_spacing=(1.0, 1.0)):
        B, C, H, W = images.shape

        # Flatten and embed
        x = images.permute(0, 2, 3, 1).reshape(B, H*W, C)
        x = self.embed(x)

        # Multi-scale attention
        coarse_out = self.coarse_attention(
            x=x,
            input_spacing=pixel_spacing,
            input_grid_shape=(H, W)
        )['x_out']  # (B, H*W/16, 128) - 4x downsampling

        fine_out = self.fine_attention(
            x=x,
            input_spacing=pixel_spacing,
            input_grid_shape=(H, W)
        )['x_out']   # (B, H*W/4, 128) - 2x downsampling

        # Upsample coarse to match fine resolution for fusion
        coarse_upsampled = torch.nn.functional.interpolate(
            coarse_out.transpose(1, 2).reshape(B, 128, H//4, W//4),
            size=(H//2, W//2),
            mode='bilinear',
            align_corners=False
        ).reshape(B, 128, -1).transpose(1, 2)

        # Fuse multi-scale features
        fused = self.fusion(torch.cat([fine_out, coarse_upsampled], dim=-1))

        # Global pooling and classification
        global_features = fused.mean(dim=1)
        return self.classifier(global_features)

# Usage example
net = HierarchicalSpatialNet(input_channels=3, num_classes=1000)
sample_images = torch.randn(4, 3, 128, 128)  # ImageNet-style input
pixel_spacing = (0.1, 0.1)  # 0.1 mm per pixel

predictions = net(sample_images, pixel_spacing)
print(f"Predictions: {predictions.shape}")  # (4, 1000)
```

### Key Principles

1. **Automatic Grid Calculation**: You only specify input (`input_spacing`, `input_grid_shape`) - the query grid is computed as:
   - `q_spacing = input_spacing * stride`
   - `q_grid_shape = (k_grid + 2*padding - kernel) // stride + 1`

2. **Spatial Grouping**: The `kernel_size` determines how many neighboring points are grouped together for attention computation.

3. **Multi-Scale Processing**: Use different `stride` values to process the same input at multiple spatial scales efficiently.

4. **Memory Efficiency**: Larger strides reduce the number of query points, making attention computation more efficient for large inputs.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the test suite (`make test`)
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.

## Citation

If you use this software in your research, please cite it using the information in [CITATION.cff](CITATION.cff).

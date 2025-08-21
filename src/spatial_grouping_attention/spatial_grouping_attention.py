from abc import abstractmethod
import math
from typing import Any, Optional, Sequence, Tuple, Type

from RoSE import RotarySpatialEmbedding
import torch

from .mlp import MLP
from .utils import to_tuple


class SpatialGroupingAttention(torch.nn.Module):
    """Base class for spatial grouping attention. Modify `attn` method in subclasses.

    Args:
        spatial_dims: Number of spatial dimensions (2 or 3)
        feature_dims: Number of input feature dimensions
        kernel_size: Size of the convolutional kernel for strided
                     convolutions producing the lower resolution group embeddings
        num_heads: Number of attention heads
        stride: Stride for the strided convolution (default is half kernel size)
        iters: Number of iterations for the attention mechanism (default is 3)
        mlp_ratio: Ratio of hidden to input/output dimensions in the MLP (default is 4)
        mlp_dropout: Dropout rate for the MLP (default is 0.0)
        mlp_bias: Whether to use bias in the MLP (default is True)
        mlp_activation: Activation function for the MLP (default is GELU)
        qkv_bias: Whether to use bias in the query/key/value linear layers
                  (default is True)
        base_theta: Base theta value for the rotary position embedding
                    (default is 1e4)
        learnable_rose: Whether to use learnable rotary spatial embeddings
                        (default is True)
        init_jitter_std: Standard deviation for initial jitter in the rotary
                         embeddings (default is 0.02)
        rotary_ratio: Fraction of the feature dimension to rotate
        frequency_scaling: Frequency scaling method for the rotary position embedding
                        (default is "sqrt")
        spacing: Default real-world pixel spacing for the input data
                 (default is None, which uses a default spacing of 1.0 for
                 all dimensions). Can be specified at initialization or passed
                 during the forward pass.
    """

    def __init__(
        self,
        feature_dims: int = 128,
        spatial_dims: int = 3,
        kernel_size: int | Sequence[int] = 7,
        num_heads: int = 16,
        stride: Optional[int | Sequence[int]] = None,
        iters: int = 3,
        mlp_ratio: float | int = 4,
        mlp_dropout: float = 0.0,
        mlp_bias: bool = True,
        mlp_activation: Type[torch.nn.Module] = torch.nn.GELU,
        qkv_bias: bool = True,
        base_theta: float = 1e4,
        learnable_rose: bool = True,
        init_jitter_std: float = 0.02,
        rotary_ratio: float = 0.5,
        frequency_scaling: str = "sqrt",
        spacing: Optional[float | Sequence[float]] = None,
    ) -> None:
        super().__init__()
        conv = {
            1: torch.nn.Conv1d,
            2: torch.nn.Conv2d,
            3: torch.nn.Conv3d,
        }[spatial_dims]
        self.feature_dims = feature_dims
        self.spatial_dims = spatial_dims
        self.rotary_ratio = rotary_ratio
        self.frequency_scaling = frequency_scaling
        if spacing is None:
            spacing = 1.0
        self._default_spacing = to_tuple(spacing, spatial_dims)
        self.kernel_size = to_tuple(
            kernel_size, spatial_dims, dtype_caster=int, allow_nested=False
        )
        # Handle stride conversion (optional parameter)
        if stride is None:
            # Default stride is half kernel size, minimum 1
            self.stride = tuple(
                max(1, k // 2) for k in self.kernel_size  # type: ignore
            )
        else:
            self.stride = to_tuple(
                stride, spatial_dims, dtype_caster=int, allow_nested=False
            )

        # Calculate proper padding when stride is not 1
        if all(s == 1 for s in self.stride):
            padding = "same"
        else:
            # Calculate manual padding for strided convolutions
            padding = tuple((k - 1) // 2 for k in self.kernel_size)  # type: ignore
        self.strider = conv(
            in_channels=feature_dims,
            out_channels=feature_dims,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=padding,
        )
        self.iters = iters
        self.qkv_bias = qkv_bias
        self.kv = torch.nn.Linear(
            in_features=feature_dims, out_features=feature_dims * 2, bias=qkv_bias
        )
        self.q = torch.nn.Linear(
            in_features=feature_dims, out_features=feature_dims, bias=qkv_bias
        )
        self.mlp_ratio = mlp_ratio
        self.mlp_activation = mlp_activation
        self.mlp_dropout = mlp_dropout
        self.mlp_bias = mlp_bias
        self.mlp = MLP(
            in_features=feature_dims,
            hidden_features=int(feature_dims * mlp_ratio),
            out_features=feature_dims,
            act_layer=mlp_activation,
            drop=mlp_dropout,
            bias=mlp_bias,
        )
        self.temp = torch.nn.Parameter(
            torch.tensor([1 / math.sqrt(feature_dims)]), requires_grad=True
        )
        self.num_heads = num_heads
        self.base_theta = base_theta
        self.learnable_rose = learnable_rose
        self.rose = RotarySpatialEmbedding(
            dim=feature_dims,
            num_heads=self.num_heads,
            spatial_dims=spatial_dims,
            base_theta=base_theta,
            learnable=learnable_rose,
            init_jitter_std=init_jitter_std,
            rotary_ratio=rotary_ratio,
            frequency_scaling=frequency_scaling,
        )
        self.norm1 = torch.nn.LayerNorm(feature_dims)
        self.norm2 = torch.nn.LayerNorm(feature_dims)
        self.norm3 = torch.nn.LayerNorm(feature_dims)
        self.mask_embedding = torch.nn.Parameter(
            torch.rand(feature_dims), requires_grad=True
        )

    def _calculate_strided_grid_shape(
        self, input_grid_shape: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        """Calculate output grid shape after strided convolution.

        Args:
            input_grid_shape: Shape of input grid

        Returns:
            Shape of output grid after applying strided convolution
        """
        # Cast to tuples of ints (they are after to_tuple with dtype_caster=int)
        stride = tuple(int(s) for s in self.stride)  # type: ignore
        kernel_size = tuple(int(k) for k in self.kernel_size)  # type: ignore

        if all(s == 1 for s in stride):
            # No striding, same shape
            return input_grid_shape

        output_shape = []
        for input_size, s, k in zip(input_grid_shape, stride, kernel_size):
            # Standard formula for conv output size
            padding = (k - 1) // 2  # Same as we used in __init__
            output_size = (input_size + 2 * padding - k) // s + 1
            output_shape.append(output_size)

        return tuple(output_shape)

    def _calculate_strided_spacing(
        self, input_spacing: Tuple[float, ...]
    ) -> Tuple[float, ...]:
        """Calculate output spacing after strided convolution.

        Args:
            input_spacing: Spacing of input grid

        Returns:
            Spacing of output grid after applying strided convolution
        """
        stride = tuple(int(s) for s in self.stride)  # type: ignore
        # When we stride, the effective spacing increases by the stride factor
        return tuple(spacing * s for spacing, s in zip(input_spacing, stride))

    @abstractmethod
    def attn(
        self,
        k: torch.Tensor,  # (B, H, N_k, dims_per_head),
        q: torch.Tensor,  # (B, H, N_q, dims_per_head),
        q_grid_shape: int | Tuple[int, ...],
        input_grid_shape: int | Tuple[int, ...],
    ) -> torch.Tensor:
        """Compute attention scores.

        Args:
            q: Query tensor of shape (B, H, N_q, dims_per_head)
            k: Key tensor of shape (B, H, N_k, dims_per_head)
            q_grid_shape: Grid shape for query tensor
            input_grid_shape: Grid shape for key tensor

        Returns:
            Tensor of attention scores of shape (B, H, N_q, N_k)
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def forward(
        self,
        x: torch.Tensor,
        input_spacing: Tuple[float, ...],
        input_grid_shape: Tuple[int, ...],
        mask: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """Forward pass with automatic query grid calculation.

        Args:
            x: Input tensor of shape (B, N_in, D)
            input_spacing: Spacing for key grid (input)
            input_grid_shape: Shape of key grid (input)
            mask: Optional mask tensor

        Returns:
            Dictionary with output tensor and optional debug info
        """
        # Auto-calculate query parameters if not provided
        q_spacing = self._calculate_strided_spacing(input_spacing)
        q_grid_shape = self._calculate_strided_grid_shape(input_grid_shape)

        B, N_in, D = x.shape
        if mask is not None:
            # Ensure mask is boolean and on the same device
            mask = mask.to(x.device) > 0  # [B, *spatial_dims]

            # Apply boolean mask
            x = torch.where(
                mask.flatten(1).unsqueeze(-1),
                self.mask_embedding.expand(B, N_in, D),
                x,
            )

        x_out = self.strider(
            x.transpose(1, 2).reshape(B, D, *input_grid_shape)
        )  # (B, D, *input_grid_shape)
        x_out = self.norm1(x_out.flatten(2).transpose(1, 2))  # (B, N_out, D)
        N_out = x_out.shape[1]
        attn_k = torch.empty(
            (B, self.num_heads, N_out, N_in),
            device=x_out.device,
        )
        attn_q = attn_k.clone()
        for _ in range(self.iters):
            k, v = self.kv(x).chunk(2, dim=-1)  # (B, N_in, D), (B, N_in, D)

            # --> (B, H, N, dims_per_head)
            k = self.temp * self.rose(k, input_spacing, input_grid_shape, flatten=False)
            q = self.rose(self.q(x_out), q_spacing, q_grid_shape, flatten=False)

            # --> (B, H, N_out, N_in)
            attn_k = self.attn(k, q, q_grid_shape, input_grid_shape)
            attn_q = attn_k / (
                attn_k.sum(dim=-2, keepdim=True) + torch.finfo(k.dtype).eps
            )

            v = v.reshape(*k.shape)  # (B, H, N_in, dims_per_head)
            # --> (B, N_out, D)N_out, N_in)}"
            x_out = x_out + self.norm2((attn_q @ v).reshape(B, N_out, D))
            x_out = x_out + self.norm3(self.mlp(x_out))  # (B, N_out, D)

        return {
            "x_out": x_out,
            "attn_q": attn_q,
            "attn_k": attn_k,
            "out_grid_shape": q_grid_shape,
            "out_spacing": q_spacing,
        }

    def __repr__(self) -> str:
        """Return string representation of the object."""
        return f"SpatialGroupingAttention{self.spatial_dims}D"


class SparseSpatialGroupingAttention(SpatialGroupingAttention):
    """Sparse version of SpatialGroupingAttention using neighborhood attention."""

    def __init__(
        self,
        *args,
        neighborhood_kernel: int | Sequence[int] = 3,
        neighborhood_dilation: int | Sequence[int] = 1,
        is_causal: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        try:
            from natten.functional import (
                na1d_av,
                na1d_qk,
                na2d_av,
                na2d_qk,
                na3d_av,
                na3d_qk,
            )
        except ImportError:
            raise ImportError(
                "NATTEN is required for SparseSpatialGroupingAttention. "
                "Please install it with `pip install natten==0.17.5`"
                "Note that NATTEN requires CUDA support."
            )

        self.neighborhood_kernel = to_tuple(
            neighborhood_kernel, self.spatial_dims, dtype_caster=int, allow_nested=False
        )
        self.neighborhood_dilation = to_tuple(
            neighborhood_dilation,
            self.spatial_dims,
            dtype_caster=int,
            allow_nested=False,
        )
        self.is_causal = is_causal

        self._qk_attn = {1: na1d_qk, 2: na2d_qk, 3: na3d_qk}[self.spatial_dims]
        self._av_attn = {1: na1d_av, 2: na2d_av, 3: na3d_av}[self.spatial_dims]

    def attn(
        self,
        k: torch.Tensor,  # (B, N_k, D)
        q: torch.Tensor,  # (B, N_q, D)
        q_grid_shape: int | Tuple[int, ...],
        input_grid_shape: int | Tuple[int, ...],
    ) -> torch.Tensor:
        """Compute sparse neighborhood attention scores."""

        B, H, _, dims_per_head = k.shape

        k = k.view(B, H, *input_grid_shape, dims_per_head)  # type: ignore
        q = q.view(B, H, *q_grid_shape, dims_per_head)  # type: ignore

        # Compute attention scores using neighborhood attention
        attn = self._qk_attn(
            k,
            q,
            self.neighborhood_kernel,
            self.neighborhood_dilation,
            is_causal=self.is_causal,
        )  # (B, H, N_q, N_k)
        attn = torch.softmax(attn, dim=-1)  # likelihood the query includes the key

        return attn


class DenseSpatialGroupingAttention(SpatialGroupingAttention):
    """Dense version of SpatialGroupingAttention using full attention."""

    def attn(
        self,
        k: torch.Tensor,  # (B, H, N_k, D)
        q: torch.Tensor,  # (B, H, N_q, D)
        q_grid_shape: int | Tuple[int, ...],
        input_grid_shape: int | Tuple[int, ...],
    ) -> torch.Tensor:
        """Compute dense full attention scores."""
        # TODO: Make fused triton softmax(k @ q.T, dim=-1)
        # q: (B, H, N_q, D), k: (B, H, N_k, D) -> attn: (B, H, N_q, N_k)
        attn = torch.einsum("bhqd,bhkd->bhqk", q, k)  # (B, H, N_q, N_k)
        attn = torch.softmax(attn, dim=-1)  # likelihood the query includes the key
        return attn


class DeformableSpatialGroupingAttention(SpatialGroupingAttention):
    """Deformable version of SpatialGroupingAttention."""

    # TODO: Implement deformable attention mechanism
    def __init__(
        self,
        *args,
        deformable_groups: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.deformable_groups = deformable_groups

    def attn(
        self,
        k: torch.Tensor,  # (B, H, N_k, D)
        q: torch.Tensor,  # (B, H, N_q, D)
        q_grid_shape: int | Tuple[int, ...],
        input_grid_shape: int | Tuple[int, ...],
    ) -> torch.Tensor:
        """Compute deformable attention scores."""
        # TODO: Implement deformable attention mechanism
        ...

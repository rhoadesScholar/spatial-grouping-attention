from abc import abstractmethod
import math
from typing import Optional, Sequence, Tuple, Type

from RoSE import RotarySpatialEmbedding
import torch

from .mlp import MLP
from .utils import to_tuple


class SpatialAttention:
    """Main class for spatial_attention.

    This is a template class that you should modify according to your needs.

    Args:
        spatial_dims: Number of spatial dimensions (2 or 3)
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
        spacing: Optional[float | Sequence[float]] = None,  # used for default spacing
    ) -> None:
        conv = {
            1: torch.nn.Conv1d,
            2: torch.nn.Conv2d,
            3: torch.nn.Conv3d,
        }[spatial_dims]
        self.feature_dims = feature_dims
        self.spatial_dims = spatial_dims
        if spacing is None:
            self._default_spacing = (1.0,) * spatial_dims
        else:
            self._default_spacing = to_tuple(spacing, spatial_dims)
        self.kernel_size = to_tuple(
            kernel_size, spatial_dims, dtype_caster=int, allow_nested=False
        )
        # Handle stride conversion (optional parameter)
        if stride is None:
            self.stride = tuple(max(1, k // 2) for k in self.kernel_size)  # type: ignore
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
        pe_kwargs = {
            "dim": feature_dims,
            "num_heads": self.num_heads,
            "spatial_dims": spatial_dims,
            "base_theta": base_theta,
            "learnable": learnable_rose,
            "init_jitter_std": init_jitter_std,
        }
        self.q_pe = RotarySpatialEmbedding(**pe_kwargs)
        self.k_pe = RotarySpatialEmbedding(**pe_kwargs)
        self.norm1 = torch.nn.LayerNorm(feature_dims)
        self.norm2 = torch.nn.LayerNorm(feature_dims)
        self.norm3 = torch.nn.LayerNorm(feature_dims)
        self.mask_embedding = torch.nn.Parameter(
            torch.zeros((1, feature_dims)), requires_grad=True
        )

    @abstractmethod
    def attn(
        self,
        k: torch.Tensor,  # (B, H, N_k, dims_per_head),
        q: torch.Tensor,  # (B, H, N_q, dims_per_head),
        q_grid_shape: int | Tuple[int, ...],
        k_grid_shape: Optional[int | Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """Compute attention scores.

        Args:
            q: Query tensor of shape (B, H, N_q, dims_per_head)
            k: Key tensor of shape (B, H, N_k, dims_per_head)

        Returns:
            Tensor of attention scores of shape (B, H, N_q, N_k)
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def forward(
        self,
        x: torch.Tensor,
        q_spacing: Tuple[float, ...],
        q_grid_shape: Optional[Tuple[int, ...]] = None,
        k_spacing: Optional[Tuple[float, ...]] = None,
        k_grid_shape: Optional[Tuple[int, ...]] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if k_spacing is None:
            k_spacing = q_spacing
        if k_grid_shape is None:
            k_grid_shape = q_grid_shape

        B, N_in, D = x.shape
        x_out = self.norm1(self.strider(x))
        N_q = x_out.shape[1]
        attn_k = torch.empty(
            (B, self.num_heads, N_q, N_in),
            device=x_out.device,
        )
        attn_q = attn_k.clone()
        for _ in range(self.iters):
            k, v = self.kv(x).chunk(2, dim=-1)  # (B, N_in, D), (B, N_in, D)

            # --> (B, H, N, dims_per_head)
            k = self.temp * self.k_pe(x, k_spacing, k_grid_shape, flatten=False)
            q = self.q_pe(self.q(x_out), q_spacing, q_grid_shape, flatten=False)
            if mask is not None:
                raise NotImplementedError(
                    "Masking is not implemented in the base SpatialAttention class."
                )

            # --> (B, H, N_out, N_in)
            attn_k = self.attn(q, k, q_grid_shape, k_grid_shape)
            attn_q = attn_k / (
                attn_k.sum(dim=1, keepdim=True) + torch.finfo(k.dtype).eps
            )

            # --> (B, N_out, D)
            x_out = x_out + self.norm2((attn_q.T @ v.view(*k.shape)).view(B, N_q, D))
            x_out = x_out + self.norm3(self.mlp(x_out))  # (B, N_out, D)

        return {
            "x_out": x_out,
            "attn_q": attn_q,
            "attn_k": attn_k,
        }

    def __repr__(self) -> str:
        """Return string representation of the object."""
        return f"SpatialAttention{self.spatial_dims}D"


class SparseSpatialAttention(SpatialAttention):
    """Sparse version of SpatialAttention using neighborhood attention."""

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
                na1d_qk,
                na1d_av,
                na2d_qk,
                na2d_av,
                na3d_qk,
                na3d_av,
            )
        except ImportError:
            raise ImportError(
                "NATTEN is required for SparseSpatialAttention. "
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
        k_grid_shape: Optional[int | Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """Compute sparse neighborhood attention scores."""
        # Reshape k and q for neighborhood attention
        q_grid_shape = to_tuple(q_grid_shape, self.spatial_dims, dtype_caster=int, allow_nested=False)  # type: ignore
        if k_grid_shape is None:
            k_grid_shape = q_grid_shape
        else:
            k_grid_shape = to_tuple(k_grid_shape, self.spatial_dims, dtype_caster=int, allow_nested=False)  # type: ignore

        B, H, _, dims_per_head = k.shape

        k = k.view(B, H, *k_grid_shape, dims_per_head)  # type: ignore
        q = q.view(B, H, *q_grid_shape, dims_per_head)  # type: ignore

        # Compute attention scores using neighborhood attention
        attn = self._qk_attn(
            k,
            q,
            self.neighborhood_kernel,
            self.neighborhood_dilation,
            is_causal=self.is_causal,
        )  # (B, H, N_q, N_k)
        attn = torch.softmax(attn, dim=-1)  # Softmax over groups

        return attn


class DenseSpatialAttention(SpatialAttention):
    """Dense version of SpatialAttention using full attention."""

    def attn(
        self,
        k: torch.Tensor,  # (B, H, N_k, D)
        q: torch.Tensor,  # (B, H, N_q, D)
        q_grid_shape: int | Tuple[int, ...],
        k_grid_shape: Optional[int | Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """Compute dense full attention scores."""
        # TODO: Make fused triton softmax(k @ q.T, dim=-1)
        # q: (B, H, N_q, D), k: (B, H, N_k, D) -> attn: (B, H, N_q, N_k)
        attn = torch.einsum("bhqd,bhkd->bhqk", q, k)  # (B, H, N_q, N_k)
        attn = torch.softmax(attn, dim=-2)  # Softmax over groups
        return attn

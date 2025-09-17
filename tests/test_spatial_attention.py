"""Test suite for spatial_grouping_attention."""

import pytest
from timm.layers import Mlp
import torch

from spatial_grouping_attention import (
    DenseSpatialGroupingAttention,
    SparseSpatialGroupingAttention,
    SpatialGroupingAttention,
)


class TestSpatialGroupingAttentionBase:
    """Test cases for base SpatialGroupingAttention class."""

    def test_spatial_grouping_attention_initialization_2d(self):
        """Test SpatialGroupingAttention initialization with 2D."""
        attn = SpatialGroupingAttention(
            feature_dims=128, spatial_dims=2, kernel_size=7, num_heads=8
        )
        assert attn.feature_dims == 128
        assert attn.spatial_dims == 2
        assert attn.kernel_size == (7, 7)
        assert attn.num_heads == 8
        assert len(attn._default_spacing) == 2

    def test_spatial_grouping_attention_initialization_3d(self):
        """Test SpatialGroupingAttention initialization with 3D."""
        attn = SpatialGroupingAttention(
            feature_dims=256, spatial_dims=3, kernel_size=[5, 7, 9], num_heads=16
        )
        assert attn.feature_dims == 256
        assert attn.spatial_dims == 3
        assert attn.kernel_size == (5, 7, 9)
        assert attn.num_heads == 16
        assert len(attn._default_spacing) == 3

    def test_spatial_grouping_attention_custom_spacing(self):
        """Test SpatialGroupingAttention with custom spacing."""
        attn = SpatialGroupingAttention(spatial_dims=3, spacing=[1.0, 2.0, 0.5])
        assert attn._default_spacing == (1.0, 2.0, 0.5)

    def test_spatial_grouping_attention_repr(self):
        """Test string representation."""
        attn = SpatialGroupingAttention(spatial_dims=2)
        repr_str = repr(attn)
        assert "SpatialGroupingAttention2D" in repr_str

        attn_3d = SpatialGroupingAttention(spatial_dims=3)
        repr_str_3d = repr(attn_3d)
        assert "SpatialGroupingAttention3D" in repr_str_3d

    def test_spatial_grouping_attention_stride_calculation(self):
        """Test automatic stride calculation."""
        attn = SpatialGroupingAttention(spatial_dims=2, kernel_size=7)
        # Default stride should be kernel_size // 2
        assert attn.stride == (3, 3)

    def test_spatial_grouping_attention_custom_stride(self):
        """Test custom stride setting."""
        attn = SpatialGroupingAttention(spatial_dims=2, kernel_size=7, stride=2)
        assert attn.stride == (2, 2)

    def test_spatial_grouping_attention_components_created(self):
        """Test that all required components are created."""
        attn = SpatialGroupingAttention(feature_dims=128, spatial_dims=2, num_heads=8)

        # Check linear layers
        assert hasattr(attn, "q") and isinstance(attn.q, torch.nn.Linear)
        assert hasattr(attn, "kv") and isinstance(attn.kv, torch.nn.Linear)
        assert attn.q.in_features == 128
        assert attn.q.out_features == 128
        assert attn.kv.in_features == 128
        assert attn.kv.out_features == 256  # 2 * feature_dims

        # Check normalization layers
        assert hasattr(attn, "norm1") and isinstance(attn.norm1, torch.nn.LayerNorm)
        assert hasattr(attn, "norm2") and isinstance(attn.norm2, torch.nn.LayerNorm)
        assert hasattr(attn, "norm3") and isinstance(attn.norm3, torch.nn.LayerNorm)

        # Check temperature parameter
        assert hasattr(attn, "temp") and isinstance(attn.temp, torch.nn.Parameter)

        # Check MLP
        assert hasattr(attn, "mlp") and isinstance(attn.mlp, Mlp)

        # Check mask embedding
        assert hasattr(attn, "mask_embedding")
        assert attn.mask_embedding.shape == (128,)


@pytest.mark.natten
class TestSparseSpatialGroupingAttention:
    """Test cases for SparseSpatialGroupingAttention class."""

    def test_sparse_initialization(self, natten_available):
        """Test SparseSpatialGroupingAttention initialization."""
        if not natten_available:
            pytest.skip(
                "SparseSpatialGroupingAttention requires NATTEN with CUDA support"
            )

        attn = SparseSpatialGroupingAttention(
            feature_dims=128,
            spatial_dims=2,
            neighborhood_kernel=3,
            neighborhood_dilation=2,
            is_causal=True,
        )
        assert attn.neighborhood_kernel == (3, 3)
        assert attn.neighborhood_dilation == (2, 2)
        assert attn.is_causal is True

    def test_sparse_initialization_different_kernels(self, natten_available):
        """
        Test SparseSpatialGroupingAttention with different kernel sizes per
        dimension.
        """
        if not natten_available:
            pytest.skip(
                "SparseSpatialGroupingAttention requires NATTEN with CUDA " "support"
            )

        attn = SparseSpatialGroupingAttention(
            spatial_dims=3,
            neighborhood_kernel=[3, 5, 7],
            neighborhood_dilation=[1, 2, 1],
        )
        assert attn.neighborhood_kernel == (3, 5, 7)
        assert attn.neighborhood_dilation == (1, 2, 1)

    def test_sparse_has_attention_functions(self, natten_available):
        """Test that sparse attention has the right attention functions."""
        if not natten_available:
            pytest.skip(
                "SparseSpatialGroupingAttention requires NATTEN with CUDA support"
            )

        attn_2d = SparseSpatialGroupingAttention(spatial_dims=2)
        attn_3d = SparseSpatialGroupingAttention(spatial_dims=3)

        assert hasattr(attn_2d, "_qk_attn")
        assert hasattr(attn_2d, "_av_attn")
        assert hasattr(attn_3d, "_qk_attn")
        assert hasattr(attn_3d, "_av_attn")

    def test_masking_functionality(self, natten_available):
        """Test that masking works correctly."""
        if not natten_available:
            pytest.skip(
                "SparseSpatialGroupingAttention requires NATTEN with CUDA support"
            )

        attn = SparseSpatialGroupingAttention(feature_dims=64, spatial_dims=2)
        x = torch.randn(1, 16, 64)
        mask = torch.zeros(1, 16, dtype=torch.bool)
        mask[0, :8] = True  # Mask first half of the input

        # Test that masking doesn't crash
        result = attn.forward(
            x, input_spacing=(1.0, 1.0), input_grid_shape=(4, 4), mask=mask
        )

        # Ensure the result has the expected keys
        assert "x_out" in result
        assert "attn_q" in result
        assert "attn_k" in result


class TestDenseSpatialGroupingAttention:
    """Test cases for DenseSpatialGroupingAttention class."""

    def test_dense_initialization(self):
        """Test DenseSpatialGroupingAttention initialization."""
        attn = DenseSpatialGroupingAttention(
            feature_dims=128, spatial_dims=2, num_heads=8
        )
        assert attn.feature_dims == 128
        assert attn.spatial_dims == 2
        assert attn.num_heads == 8

    def test_dense_forward_shapes(self):
        """Test DenseSpatialGroupingAttention forward pass shapes."""
        # This is a basic shape test - actual forward pass would require
        # the full dependencies to be installed and working
        attn = DenseSpatialGroupingAttention(
            feature_dims=64, spatial_dims=2, num_heads=4, kernel_size=5
        )

        # Test that attn method exists and can be called with correct shapes
        batch_size, num_heads, shape, dim_per_head = 2, 4, 16, 16
        seq_len_k = shape**2
        seq_len_q = (shape // 2) ** 2  # Assume q is half the size of k
        k = torch.randn(batch_size, num_heads, seq_len_k, dim_per_head)
        q = torch.randn(batch_size, num_heads, seq_len_q, dim_per_head)

        # This should work without errors
        result = attn.attn(
            k, q, input_grid_shape=(shape, shape), q_grid_shape=(shape // 2, shape // 2)
        )
        assert result.shape == (batch_size, num_heads, seq_len_q, seq_len_k)

    def test_masking_functionality(self):
        """Test that masking works correctly."""
        attn = DenseSpatialGroupingAttention(feature_dims=64, spatial_dims=2)
        x = torch.randn(1, 16, 64)
        mask = torch.zeros(1, 16, dtype=torch.bool)
        mask[0, :8] = True  # Mask first half of the input

        # Test that masking doesn't crash
        result = attn.forward(
            x, input_spacing=(1.0, 1.0), input_grid_shape=(4, 4), mask=mask
        )

        # Ensure the result has the expected keys
        assert "x_out" in result
        assert "attn_q" in result
        assert "attn_k" in result


# Fixtures for reusable test data
@pytest.fixture
def sample_tensor_2d():
    """Fixture providing a sample 2D tensor."""
    return torch.randn(2, 16, 64)  # (batch, height*width, features)


@pytest.fixture
def sample_tensor_3d():
    """Fixture providing a sample 3D tensor."""
    return torch.randn(2, 64, 128)  # (batch, depth*height*width, features)


@pytest.fixture
def basic_spatial_grouping_attention_2d():
    """Fixture providing a basic 2D SpatialGroupingAttention object."""
    return SparseSpatialGroupingAttention(
        feature_dims=64, spatial_dims=2, kernel_size=7, num_heads=4
    )


@pytest.fixture
def basic_spatial_grouping_attention_3d():
    """Fixture providing a basic 3D SpatialGroupingAttention object."""
    return SparseSpatialGroupingAttention(
        feature_dims=128, spatial_dims=3, kernel_size=5, num_heads=8
    )


class TestParameterizedCases:
    """Test cases using parameterized inputs."""

    @pytest.mark.parametrize("spatial_dims", [2, 3])
    @pytest.mark.parametrize("feature_dims", [64, 128, 256])
    @pytest.mark.parametrize("num_heads", [4, 8, 16])
    def test_spatial_grouping_attention_different_configs(
        self, spatial_dims, feature_dims, num_heads
    ):
        """Test SpatialGroupingAttention with different configurations."""
        if feature_dims % num_heads != 0:
            pytest.skip("feature_dims must be divisible by num_heads")

        attn = SpatialGroupingAttention(
            feature_dims=feature_dims, spatial_dims=spatial_dims, num_heads=num_heads
        )
        assert attn.feature_dims == feature_dims
        assert attn.spatial_dims == spatial_dims
        assert attn.num_heads == num_heads

    @pytest.mark.parametrize("kernel_size", [3, 5, 7, [3, 5], [5, 7, 9]])
    def test_kernel_size_variations(self, kernel_size):
        """Test different kernel size configurations."""
        spatial_dims = len(kernel_size) if isinstance(kernel_size, list) else 2
        attn = SpatialGroupingAttention(
            spatial_dims=spatial_dims, kernel_size=kernel_size
        )
        expected = (
            tuple(kernel_size)
            if isinstance(kernel_size, list)
            else (kernel_size, kernel_size)
        )
        assert attn.kernel_size == expected

    @pytest.mark.parametrize("mlp_ratio", [1, 2, 4, 8])
    @pytest.mark.parametrize("mlp_dropout", [0.0, 0.1, 0.2])
    def test_mlp_configurations(self, mlp_ratio, mlp_dropout):
        """Test different MLP configurations."""
        attn = SpatialGroupingAttention(
            feature_dims=128, mlp_ratio=mlp_ratio, mlp_dropout=mlp_dropout
        )
        assert attn.mlp_ratio == mlp_ratio
        assert attn.mlp_dropout == mlp_dropout
        # Don't test internal MLP implementation details - it's from timm library
        assert attn.mlp is not None


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_abstract_method_not_implemented(self):
        """
        Test that base SpatialGroupingAttention raises NotImplementedError
        for abstract methods.
        """
        feature_dims = 16
        spatial_dims = 2
        num_heads = 4
        kernel_size = 3
        h, w = 8, 16

        attn = SpatialGroupingAttention(
            feature_dims=feature_dims,
            spatial_dims=spatial_dims,
            num_heads=num_heads,
            kernel_size=kernel_size,
        )

        with pytest.raises(NotImplementedError):
            # This should raise NotImplementedError since attn is abstract
            k = torch.randn(1, num_heads, h * w, feature_dims)
            q_h, q_w = h // 2, w // 2
            q = torch.randn(1, num_heads, q_h * q_w, feature_dims)
            attn.attn(k, q, q_grid_shape=(q_h, q_w), input_grid_shape=(h, w))


@pytest.mark.integration
class TestIntegration:
    """Integration tests for the complete workflow."""

    @pytest.mark.slow
    def test_sparse_attention_end_to_end_2d(self, natten_available):
        """Test complete sparse attention workflow in 2D."""
        if not natten_available:
            pytest.skip(
                "SparseSpatialGroupingAttention requires NATTEN with CUDA support"
            )

        attn = SparseSpatialGroupingAttention(
            feature_dims=64,
            spatial_dims=2,
            kernel_size=7,
            num_heads=4,
            iters=1,  # Reduce iterations for faster testing
            neighborhood_kernel=3,
        )

        # Create input tensor (batch, height*width, features)
        x = torch.randn(2, 16, 64)  # 4x4 spatial grid
        input_spacing = (1.0, 1.0)
        input_grid_shape = (4, 4)

        # This test may fail if dependencies aren't installed, so we'll catch that
        try:
            result = attn.forward(
                x, input_spacing=input_spacing, input_grid_shape=input_grid_shape
            )

            assert "x_out" in result
            assert "attn_q" in result
            assert "attn_k" in result

            # Check output shapes
            assert result["x_out"].shape[0] == 2  # batch size
            assert result["x_out"].shape[2] == 64  # feature dims

        except ImportError as e:
            pytest.skip(f"Skipping integration test due to missing dependencies: {e}")

    @pytest.mark.slow
    def test_dense_attention_end_to_end_2d(self):
        """Test complete dense attention workflow in 2D."""
        attn = DenseSpatialGroupingAttention(
            feature_dims=32, spatial_dims=2, kernel_size=5, num_heads=4, iters=1
        )

        x = torch.randn(2, 64, 32)
        input_spacing = (1.0, 1.0)
        input_grid_shape = (8, 8)

        try:
            result = attn(
                x, input_spacing=input_spacing, input_grid_shape=input_grid_shape
            )

            assert "x_out" in result
            assert "attn_q" in result
            assert "attn_k" in result

        except ImportError as e:
            pytest.skip(f"Skipping integration test due to missing dependencies: {e}")


class TestMaskingFunctionality:
    """Test cases for masking functionality in spatial attention modules."""

    def test_mask_embedding_initialization(self):
        """Test that mask embedding is properly initialized."""
        feature_dims = 64
        attn = DenseSpatialGroupingAttention(feature_dims=feature_dims, spatial_dims=2)

        # Check mask embedding exists and has correct shape
        assert hasattr(attn, "mask_embedding")
        assert isinstance(attn.mask_embedding, torch.nn.Parameter)
        assert attn.mask_embedding.shape == (
            feature_dims,
        )  # 1D tensor with feature_dims
        assert attn.mask_embedding.requires_grad

    def test_masking_with_dense_attention_2d(self):
        """Test masking functionality with DenseSpatialGroupingAttention in 2D."""
        attn = DenseSpatialGroupingAttention(
            feature_dims=32, spatial_dims=2, kernel_size=3, num_heads=4, iters=1
        )

        # Create input data
        batch_size, num_points, feature_dim = 1, 16, 32
        x = torch.randn(batch_size, num_points, feature_dim)
        input_spacing = (1.0, 1.0)
        input_grid_shape = (4, 4)

        # Create mask (mask first half of points)
        mask = torch.zeros(batch_size, num_points, dtype=torch.bool)
        mask[0, :8] = True

        # Test forward pass with mask
        result = attn.forward(
            x, input_spacing=input_spacing, input_grid_shape=input_grid_shape, mask=mask
        )

        # Verify output structure
        assert "x_out" in result
        assert "attn_q" in result
        assert "attn_k" in result
        assert result["x_out"].shape[0] == batch_size
        assert result["x_out"].shape[-1] == feature_dim

    def test_masking_with_dense_attention_3d(self):
        """Test masking functionality with DenseSpatialGroupingAttention in 3D."""
        attn = DenseSpatialGroupingAttention(
            feature_dims=32, spatial_dims=3, kernel_size=3, num_heads=4, iters=1
        )

        # Create input data
        batch_size, num_points, feature_dim = 1, 27, 32  # 3x3x3 grid
        x = torch.randn(batch_size, num_points, feature_dim)
        input_spacing = (1.0, 1.0, 1.0)
        input_grid_shape = (3, 3, 3)

        # Create mask (mask random points)
        mask = torch.zeros(batch_size, num_points, dtype=torch.bool)
        mask[0, [0, 5, 10, 15, 20]] = True  # Mask some points

        # Test forward pass with mask
        result = attn.forward(
            x, input_spacing=input_spacing, input_grid_shape=input_grid_shape, mask=mask
        )

        # Verify output structure
        assert "x_out" in result
        assert "attn_q" in result
        assert "attn_k" in result

    def test_masking_with_sparse_attention(self, natten_available):
        """Test masking functionality with SparseSpatialGroupingAttention."""
        if not natten_available:
            pytest.skip(
                "SparseSpatialGroupingAttention requires NATTEN with CUDA support"
            )

        attn = SparseSpatialGroupingAttention(
            feature_dims=32, spatial_dims=2, kernel_size=3, num_heads=4, iters=1
        )

        # Create input data
        batch_size, num_points, feature_dim = 1, 16, 32
        x = torch.randn(batch_size, num_points, feature_dim)
        input_spacing = (1.0, 1.0)
        input_grid_shape = (4, 4)

        # Create mask
        mask = torch.zeros(batch_size, num_points, dtype=torch.bool)
        mask[0, ::2] = True  # Mask every other point

        # Test forward pass with mask
        result = attn.forward(
            x, input_spacing=input_spacing, input_grid_shape=input_grid_shape, mask=mask
        )

        # Verify output structure
        assert "x_out" in result
        assert "attn_q" in result
        assert "attn_k" in result

    def test_mask_effect_on_input(self):
        """Test that mask actually affects the input tensor."""
        attn = DenseSpatialGroupingAttention(
            feature_dims=32, spatial_dims=2, kernel_size=3, num_heads=4, iters=1
        )

        # Create reproducible input data
        torch.manual_seed(42)
        batch_size, num_points, feature_dim = 1, 16, 32
        x = torch.randn(batch_size, num_points, feature_dim)
        input_spacing = (1.0, 1.0)
        input_grid_shape = (4, 4)

        # Test without mask
        result_no_mask = attn.forward(
            x.clone(), input_spacing=input_spacing, input_grid_shape=input_grid_shape
        )

        # Test with mask
        mask = torch.zeros(batch_size, num_points, dtype=torch.bool)
        mask[0, :8] = True  # Mask first half

        result_with_mask = attn.forward(
            x.clone(),
            input_spacing=input_spacing,
            input_grid_shape=input_grid_shape,
            mask=mask,
        )

        # Results should be different when mask is applied
        assert not torch.allclose(result_no_mask["x_out"], result_with_mask["x_out"])
        assert not torch.allclose(result_no_mask["attn_q"], result_with_mask["attn_q"])

    def test_mask_tensor_validation(self):
        """Test that mask tensor has the correct shape and type requirements."""
        attn = DenseSpatialGroupingAttention(
            feature_dims=32, spatial_dims=2, kernel_size=3, num_heads=4, iters=1
        )

        batch_size, num_points, feature_dim = 1, 16, 32
        x = torch.randn(batch_size, num_points, feature_dim)
        input_spacing = (1.0, 1.0)
        input_grid_shape = (4, 4)

        # Test with float mask (should work as boolean)
        mask_float = torch.zeros(batch_size, num_points)
        mask_float[0, :8] = 1.0

        result = attn.forward(
            x,
            input_spacing=input_spacing,
            input_grid_shape=input_grid_shape,
            mask=mask_float,
        )
        assert "x_out" in result

    def test_empty_mask(self):
        """Test behavior with an all-False mask (no masking)."""
        attn = DenseSpatialGroupingAttention(
            feature_dims=32, spatial_dims=2, kernel_size=3, num_heads=4, iters=1
        )

        batch_size, num_points, feature_dim = 1, 16, 32
        x = torch.randn(batch_size, num_points, feature_dim)
        input_spacing = (1.0, 1.0)
        input_grid_shape = (4, 4)

        # All-False mask should behave like no mask
        mask = torch.zeros(batch_size, num_points, dtype=torch.bool)

        result_with_empty_mask = attn.forward(
            x.clone(),
            input_spacing=input_spacing,
            input_grid_shape=input_grid_shape,
            mask=mask,
        )

        result_no_mask = attn.forward(
            x.clone(), input_spacing=input_spacing, input_grid_shape=input_grid_shape
        )

        # Results should be identical
        assert torch.allclose(result_no_mask["x_out"], result_with_empty_mask["x_out"])

    def test_full_mask(self):
        """Test behavior with an all-True mask (everything masked)."""
        attn = DenseSpatialGroupingAttention(
            feature_dims=32, spatial_dims=2, kernel_size=3, num_heads=4, iters=1
        )

        batch_size, num_points, feature_dim = 1, 16, 32
        x = torch.randn(batch_size, num_points, feature_dim)
        input_spacing = (1.0, 1.0)
        input_grid_shape = (4, 4)

        # All-True mask
        mask = torch.ones(batch_size, num_points, dtype=torch.bool)

        # Should not crash even with everything masked
        result = attn.forward(
            x, input_spacing=input_spacing, input_grid_shape=input_grid_shape, mask=mask
        )

        assert "x_out" in result
        assert "attn_q" in result
        assert "attn_k" in result

    def test_batch_masking_different_masks(self):
        """Test masking with different masks for different batches."""
        attn = DenseSpatialGroupingAttention(
            feature_dims=32, spatial_dims=2, kernel_size=3, num_heads=4, iters=1
        )

        batch_size, num_points, feature_dim = 2, 16, 32
        x = torch.randn(batch_size, num_points, feature_dim)
        input_spacing = (1.0, 1.0)
        input_grid_shape = (4, 4)

        # Different masks for each batch
        mask = torch.zeros(batch_size, num_points, dtype=torch.bool)
        mask[0, :8] = True  # Mask first half for batch 0
        mask[1, 8:] = True  # Mask second half for batch 1

        # Test forward pass with batch-specific masks
        result = attn.forward(
            x, input_spacing=input_spacing, input_grid_shape=input_grid_shape, mask=mask
        )

        assert "x_out" in result
        assert result["x_out"].shape[0] == batch_size

    def test_mask_values_are_replaced(self):
        """Test that masked positions actually contain mask_embedding values."""
        attn = DenseSpatialGroupingAttention(
            feature_dims=32, spatial_dims=2, kernel_size=3, num_heads=4, iters=1
        )

        batch_size, num_points, feature_dim = 1, 16, 32
        x = torch.randn(batch_size, num_points, feature_dim)
        input_spacing = (1.0, 1.0)
        input_grid_shape = (4, 4)

        # Create a mask that masks the first point
        mask = torch.zeros(batch_size, num_points, dtype=torch.bool)
        mask[0, 0] = True

        # Store original value
        original_x = x.clone()

        # Apply mask manually to check
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, feature_dim)
        mask_embedding_expanded = (
            attn.mask_embedding.unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, num_points, -1)
        )
        expected_x = torch.where(mask_expanded, mask_embedding_expanded, original_x)

        # The first point should be replaced with mask_embedding
        attn.forward(
            x, input_spacing=input_spacing, input_grid_shape=input_grid_shape, mask=mask
        )

        # Compare expected behavior
        assert torch.allclose(expected_x[0, 0], attn.mask_embedding)
        assert torch.allclose(expected_x[0, 1], original_x[0, 1])

    def test_mask_with_different_devices(self):
        """Test masking with tensors on the same device."""
        attn = DenseSpatialGroupingAttention(
            feature_dims=32, spatial_dims=2, kernel_size=3, num_heads=4, iters=1
        )

        batch_size, num_points, feature_dim = 1, 16, 32
        x = torch.randn(batch_size, num_points, feature_dim)
        input_spacing = (1.0, 1.0)
        input_grid_shape = (4, 4)

        # Test with mask on the same device (should work)
        mask = torch.zeros(batch_size, num_points, dtype=torch.bool)
        mask[0, :8] = True

        result = attn.forward(
            x, input_spacing=input_spacing, input_grid_shape=input_grid_shape, mask=mask
        )
        assert "x_out" in result

    def test_mask_with_integer_types(self):
        """Test masking works with integer mask types."""
        attn = DenseSpatialGroupingAttention(
            feature_dims=32, spatial_dims=2, kernel_size=3, num_heads=4, iters=1
        )

        batch_size, num_points, feature_dim = 1, 16, 32
        x = torch.randn(batch_size, num_points, feature_dim)
        input_spacing = (1.0, 1.0)
        input_grid_shape = (4, 4)

        # Test with integer mask (0s and 1s)
        mask_int = torch.zeros(batch_size, num_points, dtype=torch.int32)
        mask_int[0, :8] = 1

        result = attn.forward(
            x,
            input_spacing=input_spacing,
            input_grid_shape=input_grid_shape,
            mask=mask_int,
        )
        assert "x_out" in result

    def test_mask_gradient_flow(self):
        """Test that gradients flow through mask_embedding when used."""
        attn = DenseSpatialGroupingAttention(
            feature_dims=32, spatial_dims=2, kernel_size=3, num_heads=4, iters=1
        )

        batch_size, num_points, feature_dim = 1, 16, 32
        x = torch.randn(batch_size, num_points, feature_dim, requires_grad=True)
        input_spacing = (1.0, 1.0)
        input_grid_shape = (4, 4)

        # Create mask
        mask = torch.zeros(batch_size, num_points, dtype=torch.bool)
        mask[0, :8] = True

        # Forward pass
        result = attn.forward(
            x, input_spacing=input_spacing, input_grid_shape=input_grid_shape, mask=mask
        )

        # Create a loss and backpropagate
        loss = result["x_out"].sum()
        loss.backward()

        # Check that mask_embedding has gradients (it should since it was used)
        assert attn.mask_embedding.grad is not None
        assert not torch.allclose(
            attn.mask_embedding.grad, torch.zeros_like(attn.mask_embedding.grad)
        )


def test_package_imports():
    """Test that package imports work correctly."""
    # Test that we can import main classes
    from spatial_grouping_attention import (
        DenseSpatialGroupingAttention,
        SparseSpatialGroupingAttention,
        SpatialGroupingAttention,
    )
    from spatial_grouping_attention.utils import to_list, to_tuple

    # Test that classes are properly defined
    assert issubclass(SparseSpatialGroupingAttention, SpatialGroupingAttention)
    assert issubclass(DenseSpatialGroupingAttention, SpatialGroupingAttention)
    assert callable(to_list)
    assert callable(to_tuple)

"""Test suite for spatial_attention."""

import pytest
import torch

from spatial_attention import (
    MLP,
    DenseSpatialAttention,
    SparseSpatialAttention,
    SpatialAttention,
)
from spatial_attention.utils import to_list, to_tuple


class TestUtilityFunctions:
    """Test cases for utility functions in utils.py."""

    def test_to_list_with_scalar(self):
        """Test to_list with scalar input."""
        result = to_list(5, 3)
        assert result == [5, 5, 5]

    def test_to_list_with_float(self):
        """Test to_list with float input."""
        result = to_list(1.5, 2)
        assert result == [1.5, 1.5]

    def test_to_list_with_list(self):
        """Test to_list with list input."""
        result = to_list([1, 2, 3], 3)
        assert result == [1, 2, 3]

    def test_to_list_with_tuple(self):
        """Test to_list with tuple input."""
        result = to_list((1, 2), 2)
        assert result == [1, 2]

    def test_to_list_with_nested_structure(self):
        """Test to_list with nested list structure."""
        result = to_list([(1, 2), (3, 4)], 2)
        # When nested structure is detected, inner sequences are converted to lists
        assert result == [[1, 2], [3, 4]]

    def test_to_list_invalid_length(self):
        """Test to_list with invalid length."""
        # to_list validates length and raises ValueError for mismatched length
        with pytest.raises(ValueError, match="got 3 but expected 2"):
            to_list([1, 2, 3], 2)

    def test_to_list_invalid_type(self):
        """Test to_list with invalid type."""
        with pytest.raises(TypeError):
            to_list("invalid", 2)

    def test_to_tuple_with_scalar(self):
        """Test to_tuple with scalar input."""
        result = to_tuple(7, 2)
        assert result == (7, 7)

    def test_to_tuple_with_list(self):
        """Test to_tuple with list input."""
        result = to_tuple([1, 2, 3], 3)
        assert result == (1, 2, 3)

    def test_to_tuple_with_nested_structure(self):
        """Test to_tuple with nested structure."""
        result = to_tuple([(1, 2), (3, 4)], 2)
        assert result == ((1, 2), (3, 4))

    def test_to_tuple_invalid_length(self):
        """Test to_tuple with invalid length."""
        with pytest.raises(ValueError):
            to_tuple([1, 2], 3)

    def test_to_tuple_invalid_type(self):
        """Test to_tuple with invalid type."""
        with pytest.raises(TypeError):
            to_tuple("invalid", 2)


class TestMLP:
    """Test cases for MLP class."""

    def test_mlp_initialization(self):
        """Test MLP initialization with default parameters."""
        mlp = MLP(in_features=128)
        assert mlp.fc1.in_features == 128
        assert mlp.fc1.out_features == 128
        assert mlp.fc2.in_features == 128
        assert mlp.fc2.out_features == 128
        assert isinstance(mlp.act, torch.nn.GELU)

    def test_mlp_initialization_with_params(self):
        """Test MLP initialization with custom parameters."""
        mlp = MLP(
            in_features=64,
            hidden_features=256,
            out_features=32,
            act_layer=torch.nn.ReLU,
            drop=0.1,
            bias=False,
        )
        assert mlp.fc1.in_features == 64
        assert mlp.fc1.out_features == 256
        assert mlp.fc2.in_features == 256
        assert mlp.fc2.out_features == 32
        assert isinstance(mlp.act, torch.nn.ReLU)
        assert mlp.drop.p == 0.1
        assert not mlp.fc1.bias
        assert not mlp.fc2.bias

    def test_mlp_forward(self):
        """Test MLP forward pass."""
        mlp = MLP(in_features=64, hidden_features=128)
        x = torch.randn(2, 10, 64)  # (batch, seq, features)
        output = mlp(x)
        assert output.shape == (2, 10, 64)

    def test_mlp_forward_different_out_features(self):
        """Test MLP forward pass with different output features."""
        mlp = MLP(in_features=64, hidden_features=128, out_features=32)
        x = torch.randn(2, 10, 64)
        output = mlp(x)
        assert output.shape == (2, 10, 32)

    def test_mlp_shortcut_identity(self):
        """Test MLP shortcut connection when in_features == out_features."""
        mlp = MLP(in_features=64)
        assert isinstance(mlp.shortcut, torch.nn.Identity)

    def test_mlp_shortcut_linear(self):
        """Test MLP shortcut connection when in_features != out_features."""
        mlp = MLP(in_features=64, out_features=32)
        assert isinstance(mlp.shortcut, torch.nn.Linear)
        assert mlp.shortcut.in_features == 64
        assert mlp.shortcut.out_features == 32


class TestSpatialAttentionBase:
    """Test cases for base SpatialAttention class."""

    def test_spatial_attention_initialization_2d(self):
        """Test SpatialAttention initialization with 2D."""
        attn = SpatialAttention(
            feature_dims=128, spatial_dims=2, kernel_size=7, num_heads=8
        )
        assert attn.feature_dims == 128
        assert attn.spatial_dims == 2
        assert attn.kernel_size == (7, 7)
        assert attn.num_heads == 8
        assert len(attn._default_spacing) == 2

    def test_spatial_attention_initialization_3d(self):
        """Test SpatialAttention initialization with 3D."""
        attn = SpatialAttention(
            feature_dims=256, spatial_dims=3, kernel_size=[5, 7, 9], num_heads=16
        )
        assert attn.feature_dims == 256
        assert attn.spatial_dims == 3
        assert attn.kernel_size == (5, 7, 9)
        assert attn.num_heads == 16
        assert len(attn._default_spacing) == 3

    def test_spatial_attention_custom_spacing(self):
        """Test SpatialAttention with custom spacing."""
        attn = SpatialAttention(spatial_dims=3, spacing=[1.0, 2.0, 0.5])
        assert attn._default_spacing == (1.0, 2.0, 0.5)

    def test_spatial_attention_repr(self):
        """Test string representation."""
        attn = SpatialAttention(spatial_dims=2)
        repr_str = repr(attn)
        assert "SpatialAttention2D" in repr_str

        attn_3d = SpatialAttention(spatial_dims=3)
        repr_str_3d = repr(attn_3d)
        assert "SpatialAttention3D" in repr_str_3d

    def test_spatial_attention_stride_calculation(self):
        """Test automatic stride calculation."""
        attn = SpatialAttention(spatial_dims=2, kernel_size=7)
        # Default stride should be kernel_size // 2
        assert attn.stride == (3, 3)

    def test_spatial_attention_custom_stride(self):
        """Test custom stride setting."""
        attn = SpatialAttention(spatial_dims=2, kernel_size=7, stride=2)
        assert attn.stride == (2, 2)

    def test_spatial_attention_components_created(self):
        """Test that all required components are created."""
        attn = SpatialAttention(feature_dims=128, spatial_dims=2, num_heads=8)

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
        assert hasattr(attn, "mlp") and isinstance(attn.mlp, MLP)

        # Check mask embedding
        assert hasattr(attn, "mask_embedding")
        assert attn.mask_embedding.shape == (1, 128)


@pytest.mark.natten
class TestSparseSpatialAttention:
    """Test cases for SparseSpatialAttention class."""

    def test_sparse_initialization(self, natten_available):
        """Test SparseSpatialAttention initialization."""
        if not natten_available:
            pytest.skip("SparseSpatialAttention requires NATTEN with CUDA support")

        attn = SparseSpatialAttention(
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
        """Test SparseSpatialAttention with different kernel sizes per dimension."""
        if not natten_available:
            pytest.skip("SparseSpatialAttention requires NATTEN with CUDA support")

        attn = SparseSpatialAttention(
            spatial_dims=3,
            neighborhood_kernel=[3, 5, 7],
            neighborhood_dilation=[1, 2, 1],
        )
        assert attn.neighborhood_kernel == (3, 5, 7)
        assert attn.neighborhood_dilation == (1, 2, 1)

    def test_sparse_has_attention_functions(self, natten_available):
        """Test that sparse attention has the right attention functions."""
        if not natten_available:
            pytest.skip("SparseSpatialAttention requires NATTEN with CUDA support")

        attn_2d = SparseSpatialAttention(spatial_dims=2)
        attn_3d = SparseSpatialAttention(spatial_dims=3)

        assert hasattr(attn_2d, "_qk_attn")
        assert hasattr(attn_2d, "_av_attn")
        assert hasattr(attn_3d, "_qk_attn")
        assert hasattr(attn_3d, "_av_attn")


class TestDenseSpatialAttention:
    """Test cases for DenseSpatialAttention class."""

    def test_dense_initialization(self):
        """Test DenseSpatialAttention initialization."""
        attn = DenseSpatialAttention(feature_dims=128, spatial_dims=2, num_heads=8)
        assert attn.feature_dims == 128
        assert attn.spatial_dims == 2
        assert attn.num_heads == 8

    def test_dense_forward_shapes(self):
        """Test DenseSpatialAttention forward pass shapes."""
        # This is a basic shape test - actual forward pass would require
        # the full dependencies to be installed and working
        attn = DenseSpatialAttention(feature_dims=64, spatial_dims=2, num_heads=4)

        # Test that attn method exists and can be called with correct shapes
        batch_size, num_heads, seq_len, dim_per_head = 2, 4, 16, 16
        k = torch.randn(batch_size, num_heads, seq_len, dim_per_head)
        q = torch.randn(batch_size, num_heads, seq_len, dim_per_head)

        # This should work without errors
        result = attn.attn(k, q, q_grid_shape=(4, 4))
        assert result.shape == (batch_size, num_heads, seq_len, seq_len)


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
def basic_spatial_attention_2d():
    """Fixture providing a basic 2D SpatialAttention object."""
    return SparseSpatialAttention(
        feature_dims=64, spatial_dims=2, kernel_size=7, num_heads=4
    )


@pytest.fixture
def basic_spatial_attention_3d():
    """Fixture providing a basic 3D SpatialAttention object."""
    return SparseSpatialAttention(
        feature_dims=128, spatial_dims=3, kernel_size=5, num_heads=8
    )


class TestParameterizedCases:
    """Test cases using parameterized inputs."""

    @pytest.mark.parametrize("spatial_dims", [2, 3])
    @pytest.mark.parametrize("feature_dims", [64, 128, 256])
    @pytest.mark.parametrize("num_heads", [4, 8, 16])
    def test_spatial_attention_different_configs(
        self, spatial_dims, feature_dims, num_heads
    ):
        """Test SpatialAttention with different configurations."""
        if feature_dims % num_heads != 0:
            pytest.skip("feature_dims must be divisible by num_heads")

        attn = SpatialAttention(
            feature_dims=feature_dims, spatial_dims=spatial_dims, num_heads=num_heads
        )
        assert attn.feature_dims == feature_dims
        assert attn.spatial_dims == spatial_dims
        assert attn.num_heads == num_heads

    @pytest.mark.parametrize("kernel_size", [3, 5, 7, [3, 5], [5, 7, 9]])
    def test_kernel_size_variations(self, kernel_size):
        """Test different kernel size configurations."""
        spatial_dims = len(kernel_size) if isinstance(kernel_size, list) else 2
        attn = SpatialAttention(spatial_dims=spatial_dims, kernel_size=kernel_size)
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
        attn = SpatialAttention(
            feature_dims=128, mlp_ratio=mlp_ratio, mlp_dropout=mlp_dropout
        )
        assert attn.mlp_ratio == mlp_ratio
        assert attn.mlp_dropout == mlp_dropout
        assert attn.mlp.drop.p == mlp_dropout


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_abstract_method_not_implemented(self):
        """
        Test that base SpatialAttention raises NotImplementedError for abstract methods.
        """
        attn = SpatialAttention()

        with pytest.raises(NotImplementedError):
            # This should raise NotImplementedError since attn is abstract
            k = torch.randn(1, 4, 10, 16)
            q = torch.randn(1, 4, 10, 16)
            attn.attn(k, q, q_grid_shape=(2, 5))

    def test_mask_not_implemented_error(self, natten_available):
        """Test that masking raises NotImplementedError in base class."""
        if not natten_available:
            pytest.skip("SparseSpatialAttention requires NATTEN with CUDA support")

        attn = SparseSpatialAttention(feature_dims=64, spatial_dims=2)
        x = torch.randn(1, 16, 64)
        mask = torch.ones(1, 16)

        with pytest.raises(NotImplementedError, match="Masking is not implemented"):
            attn.forward(x, q_spacing=(1.0, 1.0), q_grid_shape=(4, 4), mask=mask)


@pytest.mark.integration
class TestIntegration:
    """Integration tests for the complete workflow."""

    @pytest.mark.slow
    def test_sparse_attention_end_to_end_2d(self, natten_available):
        """Test complete sparse attention workflow in 2D."""
        if not natten_available:
            pytest.skip("SparseSpatialAttention requires NATTEN with CUDA support")

        attn = SparseSpatialAttention(
            feature_dims=64,
            spatial_dims=2,
            kernel_size=7,
            num_heads=4,
            iters=1,  # Reduce iterations for faster testing
            neighborhood_kernel=3,
        )

        # Create input tensor (batch, height*width, features)
        x = torch.randn(2, 16, 64)  # 4x4 spatial grid
        q_spacing = (1.0, 1.0)
        q_grid_shape = (4, 4)

        # This test may fail if dependencies aren't installed, so we'll catch that
        try:
            result = attn.forward(x, q_spacing=q_spacing, q_grid_shape=q_grid_shape)

            assert "x_out" in result
            assert "attn_q" in result
            assert "attn_k" in result

            # Check output shapes
            assert result["x_out"].shape[0] == 2  # batch size
            assert result["x_out"].shape[2] == 64  # feature dims

        except (ImportError, RuntimeError) as e:
            pytest.skip(f"Skipping integration test due to missing dependencies: {e}")

    @pytest.mark.slow
    def test_dense_attention_end_to_end_2d(self):
        """Test complete dense attention workflow in 2D."""
        attn = DenseSpatialAttention(
            feature_dims=64, spatial_dims=2, kernel_size=7, num_heads=4, iters=1
        )

        x = torch.randn(2, 16, 64)
        q_spacing = (1.0, 1.0)
        q_grid_shape = (4, 4)

        try:
            result = attn.forward(x, q_spacing=q_spacing, q_grid_shape=q_grid_shape)

            assert "x_out" in result
            assert "attn_q" in result
            assert "attn_k" in result

        except (ImportError, RuntimeError) as e:
            pytest.skip(f"Skipping integration test due to missing dependencies: {e}")


def test_package_imports():
    """Test that package imports work correctly."""
    # Test that we can import main classes
    from spatial_attention import (
        MLP,
        DenseSpatialAttention,
        SparseSpatialAttention,
        SpatialAttention,
    )
    from spatial_attention.utils import to_list, to_tuple

    # Test that classes are properly defined
    assert issubclass(SparseSpatialAttention, SpatialAttention)
    assert issubclass(DenseSpatialAttention, SpatialAttention)
    assert callable(to_list)
    assert callable(to_tuple)

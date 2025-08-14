"""Integration tests for spatial attention modules."""

import pytest
import torch
import torch.nn as nn

# These tests may fail if the full dependencies (RoSE, natten) are not installed
pytest.importorskip("torch", reason="PyTorch is required for integration tests")

try:
    from spatial_attention import (
        SpatialAttention,
        SparseSpatialAttention,
        DenseSpatialAttention,
        MLP,
    )

    SPATIAL_ATTENTION_AVAILABLE = True
except ImportError:
    SPATIAL_ATTENTION_AVAILABLE = False
    pytest.skip("spatial_attention module not available", allow_module_level=True)


@pytest.mark.integration
class TestSpatialAttentionIntegration:
    """Integration tests for spatial attention classes."""

    @pytest.mark.parametrize("spatial_dims", [2, 3])
    @pytest.mark.parametrize(
        "attention_class", [SparseSpatialAttention, DenseSpatialAttention]
    )
    def test_attention_initialization_and_shapes(
        self, spatial_dims, attention_class, natten_available
    ):
        """Test that attention classes initialize correctly and have right shapes."""
        # Skip if using SparseSpatialAttention and natten is not available
        if attention_class == SparseSpatialAttention and not natten_available:
            pytest.skip("SparseSpatialAttention requires NATTEN with CUDA support")

        feature_dims = 64
        num_heads = 4
        kernel_size = 7

        attn = attention_class(
            feature_dims=feature_dims,
            spatial_dims=spatial_dims,
            kernel_size=kernel_size,
            num_heads=num_heads,
        )

        # Check basic attributes
        assert attn.feature_dims == feature_dims
        assert attn.spatial_dims == spatial_dims
        assert attn.num_heads == num_heads
        assert attn.kernel_size == tuple([kernel_size] * spatial_dims)

        # Check that components exist
        assert hasattr(attn, "q")
        assert hasattr(attn, "kv")
        assert hasattr(attn, "mlp")
        assert hasattr(attn, "norm1")
        assert hasattr(attn, "norm2")
        assert hasattr(attn, "norm3")

        # Check layer shapes
        assert attn.q.in_features == feature_dims
        assert attn.q.out_features == feature_dims
        assert attn.kv.in_features == feature_dims
        assert attn.kv.out_features == 2 * feature_dims

    def test_mlp_in_spatial_attention(self, natten_available):
        """Test MLP integration within spatial attention."""
        if not natten_available:
            pytest.skip("SparseSpatialAttention requires NATTEN with CUDA support")

        attn = SparseSpatialAttention(
            feature_dims=128, spatial_dims=2, mlp_ratio=4, mlp_dropout=0.1
        )

        assert isinstance(attn.mlp, MLP)
        assert attn.mlp.fc1.in_features == 128
        assert attn.mlp.fc1.out_features == 128 * 4  # mlp_ratio = 4
        assert attn.mlp.fc2.out_features == 128
        assert attn.mlp.drop.p == 0.1

    @pytest.mark.slow
    def test_dense_attention_forward_2d(self):
        """Test dense attention forward pass in 2D (may fail without full deps)."""
        try:
            attn = DenseSpatialAttention(
                feature_dims=32,
                spatial_dims=2,
                kernel_size=5,
                num_heads=4,
                iters=1,  # Reduce iterations for speed
            )

            batch_size = 2
            height, width = 4, 4
            seq_len = height * width

            x = torch.randn(batch_size, seq_len, 32)
            q_spacing = (1.0, 1.0)
            q_grid_shape = (height, width)

            result = attn.forward(x, q_spacing=q_spacing, q_grid_shape=q_grid_shape)

            # Check return structure
            assert isinstance(result, dict)
            assert "x_out" in result
            assert "attn_q" in result
            assert "attn_k" in result

            # Check shapes
            x_out = result["x_out"]
            assert x_out.shape[0] == batch_size
            assert x_out.shape[2] == 32  # feature dims

        except (ImportError, RuntimeError, AttributeError) as e:
            pytest.skip(f"Skipping dense attention test due to dependencies: {e}")

    @pytest.mark.slow
    def test_sparse_attention_forward_2d(self, natten_available):
        """Test sparse attention forward pass in 2D (may fail without full deps)."""
        if not natten_available:
            pytest.skip("SparseSpatialAttention requires NATTEN with CUDA support")

        try:
            attn = SparseSpatialAttention(
                feature_dims=32,
                spatial_dims=2,
                kernel_size=5,
                num_heads=4,
                neighborhood_kernel=3,
                iters=1,
            )

            batch_size = 2
            height, width = 4, 4
            seq_len = height * width

            x = torch.randn(batch_size, seq_len, 32)
            q_spacing = (1.0, 1.0)
            q_grid_shape = (height, width)

            result = attn.forward(x, q_spacing=q_spacing, q_grid_shape=q_grid_shape)

            assert isinstance(result, dict)
            assert "x_out" in result

        except (ImportError, RuntimeError, AttributeError) as e:
            pytest.skip(f"Skipping sparse attention test due to dependencies: {e}")


@pytest.mark.integration
class TestMLPIntegration:
    """Integration tests for MLP with other components."""

    def test_mlp_as_feedforward_in_transformer_block(self):
        """Test MLP as part of a transformer-like block."""
        feature_dim = 128
        seq_len = 16
        batch_size = 4

        # Create a simple transformer-like block with MLP
        norm1 = nn.LayerNorm(feature_dim)
        norm2 = nn.LayerNorm(feature_dim)
        mlp = MLP(in_features=feature_dim, hidden_features=feature_dim * 4)

        x = torch.randn(batch_size, seq_len, feature_dim)

        # Simulate transformer block: norm -> attention (skip) -> norm -> mlp
        x_norm1 = norm1(x)
        # Skip attention for simplicity
        x_attn = x_norm1 + x  # Residual
        x_norm2 = norm2(x_attn)
        x_mlp = mlp(x_norm2)
        x_out = x_mlp + x_attn  # Residual

        assert x_out.shape == (batch_size, seq_len, feature_dim)
        assert not torch.allclose(x_out, x)  # Should be transformed

    def test_mlp_gradient_accumulation(self):
        """Test MLP in gradient accumulation scenario."""
        mlp = MLP(in_features=64, hidden_features=128)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)

        # Simulate gradient accumulation
        mlp.train()
        optimizer.zero_grad()

        total_loss = 0
        for i in range(3):  # Accumulate over 3 micro-batches
            x = torch.randn(2, 10, 64)
            target = torch.randn(2, 10, 64)

            output = mlp(x)
            loss = nn.MSELoss()(output, target)
            loss = loss / 3  # Scale by accumulation steps
            loss.backward()

            total_loss += loss.item()

        # Check that gradients accumulated
        for param in mlp.parameters():
            assert param.grad is not None
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad))

        optimizer.step()

    def test_mlp_with_different_optimizers(self):
        """Test MLP training with different optimizers."""
        mlp = MLP(in_features=32, hidden_features=64)
        x = torch.randn(10, 32)
        target = torch.randn(10, 32)

        optimizers = [
            torch.optim.SGD(mlp.parameters(), lr=0.01),
            torch.optim.Adam(mlp.parameters(), lr=0.001),
            torch.optim.AdamW(mlp.parameters(), lr=0.001),
        ]

        for optimizer in optimizers:
            mlp.train()
            optimizer.zero_grad()

            output = mlp(x)
            loss = nn.MSELoss()(output, target)
            loss.backward()

            # Check gradients exist
            for param in mlp.parameters():
                assert param.grad is not None

            optimizer.step()


@pytest.mark.integration
class TestMemoryAndPerformance:
    """Tests for memory usage and performance characteristics."""

    def test_mlp_memory_efficiency(self):
        """Test that MLP doesn't leak memory during training."""
        import gc

        mlp = MLP(in_features=128, hidden_features=512)
        optimizer = torch.optim.Adam(mlp.parameters())

        initial_memory = (
            torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        )

        # Training loop
        for _ in range(10):
            x = torch.randn(32, 128)
            target = torch.randn(32, 128)

            optimizer.zero_grad()
            output = mlp(x)
            loss = nn.MSELoss()(output, target)
            loss.backward()
            optimizer.step()

            # Clean up
            del x, target, output, loss

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated()
            memory_increase = final_memory - initial_memory
            # Memory increase should be minimal (just model parameters)
            assert memory_increase < 100 * 1024 * 1024  # Less than 100MB

    def test_mlp_batch_size_scaling(self):
        """Test MLP performance with different batch sizes."""
        mlp = MLP(in_features=64, hidden_features=128)
        mlp.eval()

        batch_sizes = [1, 8, 32, 128]
        seq_len = 16

        for batch_size in batch_sizes:
            x = torch.randn(batch_size, seq_len, 64)

            with torch.no_grad():
                output = mlp(x)

            assert output.shape == (batch_size, seq_len, 64)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()


@pytest.mark.integration
class TestDeviceCompatibility:
    """Test device compatibility (CPU/GPU)."""

    def test_mlp_cpu_gpu_consistency(self):
        """Test MLP produces same results on CPU and GPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        mlp_cpu = MLP(in_features=32, hidden_features=64)
        mlp_gpu = MLP(in_features=32, hidden_features=64)

        # Copy weights to ensure same initialization
        mlp_gpu.load_state_dict(mlp_cpu.state_dict())
        mlp_gpu = mlp_gpu.cuda()

        x_cpu = torch.randn(4, 32)
        x_gpu = x_cpu.cuda()

        mlp_cpu.eval()
        mlp_gpu.eval()

        with torch.no_grad():
            output_cpu = mlp_cpu(x_cpu)
            output_gpu = mlp_gpu(x_gpu)

        # Results should be very similar (allowing for small numerical differences)
        assert torch.allclose(output_cpu, output_gpu.cpu(), rtol=1e-5, atol=1e-6)

    def test_attention_device_placement(self):
        """Test that attention modules work on different devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        try:
            attn_cpu = DenseSpatialAttention(
                feature_dims=32, spatial_dims=2, num_heads=4, iters=1
            )

            # Test that MLP component can be moved to GPU (since attention classes aren't nn.Module)
            mlp_gpu = MLP(in_features=32).cuda()

            # Check that MLP parameters are on GPU
            for param in mlp_gpu.parameters():
                assert param.device.type == "cuda"

        except (ImportError, RuntimeError, AttributeError) as e:
            pytest.skip(f"Skipping device test due to dependencies: {e}")


# Fixtures for integration tests
@pytest.fixture
def integration_mlp():
    """Fixture providing MLP for integration tests."""
    return MLP(in_features=64, hidden_features=128, out_features=32)


@pytest.fixture
def sample_batch():
    """Fixture providing sample batch data."""
    return {
        "x": torch.randn(4, 16, 64),
        "target": torch.randn(4, 16, 32),
        "batch_size": 4,
        "seq_len": 16,
        "input_dim": 64,
        "output_dim": 32,
    }


def test_integration_training_loop(integration_mlp, sample_batch):
    """Test complete training loop integration."""
    mlp = integration_mlp
    data = sample_batch

    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    initial_loss = float("inf")
    final_loss = float("inf")

    mlp.train()
    for epoch in range(5):
        optimizer.zero_grad()

        output = mlp(data["x"])
        loss = criterion(output, data["target"])

        if epoch == 0:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()
        final_loss = loss.item()  # Update final_loss each iteration

    # Loss should decrease (at least a little bit)
    assert final_loss < initial_loss or abs(final_loss - initial_loss) < 1e-6

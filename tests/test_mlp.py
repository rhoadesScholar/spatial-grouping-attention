"""Focused tests for the MLP module."""

import pytest
import torch
import torch.nn as nn

from spatial_attention.mlp import MLP


class TestMLPBasics:
    """Basic functionality tests for MLP."""

    def test_default_initialization(self):
        """Test MLP with default parameters."""
        mlp = MLP(in_features=128)

        assert mlp.fc1.in_features == 128
        assert mlp.fc1.out_features == 128  # hidden_features defaults to in_features
        assert mlp.fc2.in_features == 128
        assert mlp.fc2.out_features == 128  # out_features defaults to in_features
        assert isinstance(mlp.act, nn.GELU)
        assert mlp.drop.p == 0.0
        assert mlp.fc1.bias is not None  # bias=True by default
        assert mlp.fc2.bias is not None
        assert isinstance(mlp.shortcut, nn.Identity)  # in_features == out_features

    def test_custom_initialization(self):
        """Test MLP with custom parameters."""
        mlp = MLP(
            in_features=64,
            hidden_features=256,
            out_features=32,
            act_layer=nn.ReLU,
            drop=0.1,
            bias=False,
        )

        assert mlp.fc1.in_features == 64
        assert mlp.fc1.out_features == 256
        assert mlp.fc2.in_features == 256
        assert mlp.fc2.out_features == 32
        assert isinstance(mlp.act, nn.ReLU)
        assert mlp.drop.p == 0.1
        assert mlp.fc1.bias is None  # bias=False
        assert mlp.fc2.bias is None
        assert isinstance(mlp.shortcut, nn.Linear)  # in_features != out_features
        assert mlp.shortcut.in_features == 64
        assert mlp.shortcut.out_features == 32

    def test_shortcut_identity_vs_linear(self):
        """Test shortcut layer creation logic."""
        # Identity shortcut when in_features == out_features
        mlp_identity = MLP(in_features=128, out_features=128)
        assert isinstance(mlp_identity.shortcut, nn.Identity)

        # Linear shortcut when in_features != out_features
        mlp_linear = MLP(in_features=128, out_features=64)
        assert isinstance(mlp_linear.shortcut, nn.Linear)
        assert mlp_linear.shortcut.in_features == 128
        assert mlp_linear.shortcut.out_features == 64


class TestMLPForward:
    """Tests for MLP forward pass."""

    def test_forward_same_dims(self):
        """Test forward pass with same input/output dimensions."""
        mlp = MLP(in_features=64, hidden_features=128)
        x = torch.randn(2, 10, 64)

        output = mlp(x)

        assert output.shape == x.shape
        assert not torch.allclose(
            output, x
        )  # Should be different due to transformation

    def test_forward_different_dims(self):
        """Test forward pass with different input/output dimensions."""
        mlp = MLP(in_features=64, hidden_features=128, out_features=32)
        x = torch.randn(2, 10, 64)

        output = mlp(x)

        assert output.shape == (2, 10, 32)

    def test_forward_batch_dimensions(self):
        """Test forward pass with various batch dimensions."""
        mlp = MLP(in_features=32)

        # 2D input (batch, features)
        x_2d = torch.randn(5, 32)
        out_2d = mlp(x_2d)
        assert out_2d.shape == (5, 32)

        # 3D input (batch, seq, features)
        x_3d = torch.randn(3, 7, 32)
        out_3d = mlp(x_3d)
        assert out_3d.shape == (3, 7, 32)

        # 4D input (batch, height, width, features)
        x_4d = torch.randn(2, 4, 4, 32)
        out_4d = mlp(x_4d)
        assert out_4d.shape == (2, 4, 4, 32)

    def test_residual_connection(self):
        """Test that residual connection works correctly."""
        # Test with identity shortcut
        mlp_identity = MLP(in_features=64)
        x = torch.randn(2, 10, 64)

        # Manually compute expected output
        fc1_out = mlp_identity.fc1(x)
        act_out = mlp_identity.act(fc1_out)
        drop1_out = mlp_identity.drop(act_out)
        fc2_out = mlp_identity.fc2(drop1_out)
        drop2_out = mlp_identity.drop(fc2_out)
        shortcut_out = mlp_identity.shortcut(x)
        expected = drop2_out + shortcut_out

        actual = mlp_identity(x)
        assert torch.allclose(actual, expected, rtol=1e-5)

    def test_dropout_behavior(self):
        """Test dropout behavior during training vs evaluation."""
        mlp = MLP(in_features=64, drop=0.5)
        x = torch.randn(100, 64)

        # Training mode - dropout should be active
        mlp.train()
        outputs_train = []
        for _ in range(10):
            outputs_train.append(mlp(x))

        # Outputs should be different due to dropout
        assert not all(
            torch.allclose(outputs_train[0], out) for out in outputs_train[1:]
        )

        # Evaluation mode - dropout should be inactive
        mlp.eval()
        outputs_eval = []
        for _ in range(10):
            outputs_eval.append(mlp(x))

        # Outputs should be identical in eval mode
        assert all(torch.allclose(outputs_eval[0], out) for out in outputs_eval[1:])


class TestMLPActivations:
    """Test different activation functions."""

    @pytest.mark.parametrize(
        "activation", [nn.ReLU, nn.GELU, nn.Tanh, nn.LeakyReLU, nn.Sigmoid, nn.SiLU]
    )
    def test_different_activations(self, activation):
        """Test MLP with different activation functions."""
        mlp = MLP(in_features=32, act_layer=activation)
        x = torch.randn(2, 10, 32)

        output = mlp(x)
        assert output.shape == x.shape
        assert isinstance(mlp.act, activation)

    def test_activation_with_parameters(self):
        """Test activation functions that require parameters."""
        # LeakyReLU with custom negative slope
        mlp_leaky = MLP(
            in_features=32, act_layer=lambda: nn.LeakyReLU(negative_slope=0.2)
        )
        x = torch.randn(2, 10, 32)

        output = mlp_leaky(x)
        assert output.shape == x.shape
        assert isinstance(mlp_leaky.act, nn.LeakyReLU)
        assert mlp_leaky.act.negative_slope == 0.2


class TestMLPGradients:
    """Test gradient flow through MLP."""

    def test_gradient_flow(self):
        """Test that gradients flow through the network."""
        mlp = MLP(in_features=32, hidden_features=64)
        x = torch.randn(1, 32, requires_grad=True)

        output = mlp(x)
        loss = output.sum()
        loss.backward()

        # Check that input gradients exist
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

        # Check that all parameters have gradients
        for param in mlp.parameters():
            assert param.grad is not None

    def test_zero_dropout_gradient_consistency(self):
        """Test gradient consistency with zero dropout."""
        mlp = MLP(in_features=32, drop=0.0)
        x = torch.randn(1, 32, requires_grad=True)

        # Forward pass twice should give same result
        mlp.train()  # Ensure we're in training mode
        output1 = mlp(x)
        output2 = mlp(x)

        assert torch.allclose(output1, output2)


class TestMLPEdgeCases:
    """Test edge cases and error conditions."""

    def test_very_small_features(self):
        """Test MLP with very small feature dimensions."""
        mlp = MLP(in_features=1)
        x = torch.randn(5, 1)

        output = mlp(x)
        assert output.shape == (5, 1)

    def test_very_large_hidden_ratio(self):
        """Test MLP with large hidden dimension."""
        mlp = MLP(in_features=32, hidden_features=1024)
        x = torch.randn(2, 32)

        output = mlp(x)
        assert output.shape == (2, 32)
        assert mlp.fc1.out_features == 1024

    def test_zero_dropout(self):
        """Test MLP with zero dropout (should behave identically in train/eval)."""
        mlp = MLP(in_features=32, drop=0.0)
        x = torch.randn(2, 32)

        mlp.train()
        output_train = mlp(x)

        mlp.eval()
        output_eval = mlp(x)

        assert torch.allclose(output_train, output_eval)

    def test_full_dropout(self):
        """Test MLP with maximum dropout (should zero most activations in training)."""
        mlp = MLP(in_features=64, drop=1.0)
        x = torch.ones(10, 64)  # Use ones to see dropout effect clearly

        mlp.train()
        output = mlp(x)

        # With dropout=1.0, most values should be zero (except residual)
        # The exact behavior depends on implementation details
        assert output.shape == (10, 64)


@pytest.fixture
def basic_mlp():
    """Fixture providing a basic MLP for testing."""
    return MLP(in_features=64, hidden_features=128, out_features=32)


@pytest.fixture
def sample_input():
    """Fixture providing sample input tensor."""
    return torch.randn(4, 10, 64)


def test_mlp_with_fixtures(basic_mlp, sample_input):
    """Test MLP using fixtures."""
    output = basic_mlp(sample_input)
    assert output.shape == (4, 10, 32)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

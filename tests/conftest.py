"""Pytest configuration and fixtures for spatial_attention tests."""

import pytest
import torch


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "cuda: marks tests as requiring CUDA")
    config.addinivalue_line(
        "markers", "natten: marks tests as requiring NATTEN library"
    )


def pytest_collection_modifyitems(config, items):
    """
    Automatically mark tests based on their location/name and skip tests
    requiring unavailable dependencies.
    """
    # Check availability once
    cuda_available = torch.cuda.is_available()
    natten_available = _check_natten_availability()

    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid or "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Mark slow tests
        if (
            "slow" in item.name
            or "end_to_end" in item.name
            or "forward" in item.name
            or "workflow" in item.name
        ):
            item.add_marker(pytest.mark.slow)

        # Mark and skip CUDA-dependent tests (including SparseSpatialAttention)
        requires_cuda_or_natten = (
            "SparseSpatialAttention" in item.name
            or "sparse" in item.name.lower()
            or "natten" in item.name.lower()
            or "neighborhood" in item.name.lower()
            or any(
                "SparseSpatialAttention" in str(param)
                for param in getattr(item, "callspec", None)
                and getattr(item.callspec, "params", {}).values()
                or []
            )
        )

        if requires_cuda_or_natten:
            item.add_marker(pytest.mark.natten)
            if not natten_available:
                item.add_marker(
                    pytest.mark.skip(
                        reason=(
                            "Test requires NATTEN with CUDA support "
                            "(not available in CPU-only environment)"
                        )
                    )
                )

        # Mark explicit CUDA tests
        requires_cuda = (
            "cuda" in item.name.lower()
            or "gpu" in item.name.lower()
            or ("device" in item.name.lower() and "cpu" not in item.name.lower())
        )

        if requires_cuda:
            item.add_marker(pytest.mark.cuda)
            if not cuda_available:
                item.add_marker(pytest.mark.skip(reason="CUDA not available"))

        # Mark unit tests (everything else)
        if not any(
            mark.name in ["integration", "slow", "cuda", "natten"]
            for mark in item.iter_markers()
        ):
            item.add_marker(pytest.mark.unit)


def _check_natten_availability():
    """Check if NATTEN is available and functional."""
    try:
        # Import natten first
        import natten  # noqa: F401

        # Try to import the functional modules that SparseSpatialAttention uses
        from natten.functional import na2d_av, na2d_qk  # noqa: F401

        # Try creating small tensors to test CUDA functionality
        import torch

        if torch.cuda.is_available():
            # Test if natten functions work with actual tensors
            test_tensor = torch.randn(1, 1, 3, 3, 4)
            try:
                # This should work if natten + CUDA is properly set up
                _ = na2d_qk(test_tensor, test_tensor, (3, 3), (1, 1))
                return True
            except Exception:
                return False
        else:
            # CUDA not available, natten won't work
            return False
    except (ImportError, AssertionError, RuntimeError, Exception):
        return False


@pytest.fixture(scope="session")
def torch_device():
    """Provide torch device for tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def cuda_available():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def natten_available():
    """Check if NATTEN is available and functional."""
    return _check_natten_availability()


@pytest.fixture(scope="session")
def sparse_attention_available():
    """Check if SparseSpatialAttention can be instantiated."""
    try:
        from spatial_attention import SparseSpatialAttention

        # Try to create a minimal instance
        attn = SparseSpatialAttention(
            feature_dims=32,
            spatial_dims=2,
            kernel_size=3,
            num_heads=4,
        )
        return True
    except Exception:
        return False


@pytest.fixture
def skip_if_no_cuda():
    """Skip test if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.fixture
def skip_if_no_natten():
    """Skip test if NATTEN is not available."""
    if not _check_natten_availability():
        pytest.skip("NATTEN not available (requires CUDA support)")


@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    yield
    # Reset seed after test
    torch.seed()


@pytest.fixture
def temp_model_state():
    """Fixture for temporary model state management."""
    states = {}

    def save_state(model, name):
        states[name] = model.state_dict().copy()

    def load_state(model, name):
        if name in states:
            model.load_state_dict(states[name])

    def clear_states():
        states.clear()

    yield {"save": save_state, "load": load_state, "clear": clear_states}

    # Cleanup
    states.clear()


class MockRoSE:
    """Mock RoSE class for testing when RoSE is not available."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x, spacing, grid_shape, flatten=True):
        # Return mock tensor with same shape as input
        batch, num_heads, seq_len, dim = x.shape
        return torch.randn_like(x)


class MockNATTEN:
    """Mock NATTEN functions for testing when NATTEN is not available."""

    @staticmethod
    def na2d_qk(*args, **kwargs):
        # Mock neighborhood attention QK
        q = args[1]  # Query tensor
        return torch.randn(
            q.shape[0], q.shape[1], q.shape[2] * q.shape[3], 9
        )  # Mock neighborhood size

    @staticmethod
    def na2d_av(*args, **kwargs):
        # Mock neighborhood attention AV
        attn = args[0]
        v = args[1]
        return torch.randn_like(v)


@pytest.fixture
def mock_dependencies(monkeypatch):
    """Mock external dependencies for testing."""
    try:
        import RoSE
    except ImportError:
        # Mock RoSE if not available
        monkeypatch.setattr(
            "spatial_attention.spatial_attention.RotarySpatialEmbedding", MockRoSE
        )

    try:
        import natten
    except ImportError:
        # Mock NATTEN functions if not available
        mock_natten = MockNATTEN()
        monkeypatch.setattr(
            "spatial_attention.spatial_attention.na2d_qk", mock_natten.na2d_qk
        )
        monkeypatch.setattr(
            "spatial_attention.spatial_attention.na2d_av", mock_natten.na2d_av
        )
        monkeypatch.setattr(
            "spatial_attention.spatial_attention.na3d_qk", mock_natten.na2d_qk
        )  # Same mock
        monkeypatch.setattr(
            "spatial_attention.spatial_attention.na3d_av", mock_natten.na2d_av
        )  # Same mock
        monkeypatch.setattr(
            "spatial_attention.spatial_attention.na1d_qk", mock_natten.na2d_qk
        )  # Same mock
        monkeypatch.setattr(
            "spatial_attention.spatial_attention.na1d_av", mock_natten.na2d_av
        )  # Same mock


@pytest.fixture
def cleanup_cuda():
    """Cleanup CUDA memory after tests."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Test data fixtures
@pytest.fixture
def small_tensor_2d():
    """Small 2D tensor for quick tests."""
    return torch.randn(2, 8, 32)  # (batch, seq, features)


@pytest.fixture
def small_tensor_3d():
    """Small 3D tensor for quick tests."""
    return torch.randn(2, 12, 64)  # (batch, seq, features)


@pytest.fixture
def medium_tensor_2d():
    """Medium 2D tensor for more comprehensive tests."""
    return torch.randn(4, 64, 128)  # (batch, seq, features)


@pytest.fixture
def large_tensor_2d():
    """Large 2D tensor for performance tests."""
    return torch.randn(8, 256, 256)  # (batch, seq, features)


@pytest.fixture
def spatial_configs():
    """Common spatial attention configurations."""
    return {
        "small_2d": {
            "feature_dims": 32,
            "spatial_dims": 2,
            "kernel_size": 3,
            "num_heads": 4,
            "iters": 1,
        },
        "medium_2d": {
            "feature_dims": 64,
            "spatial_dims": 2,
            "kernel_size": 7,
            "num_heads": 8,
            "iters": 2,
        },
        "small_3d": {
            "feature_dims": 48,
            "spatial_dims": 3,
            "kernel_size": 3,
            "num_heads": 6,
            "iters": 1,
        },
    }

"""Test suite for spatial_attention."""

import pytest

from spatial_attention import SpatialAttention
from spatial_attention.core import helper_function


class TestSpatialAttention:
    """Test cases for SpatialAttention."""
    
    def test_initialization(self):
        """Test basic initialization."""
        obj = SpatialAttention(param1="test")
        assert obj.param1 == "test"
        assert obj.param2 is None
    
    def test_initialization_with_params(self):
        """Test initialization with all parameters."""
        obj = SpatialAttention(param1="test", param2=42)
        assert obj.param1 == "test"
        assert obj.param2 == 42
    
    def test_process(self):
        """Test the process method."""
        obj = SpatialAttention(param1="test")
        result = obj.process()
        assert "Processed: test" in result
    
    def test_repr(self):
        """Test string representation."""
        obj = SpatialAttention(param1="test", param2=42)
        repr_str = repr(obj)
        assert "SpatialAttention" in repr_str
        assert "test" in repr_str
        assert "42" in repr_str
    
    @pytest.mark.parametrize("param1,expected", [
        ("hello", "Processed: hello"),
        ("world", "Processed: world"),
        ("", "Processed: "),
    ])
    def test_process_parametrized(self, param1: str, expected: str):
        """Test process method with different parameters."""
        obj = SpatialAttention(param1=param1)
        result = obj.process()
        assert result == expected


class TestHelperFunction:
    """Test cases for helper functions."""
    
    def test_helper_function(self):
        """Test helper function."""
        input_data = "test_input"
        result = helper_function(input_data)
        assert result == input_data
    
    def test_helper_function_with_dict(self):
        """Test helper function with dictionary input."""
        input_data = {"key": "value"}
        result = helper_function(input_data)
        assert result == input_data
    
    def test_helper_function_with_none(self):
        """Test helper function with None input."""
        result = helper_function(None)
        assert result is None


# Integration tests
class TestIntegration:
    """Integration test cases."""
    
    def test_full_workflow(self):
        """Test complete workflow."""
        # Initialize object
        obj = SpatialAttention(param1="integration_test")
        
        # Process data
        result = obj.process()
        
        # Verify result
        assert "integration_test" in result
        
        # Use helper function
        helper_result = helper_function(result)
        assert helper_result == result


# Fixtures for common test data
@pytest.fixture
def sample_spatial_attention_object():
    """Fixture providing a sample SpatialAttention object."""
    return SpatialAttention(param1="fixture_test", param2=123)


@pytest.fixture
def sample_data():
    """Fixture providing sample test data."""
    return {
        "test_string": "hello world",
        "test_number": 42,
        "test_list": [1, 2, 3, 4, 5],
        "test_dict": {"nested": "value"}
    }


def test_with_fixtures(sample_spatial_attention_object, sample_data):
    """Test using fixtures."""
    assert sample_spatial_attention_object.param1 == "fixture_test"
    assert sample_spatial_attention_object.param2 == 123
    assert "test_string" in sample_data
    assert sample_data["test_number"] == 42

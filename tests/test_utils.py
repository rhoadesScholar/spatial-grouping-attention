"""Focused tests for utility functions."""

import pytest
from spatial_attention.utils import to_list, to_tuple


class TestToList:
    """Test cases for the to_list function."""

    def test_scalar_int(self):
        """Test to_list with integer scalar."""
        result = to_list(5, 3)
        assert result == [5, 5, 5]
        assert isinstance(result, list)
        assert all(isinstance(x, int) for x in result)

    def test_scalar_float(self):
        """Test to_list with float scalar."""
        result = to_list(2.5, 4)
        assert result == [2.5, 2.5, 2.5, 2.5]
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)

    def test_scalar_zero_ndim(self):
        """Test to_list with ndim=0 (edge case)."""
        result = to_list(42, 0)
        assert result == []

    def test_list_exact_length(self):
        """Test to_list with list of correct length."""
        input_list = [1, 2, 3]
        result = to_list(input_list, 3)
        assert result == [1, 2, 3]
        assert result is not input_list  # Should be a new list
        assert isinstance(result, list)

    def test_tuple_exact_length(self):
        """Test to_list with tuple of correct length."""
        input_tuple = (4, 5, 6)
        result = to_list(input_tuple, 3)
        assert result == [4, 5, 6]
        assert isinstance(result, list)

    def test_mixed_types_in_sequence(self):
        """Test to_list with mixed int/float types."""
        input_list = [1, 2.5, 3]
        result = to_list(input_list, 3)
        assert result == [1, 2.5, 3]
        assert isinstance(result, list)

    def test_nested_structure_valid(self):
        """Test to_list with valid nested structure."""
        input_nested = [(1, 2), (3, 4), (5, 6)]
        result = to_list(input_nested, 2)  # Each sub-element should have length 2
        expected = [[1, 2], [3, 4], [5, 6]]
        assert result == expected
        assert isinstance(result, list)
        assert all(isinstance(sublist, list) for sublist in result)

    def test_nested_lists(self):
        """Test to_list with nested lists."""
        input_nested = [[1, 2], [3, 4]]
        result = to_list(input_nested, 2)
        expected = [[1, 2], [3, 4]]
        assert result == expected

    def test_nested_tuples_and_lists_mixed(self):
        """Test to_list with mixed nested structures."""
        input_nested = [(1, 2), [3, 4]]
        result = to_list(input_nested, 2)
        # When nested structure is detected, all inner sequences are converted to lists
        expected = [[1, 2], [3, 4]]
        assert result == expected

    def test_wrong_length_raises_error(self):
        """Test to_list with wrong length raises ValueError."""
        with pytest.raises(ValueError, match="got 2 but expected 3"):
            to_list([1, 2], 3)

        with pytest.raises(ValueError, match="got 4 but expected 2"):
            to_list([1, 2, 3, 4], 2)

    def test_nested_wrong_length_raises_error(self):
        """Test to_list with nested structure of wrong length."""
        with pytest.raises(AssertionError, match="got nested structure with lengths"):
            to_list([(1, 2), (3, 4, 5)], 2)

    def test_invalid_type_raises_error(self):
        """Test to_list with invalid types raises TypeError."""
        with pytest.raises(TypeError):
            to_list("string", 3)

        with pytest.raises(TypeError):
            to_list({"dict": "value"}, 2)

        # Note: None is handled specially and creates [None, None, ...] so no error

    def test_empty_list_zero_ndim(self):
        """Test to_list with empty list and ndim=0."""
        result = to_list([], 0)
        assert result == []

    def test_single_element_list(self):
        """Test to_list with single element list."""
        result = to_list([42], 1)
        assert result == [42]

    def test_none_input(self):
        """Test to_list with None input (special case)."""
        result = to_list(None, 3)
        assert result == [None, None, None]
        assert isinstance(result, list)


class TestToTuple:
    """Test cases for the to_tuple function."""

    def test_scalar_int(self):
        """Test to_tuple with integer scalar."""
        result = to_tuple(7, 2)
        assert result == (7, 7)
        assert isinstance(result, tuple)

    def test_scalar_float(self):
        """Test to_tuple with float scalar."""
        result = to_tuple(3.14, 3)
        assert result == (3.14, 3.14, 3.14)
        assert isinstance(result, tuple)

    def test_scalar_zero_ndim(self):
        """Test to_tuple with ndim=0."""
        result = to_tuple(42, 0)
        assert result == ()

    def test_list_exact_length(self):
        """Test to_tuple with list of correct length."""
        input_list = [1, 2, 3]
        result = to_tuple(input_list, 3)
        assert result == (1, 2, 3)
        assert isinstance(result, tuple)

    def test_tuple_exact_length(self):
        """Test to_tuple with tuple of correct length."""
        input_tuple = (4, 5, 6)
        result = to_tuple(input_tuple, 3)
        assert result == (4, 5, 6)
        assert isinstance(result, tuple)

    def test_nested_structure_valid(self):
        """Test to_tuple with valid nested structure."""
        input_nested = [(1, 2), (3, 4)]
        result = to_tuple(input_nested, 2)
        expected = ((1, 2), (3, 4))
        assert result == expected
        assert isinstance(result, tuple)
        assert all(isinstance(subtuple, tuple) for subtuple in result)

    def test_nested_mixed_types(self):
        """Test to_tuple with nested mixed list/tuple types."""
        input_nested = [[1, 2], (3, 4)]
        result = to_tuple(input_nested, 2)
        # When nested structure is detected, all inner sequences are converted to tuples
        expected = ((1, 2), (3, 4))
        assert result == expected

    def test_wrong_length_raises_error(self):
        """Test to_tuple with wrong length raises ValueError."""
        with pytest.raises(ValueError):
            to_tuple([1, 2], 3)

        with pytest.raises(ValueError):
            to_tuple((1, 2, 3), 2)

    def test_nested_wrong_length_raises_error(self):
        """Test to_tuple with nested structure of wrong length."""
        # This should raise AssertionError when lengths don't match
        with pytest.raises(AssertionError, match="got nested structure with lengths"):
            to_tuple([(1, 2), (3, 4, 5)], 2)

    def test_invalid_type_raises_error(self):
        """Test to_tuple with invalid types raises TypeError."""
        with pytest.raises(TypeError):
            to_tuple("string", 3)

        with pytest.raises(TypeError):
            to_tuple({"dict": "value"}, 2)

    def test_empty_structures(self):
        """Test to_tuple with empty structures."""
        result = to_tuple([], 0)
        assert result == ()

        result = to_tuple((), 0)
        assert result == ()

    def test_none_input(self):
        """Test to_tuple with None input (special case)."""
        result = to_tuple(None, 3)
        assert result == (None, None, None)
        assert isinstance(result, tuple)


class TestParameterized:
    """Parameterized tests for both functions."""

    @pytest.mark.parametrize("value", [0, 1, 5, 10, 100])
    @pytest.mark.parametrize("ndim", [1, 2, 3, 4, 5])
    def test_scalars_both_functions(self, value, ndim):
        """Test both functions with various scalar values and dimensions."""
        list_result = to_list(value, ndim)
        tuple_result = to_tuple(value, ndim)

        assert len(list_result) == ndim
        assert len(tuple_result) == ndim
        assert all(x == value for x in list_result)
        assert all(x == value for x in tuple_result)
        assert isinstance(list_result, list)
        assert isinstance(tuple_result, tuple)

    @pytest.mark.parametrize(
        "sequence",
        [
            [1, 2, 3],
            (1, 2, 3),
            [1.0, 2.0, 3.0],
            (1.0, 2.0, 3.0),
            [1, 2.5, 3],
        ],
    )
    def test_sequences_both_functions(self, sequence):
        """Test both functions with various sequence types."""
        ndim = len(sequence)

        list_result = to_list(sequence, ndim)
        tuple_result = to_tuple(sequence, ndim)

        assert list_result == list(sequence)
        assert tuple_result == tuple(sequence)
        assert isinstance(list_result, list)
        assert isinstance(tuple_result, tuple)

    @pytest.mark.parametrize(
        "invalid_input",
        [
            "string",
            {"key": "value"},
            42.0j,  # Complex number
            object(),
        ],
    )
    def test_invalid_inputs_both_functions(self, invalid_input):
        """Test both functions with invalid input types."""
        # Note: None is handled specially by both functions, so it's not invalid
        with pytest.raises(TypeError):
            to_list(invalid_input, 2)

        with pytest.raises(TypeError):
            to_tuple(invalid_input, 2)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_large_ndim(self):
        """Test with very large ndim values."""
        result_list = to_list(1, 1000)
        result_tuple = to_tuple(1, 1000)

        assert len(result_list) == 1000
        assert len(result_tuple) == 1000
        assert all(x == 1 for x in result_list)
        assert all(x == 1 for x in result_tuple)

    def test_negative_values(self):
        """Test with negative values."""
        result_list = to_list(-5, 3)
        result_tuple = to_tuple(-5, 3)

        assert result_list == [-5, -5, -5]
        assert result_tuple == (-5, -5, -5)

    def test_float_precision(self):
        """Test with high precision floats."""
        value = 3.141592653589793
        result_list = to_list(value, 2)
        result_tuple = to_tuple(value, 2)

        assert result_list == [value, value]
        assert result_tuple == (value, value)
        assert all(x == value for x in result_list)

    def test_deeply_nested_structure(self):
        """Test with deeply nested structures."""
        # The actual implementation doesn't deeply validate nested structures
        input_nested = [[(1, 2)], [(3, 4)]]
        result = to_list(input_nested, 2)
        assert result == [[(1, 2)], [(3, 4)]]  # Returns as-is

    def test_empty_nested_structure(self):
        """Test with empty nested structure."""
        # Empty outer list with ndim=0 should work
        result_list = to_list([], 0)
        result_tuple = to_tuple([], 0)

        assert result_list == []
        assert result_tuple == ()


# Fixtures for common test data
@pytest.fixture
def sample_2d_data():
    """Sample data for 2D operations."""
    return {
        "scalar": 5,
        "list": [1, 2],
        "tuple": (3, 4),
        "nested": [(1, 2), (3, 4)],
        "ndim": 2,
    }


@pytest.fixture
def sample_3d_data():
    """Sample data for 3D operations."""
    return {
        "scalar": 7.5,
        "list": [1, 2, 3],
        "tuple": (4, 5, 6),
        "nested": [(1, 2, 3), (4, 5, 6)],
        "ndim": 3,
    }


def test_with_2d_fixture(sample_2d_data):
    """Test using 2D fixture data."""
    data = sample_2d_data

    # Test scalar
    assert to_list(data["scalar"], data["ndim"]) == [5, 5]
    assert to_tuple(data["scalar"], data["ndim"]) == (5, 5)

    # Test sequences
    assert to_list(data["list"], data["ndim"]) == [1, 2]
    assert to_tuple(data["tuple"], data["ndim"]) == (3, 4)


def test_with_3d_fixture(sample_3d_data):
    """Test using 3D fixture data."""
    data = sample_3d_data

    # Test scalar
    assert to_list(data["scalar"], data["ndim"]) == [7.5, 7.5, 7.5]
    assert to_tuple(data["scalar"], data["ndim"]) == (7.5, 7.5, 7.5)

    # Test sequences
    assert to_list(data["list"], data["ndim"]) == [1, 2, 3]
    assert to_tuple(data["tuple"], data["ndim"]) == (4, 5, 6)

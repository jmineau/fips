"""Test suite for fips.indices module."""

import pytest
import pandas as pd
import numpy as np
import warnings

from fips.indices import check_overlap, sanitize_index


class TestCheckOverlap:
    """Tests for check_overlap function."""

    def test_check_overlap_perfect_match(self):
        """Test when indices match perfectly."""
        idx1 = pd.Index(['a', 'b', 'c'])
        idx2 = pd.Index(['a', 'b', 'c'])
        
        # Should not warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_overlap(idx1, idx2, "Test")
            assert len(w) == 0

    def test_check_overlap_partial(self):
        """Test when indices partially overlap."""
        idx1 = pd.Index(['a', 'b', 'c', 'd'])
        idx2 = pd.Index(['a', 'b', 'x', 'y'])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_overlap(idx1, idx2, "Test")
            assert len(w) == 1
            assert "Partial overlap" in str(w[0].message)

    def test_check_overlap_no_overlap(self):
        """Test when indices have no overlap."""
        idx1 = pd.Index(['a', 'b', 'c'])
        idx2 = pd.Index(['x', 'y', 'z'])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_overlap(idx1, idx2, "Test")
            assert len(w) == 1
            assert "No overlap" in str(w[0].message)

    def test_check_overlap_subset(self):
        """Test when one index is a subset of another."""
        idx1 = pd.Index(['a', 'b'])  # target
        idx2 = pd.Index(['a', 'b', 'c', 'd'])  # available
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_overlap(idx1, idx2, "Test")
            assert len(w) == 0

    def test_check_overlap_single_element(self):
        """Test overlap with single element."""
        idx1 = pd.Index(['a', 'b', 'c'])
        idx2 = pd.Index(['b', 'x', 'y'])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_overlap(idx1, idx2, "Test")
            assert len(w) == 1
            assert "Partial overlap" in str(w[0].message)

    def test_check_overlap_with_numeric_index(self):
        """Test check_overlap with numeric indices."""
        idx1 = pd.Index([1, 2, 3])
        idx2 = pd.Index([1, 2, 4])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_overlap(idx1, idx2, "Test")
            assert len(w) == 1

    def test_check_overlap_includes_name_in_warning(self):
        """Test that warning includes the provided name."""
        idx1 = pd.Index(['a', 'b', 'c'])
        idx2 = pd.Index(['x', 'y', 'z'])
        name = "MyDimension"
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_overlap(idx1, idx2, name)
            assert name in str(w[0].message)


class TestSanitizeIndex:
    """Tests for sanitize_index function."""

    def test_sanitize_index_string_to_numeric(self):
        """Test converting string index to numeric."""
        idx = pd.Index(['1', '2', '3'])
        result = sanitize_index(idx)
        
        assert pd.api.types.is_numeric_dtype(result)

    def test_sanitize_index_already_numeric(self):
        """Test index that's already numeric."""
        idx = pd.Index([1, 2, 3])
        result = sanitize_index(idx)
        
        assert pd.api.types.is_numeric_dtype(result)

    def test_sanitize_index_float_precision(self):
        """Test rounding to specified decimal places."""
        idx = pd.Index([1.123456, 2.654321, 3.141592])
        result = sanitize_index(idx, decimals=2)
        
        expected = pd.Index([1.12, 2.65, 3.14])
        assert np.allclose(result, expected)

    def test_sanitize_index_preserves_order(self):
        """Test that order is preserved."""
        idx = pd.Index([3.456, 1.234, 2.345])
        result = sanitize_index(idx, decimals=1)
        
        assert result.tolist() == [3.5, 1.2, 2.3] or np.allclose(result, [3.5, 1.2, 2.3])

    def test_sanitize_index_text_not_convertible(self):
        """Test that non-convertible text stays as text."""
        idx = pd.Index(['alpha', 'beta', 'gamma'])
        result = sanitize_index(idx)
        
        # Should remain as text since conversion fails
        assert isinstance(result, pd.Index)

    def test_sanitize_index_multiindex(self):
        """Test sanitizing MultiIndex."""
        idx = pd.MultiIndex.from_product([['1', '2'], ['a', 'b']], 
                                         names=['nums', 'letters'])
        result = sanitize_index(idx)
        
        assert isinstance(result, pd.MultiIndex)
        assert result.nlevels == 2

    def test_sanitize_index_multiindex_numeric_conversion(self):
        """Test MultiIndex level conversion."""
        idx = pd.MultiIndex.from_product([['1.0', '2.0'], ['3.0', '4.0']])
        result = sanitize_index(idx)
        
        assert isinstance(result, pd.MultiIndex)
        # First level should be numeric
        level0 = result.get_level_values(0)
        assert pd.api.types.is_numeric_dtype(level0)

    def test_sanitize_index_preserves_name(self):
        """Test that index name is preserved."""
        idx = pd.Index(['1', '2', '3'], name='my_index')
        result = sanitize_index(idx)
        
        assert result.name == 'my_index'

    def test_sanitize_index_with_decimals_none(self):
        """Test that decimals=None doesn't round."""
        idx = pd.Index([1.123456, 2.654321])
        result = sanitize_index(idx, decimals=None)
        
        # Values should be unchanged
        assert np.allclose(result, idx)

    def test_sanitize_index_mixed_types(self):
        """Test index with mixed numeric types."""
        idx = pd.Index([1, 2.5, 3])
        result = sanitize_index(idx)
        
        assert pd.api.types.is_numeric_dtype(result)

    def test_sanitize_index_with_nan(self):
        """Test index containing NaN values."""
        idx = pd.Index([1.0, np.nan, 3.0])
        result = sanitize_index(idx, decimals=1)
        
        assert len(result) == 3
        assert np.isnan(result[1])

    def test_sanitize_multiindex_nested_conversion(self):
        """Test deep MultiIndex conversion."""
        level0 = pd.Index(['1', '2'], name='l0')
        level1 = pd.Index(['3', '4'], name='l1')
        idx = pd.MultiIndex.from_product([level0, level1])
        result = sanitize_index(idx, decimals=2)
        
        assert isinstance(result, pd.MultiIndex)
        assert result.nlevels == 2


class TestSanitizeIndexEdgeCases:
    """Edge case tests for sanitize_index."""

    def test_empty_index(self):
        """Test sanitizing an empty index."""
        idx = pd.Index([])
        result = sanitize_index(idx)
        
        assert len(result) == 0

    def test_single_element_index(self):
        """Test single element index."""
        idx = pd.Index(['1.5'])
        result = sanitize_index(idx, decimals=1)
        
        assert len(result) == 1

    def test_large_index(self):
        """Test sanitizing a large index."""
        idx = pd.Index([f'{i}.123456' for i in range(1000)])
        result = sanitize_index(idx, decimals=2)
        
        assert len(result) == 1000

    def test_negative_numbers(self):
        """Test with negative numbers."""
        idx = pd.Index(['-1.123', '-2.456', '3.789'])
        result = sanitize_index(idx, decimals=2)
        
        assert result[0] < 0
        assert result[1] < 0
        assert result[2] > 0

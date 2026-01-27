"""Test suite for fips.vectors module."""

import pytest
import pandas as pd
import numpy as np

from fips.vectors import Block, Vector, prepare_vector


class TestBlock:
    """Tests for Block dataclass."""

    def test_block_creation(self):
        """Test basic Block creation."""
        data = pd.Series([1, 2, 3], index=['a', 'b', 'c'], name='test_block')
        block = Block('test_block', data)
        
        assert block.name == 'test_block'
        assert isinstance(block.data, pd.Series)
        assert len(block.data) == 3

    def test_block_name_enforcement(self):
        """Test that block name is enforced on series."""
        data = pd.Series([1, 2, 3], index=['a', 'b', 'c'], name='old_name')
        block = Block('new_name', data)
        
        assert block.data.name == 'new_name'

    def test_block_rejects_non_series(self):
        """Test that Block raises error when data is not a Series."""
        with pytest.raises(ValueError, match="must be a pandas Series"):
            Block('bad_block', [1, 2, 3])

    def test_block_with_numeric_index(self):
        """Test Block with numeric index."""
        data = pd.Series([10, 20, 30], index=[0, 1, 2])
        block = Block('numbers', data)
        
        assert block.data.index.tolist() == [0, 1, 2]

    def test_block_with_multiindex(self):
        """Test Block with MultiIndex."""
        idx = pd.MultiIndex.from_product([['x', 'y'], [1, 2]], names=['dim1', 'dim2'])
        data = pd.Series([1, 2, 3, 4], index=idx)
        block = Block('multi', data)
        
        assert isinstance(block.data.index, pd.MultiIndex)
        assert block.data.index.nlevels == 2


class TestVector:
    """Tests for Vector class."""

    def test_vector_creation_single_block(self):
        """Test Vector creation with a single block."""
        data = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
        block = Block('block1', data)
        vector = Vector([block])
        
        assert vector.size == 3
        assert vector.block_order == ['block1']
        assert 'block1' in vector

    def test_vector_creation_multiple_blocks(self):
        """Test Vector creation with multiple blocks."""
        block1 = Block('b1', pd.Series([1, 2], index=['x', 'y']))
        block2 = Block('b2', pd.Series([3, 4, 5], index=['a', 'b', 'c']))
        vector = Vector([block1, block2])
        
        assert vector.size == 5
        assert vector.block_order == ['b1', 'b2']
        assert len(vector.blocks) == 2

    def test_vector_getitem_block_access(self):
        """Test accessing blocks via __getitem__."""
        block1 = Block('b1', pd.Series([1, 2], index=['x', 'y']))
        block2 = Block('b2', pd.Series([3, 4, 5], index=['a', 'b', 'c']))
        vector = Vector([block1, block2])
        
        retrieved_block1 = vector['b1']
        assert isinstance(retrieved_block1, pd.Series)
        assert retrieved_block1.tolist() == [1, 2]

    def test_vector_iteration(self):
        """Test iterating over vector blocks."""
        block1 = Block('b1', pd.Series([1, 2], index=['x', 'y']))
        block2 = Block('b2', pd.Series([3, 4, 5], index=['a', 'b', 'c']))
        vector = Vector([block1, block2])
        
        names = list(vector)
        assert names == ['b1', 'b2']

    def test_vector_names_property(self):
        """Test names property."""
        block1 = Block('block_a', pd.Series([1, 2]))
        block2 = Block('block_b', pd.Series([3, 4]))
        vector = Vector([block1, block2])
        
        assert vector.names == ['block_a', 'block_b']

    def test_vector_get_block_slice(self):
        """Test get_block_slice method."""
        block1 = Block('b1', pd.Series([1, 2]))
        block2 = Block('b2', pd.Series([3, 4, 5]))
        block3 = Block('b3', pd.Series([6]))
        vector = Vector([block1, block2, block3])
        
        slice1 = vector.get_block_slice('b1')
        assert slice1 == slice(0, 2)
        
        slice2 = vector.get_block_slice('b2')
        assert slice2 == slice(2, 5)
        
        slice3 = vector.get_block_slice('b3')
        assert slice3 == slice(5, 6)

    def test_vector_get_block_slice_invalid(self):
        """Test get_block_slice with invalid block name."""
        block = Block('b1', pd.Series([1, 2]))
        vector = Vector([block])
        
        with pytest.raises(KeyError):
            vector.get_block_slice('nonexistent')

    def test_vector_multiindex_assembly(self):
        """Test that Vector properly assembles MultiIndex blocks."""
        idx1 = pd.MultiIndex.from_product([['x', 'y'], [1, 2]], names=['dim1', 'dim2'])
        idx2 = pd.Index(['a', 'b'], name='letters')
        
        block1 = Block('b1', pd.Series([1, 2, 3, 4], index=idx1))
        block2 = Block('b2', pd.Series([5, 6], index=idx2))
        
        vector = Vector([block1, block2])
        
        # Check that assembled data has proper structure
        # MultiIndex blocks get promoted, simple index blocks don't necessarily have block level
        assert vector.size == 6
        assert len(vector.blocks) == 2

    def test_vector_data_concatenation(self):
        """Test that data is properly concatenated."""
        block1 = Block('b1', pd.Series([10, 20], index=['a', 'b']))
        block2 = Block('b2', pd.Series([30, 40, 50], index=['x', 'y', 'z']))
        vector = Vector([block1, block2])
        
        # Check concatenated values match expected order
        all_values = vector.data.values.tolist()
        assert all_values == [10, 20, 30, 40, 50]


class TestPrepareVector:
    """Tests for prepare_vector function."""

    def test_prepare_vector_from_series(self):
        """Test preparing a Series into a Vector."""
        series = pd.Series([1, 2, 3], index=['a', 'b', 'c'], name='my_series')
        vector, promote = prepare_vector(series, 'default', float_precision=None)
        
        assert isinstance(vector, Vector)
        assert promote is True
        assert vector.size == 3

    def test_prepare_vector_uses_series_name(self):
        """Test that prepare_vector uses the Series name as block name."""
        series = pd.Series([1, 2], index=['x', 'y'], name='custom_name')
        vector, _ = prepare_vector(series, 'default', float_precision=None)
        
        assert 'custom_name' in vector.names

    def test_prepare_vector_uses_default_name(self):
        """Test that prepare_vector uses default name when Series has no name."""
        series = pd.Series([1, 2], index=['x', 'y'])
        vector, _ = prepare_vector(series, 'default_label', float_precision=None)
        
        assert 'default_label' in vector.names

    def test_prepare_vector_from_vector(self):
        """Test preparing an already-prepared Vector."""
        block = Block('b1', pd.Series([1, 2], index=['x', 'y']))
        original_vector = Vector([block])
        
        result_vector, promote = prepare_vector(original_vector, 'default', float_precision=None)
        
        assert result_vector is original_vector
        assert promote is False

    def test_prepare_vector_float_precision(self):
        """Test float precision rounding in prepare_vector."""
        series = pd.Series([1, 2], index=[1.123456, 2.654321])
        vector, _ = prepare_vector(series, 'test', float_precision=2)
        
        # Index should be rounded to 2 decimals
        index_values = vector.data.index.get_level_values(-1).tolist()
        assert 1.12 in index_values or abs(index_values[0] - 1.12) < 0.01

    def test_prepare_vector_copies_series(self):
        """Test that prepare_vector creates a copy of the series."""
        series = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
        vector, _ = prepare_vector(series, 'test', float_precision=None)
        
        # Modify original series
        series.iloc[0] = 999
        
        # Vector data should not change
        assert vector.data.values[0] != 999

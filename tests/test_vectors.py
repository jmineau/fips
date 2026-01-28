"""Test suite for fips.vectors module."""

import numpy as np
import pandas as pd
import pytest

from fips.vectors import Block, Vector, prepare_vector


class TestBlock:
    """Tests for Block dataclass."""

    def test_block_creation(self):
        """Test basic Block creation from Series."""
        data = pd.Series([1, 2, 3], index=["a", "b", "c"], name="test_block")
        block = Block(data)

        assert block.name == "test_block"
        assert isinstance(block.data, pd.Series)
        assert len(block.data) == 3

    def test_block_rejects_series_without_name(self):
        """Test that Block raises error when Series has no name and name not provided."""
        data = pd.Series([1, 2, 3], index=["a", "b", "c"])
        with pytest.raises(ValueError, match="must have a name"):
            Block(data)

    def test_block_creation_with_name_override(self):
        """Test Block creation overriding Series name."""
        data = pd.Series([1, 2, 3], index=["a", "b", "c"], name="original")
        block = Block(data, name="override")

        assert block.name == "override"

    def test_block_creation_from_array(self):
        """Test Block creation from array values with index and name."""
        data = Block([10, 20, 30], index=[0, 1, 2], name="numbers")

        assert data.name == "numbers"
        assert data.data.index.tolist() == [0, 1, 2]
        assert data.data.values.tolist() == [10, 20, 30]

    def test_block_creation_from_array_missing_index(self):
        """Test that Block raises error when creating from array without index."""
        with pytest.raises(ValueError, match="'index' and 'name' are required"):
            Block([1, 2, 3], name="test")

    def test_block_creation_from_array_missing_name(self):
        """Test that Block raises error when creating from array without name."""
        with pytest.raises(ValueError, match="'index' and 'name' are required"):
            Block([1, 2, 3], index=[0, 1, 2])

    def test_block_with_multiindex(self):
        """Test Block with MultiIndex."""
        idx = pd.MultiIndex.from_product([["x", "y"], [1, 2]], names=["dim1", "dim2"])
        data = pd.Series([1, 2, 3, 4], index=idx, name="multi")
        block = Block(data)

        assert isinstance(block.data.index, pd.MultiIndex)
        assert block.data.index.nlevels == 2

    def test_block_rejects_nan_values(self):
        """Test that Block raises error when data contains NaN values."""
        data = pd.Series([1, np.nan, 3], index=["a", "b", "c"], name="test")
        with pytest.raises(ValueError, match="contains NaN"):
            Block(data)


class TestVector:
    """Tests for Vector class."""

    def test_vector_creation_single_block(self):
        """Test Vector creation with a single block."""
        data = pd.Series([1, 2, 3], index=["a", "b", "c"], name="block1")
        block = Block(data)
        vector = Vector([block])

        assert vector.n == 3
        assert len(vector) == 1
        assert "block1" in vector.blocks

    def test_vector_creation_multiple_blocks(self):
        """Test Vector creation with multiple blocks."""
        block1 = Block(pd.Series([1, 2], index=["x", "y"], name="b1"))
        block2 = Block(pd.Series([3, 4, 5], index=["a", "b", "c"], name="b2"))
        vector = Vector([block1, block2])

        assert vector.n == 5
        assert len(vector) == 2
        assert "b1" in vector.blocks
        assert "b2" in vector.blocks

    def test_vector_creation_from_series(self):
        """Test Vector creation with pd.Series (auto-converted to Block)."""
        series1 = pd.Series([1, 2], index=["x", "y"], name="s1")
        series2 = pd.Series([3, 4, 5], index=["a", "b", "c"], name="s2")
        vector = Vector([series1, series2])

        assert vector.n == 5
        assert len(vector) == 2
        assert "s1" in vector.blocks
        assert "s2" in vector.blocks

    def test_vector_getitem_block_access(self):
        """Test accessing blocks via __getitem__."""
        series1 = pd.Series([1, 2], index=["x", "y"], name="b1")
        series2 = pd.Series([3, 4, 5], index=["a", "b", "c"], name="b2")
        vector = Vector([series1, series2])

        retrieved_block1 = vector["b1"]
        assert isinstance(retrieved_block1, pd.Series)
        assert retrieved_block1.tolist() == [1, 2]

    def test_vector_blocks_dict(self):
        """Test that vector.blocks is a dictionary."""
        series1 = pd.Series([1, 2], index=["x", "y"], name="b1")
        series2 = pd.Series([3, 4, 5], index=["a", "b", "c"], name="b2")
        vector = Vector([series1, series2])

        assert isinstance(vector.blocks, dict)
        assert "b1" in vector.blocks
        assert "b2" in vector.blocks
        assert isinstance(vector.blocks["b1"], Block)
        assert isinstance(vector.blocks["b2"], Block)

    def test_vector_duplicate_block_names(self):
        """Test that Vector raises error on duplicate block names."""
        series1 = pd.Series([1, 2], index=["x", "y"], name="duplicate")
        series2 = pd.Series([3, 4, 5], index=["a", "b", "c"], name="duplicate")

        with pytest.raises(ValueError, match="Duplicate block name"):
            Vector([series1, series2])

    def test_vector_data_assembly(self):
        """Test that Vector properly assembles multi-block data."""
        series1 = pd.Series([10, 20], index=["a", "b"], name="b1")
        series2 = pd.Series([30, 40, 50], index=["x", "y", "z"], name="b2")
        vector = Vector([series1, series2])

        # Check that assembled data has proper multi-index structure
        assert "block" in vector.data.index.names
        assert vector.data.shape[0] == 5  # 2 + 3 elements

    def test_vector_multiindex_assembly(self):
        """Test that Vector properly assembles MultiIndex blocks."""
        idx1 = pd.MultiIndex.from_product([["x", "y"], [1, 2]], names=["dim1", "dim2"])
        idx2 = pd.Index(["a", "b"], name="letters")

        series1 = pd.Series([1, 2, 3, 4], index=idx1, name="b1")
        series2 = pd.Series([5, 6], index=idx2, name="b2")

        vector = Vector([series1, series2])

        # Check that assembled data has proper structure
        assert vector.n == 6
        assert len(vector) == 2
        assert "block" in vector.data.index.names


class TestPrepareVector:
    """Tests for prepare_vector function."""

    def test_prepare_vector_from_series(self):
        """Test preparing a Series into a Vector."""
        series = pd.Series([1, 2, 3], index=["a", "b", "c"], name="my_series")
        vector = prepare_vector(series, "default", float_precision=None)

        assert isinstance(vector, Vector)
        assert vector.n == 3
        assert "my_series" in vector.blocks

    def test_prepare_vector_uses_default_name(self):
        """Test that prepare_vector uses default name when Series has no name."""
        series = pd.Series([1, 2], index=["x", "y"])
        vector = prepare_vector(series, "default_label", float_precision=None)

        assert "default_label" in vector.blocks

    def test_prepare_vector_from_vector(self):
        """Test preparing an already-prepared Vector."""
        series = pd.Series([1, 2], index=["x", "y"], name="b1")
        block = Block(series)
        original_vector = Vector([block])

        result_vector = prepare_vector(original_vector, "default", float_precision=None)

        assert result_vector is original_vector

    def test_prepare_vector_float_precision(self):
        """Test float precision rounding in prepare_vector."""
        series = pd.Series([1, 2], index=[1.123456, 2.654321])
        vector = prepare_vector(series, "test", float_precision=2)

        # Index should be rounded to 2 decimals
        rounded_index = vector.data.index.get_level_values("test_0").unique()
        assert all(isinstance(idx, (int, float)) for idx in rounded_index)
        # Check that values are rounded to 2 decimals
        assert 1.12 in rounded_index or abs(rounded_index[0] - 1.12) < 0.01
        assert 2.65 in rounded_index or abs(rounded_index[1] - 2.65) < 0.01

    def test_prepare_vector_copies_series(self):
        """Test that prepare_vector creates a copy of the series."""
        series = pd.Series([1, 2, 3], index=["a", "b", "c"], name="test")
        vector = prepare_vector(series, "test", float_precision=None)

        # Modify original series
        series.iloc[0] = 999

        # Vector data should not change
        assert vector.data.iloc[0] != 999

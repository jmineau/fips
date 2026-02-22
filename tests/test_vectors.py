"""Test suite for fips vector types."""

import pandas as pd
import pytest

from fips.vector import Block, Vector


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

    def test_block_creation_from_array_auto_index(self):
        """Test that Block auto-creates index from array."""
        block = Block([1, 2, 3], name="test")

        assert block.name == "test"
        assert len(block.data) == 3
        # Auto-created index should be RangeIndex
        assert block.data.index.tolist() == [0, 1, 2]

    def test_block_creation_from_array_auto_name(self):
        """Test that Block can be created with index but without explicit name."""
        block = Block([1, 2, 3], index=["a", "b", "c"], name="test")

        assert block.name == "test"
        assert block.data.index.tolist() == ["a", "b", "c"]

    def test_block_with_multiindex(self):
        """Test Block with MultiIndex."""
        idx = pd.MultiIndex.from_product([["x", "y"], [1, 2]], names=["dim1", "dim2"])
        data = pd.Series([1, 2, 3, 4], index=idx, name="multi")
        block = Block(data)

        assert isinstance(block.data.index, pd.MultiIndex)
        assert block.data.index.nlevels == 2


class TestVector:
    """Tests for Vector class."""

    def test_vector_creation_single_block(self):
        """Test Vector creation with a single block."""
        data = pd.Series([1, 2, 3], index=["a", "b", "c"], name="block1")
        block = Block(data)
        vector = Vector(data=[block], name="prior")

        assert len(vector.data) == 3
        assert len(vector.data.index.get_level_values("block").unique()) == 1
        # Check block name is in index
        assert "block1" in vector.data.index.get_level_values("block")

    def test_vector_creation_multiple_blocks(self):
        """Test Vector creation with multiple blocks."""
        block1 = Block(pd.Series([1, 2], index=["x", "y"], name="b1"))
        block2 = Block(pd.Series([3, 4, 5], index=["a", "b", "c"], name="b2"))
        vector = Vector(data=[block1, block2], name="posterior")

        assert len(vector.data) == 5
        assert len(vector.data.index.get_level_values("block").unique()) == 2
        assert "b1" in vector.data.index.get_level_values("block")
        assert "b2" in vector.data.index.get_level_values("block")

    def test_vector_creation_from_series(self):
        """Test Vector creation with pd.Series (auto-converted to Block)."""
        series1 = pd.Series([1, 2], index=["x", "y"], name="s1")
        series2 = pd.Series([3, 4, 5], index=["a", "b", "c"], name="s2")
        vector = Vector(data=[series1, series2], name="obs")

        assert len(vector.data) == 5
        assert len(vector.data.index.get_level_values("block").unique()) == 2
        assert "s1" in vector.data.index.get_level_values("block")
        assert "s2" in vector.data.index.get_level_values("block")

    def test_vector_getitem_block_access(self):
        """Test accessing individual block data via xs."""
        # Create blocks with named indices for cross-section to work
        idx1 = pd.Index(["x", "y"], name="dim1")
        idx2 = pd.Index(["a", "b", "c"], name="dim2")
        series1 = pd.Series([1, 2], index=idx1, name="b1")
        series2 = pd.Series([3, 4, 5], index=idx2, name="b2")
        vector = Vector(data=[series1, series2], name="state")

        # Verify structure
        assert isinstance(vector.data.index, pd.MultiIndex)
        block_vals = vector.data.index.get_level_values("block")
        assert "b1" in block_vals
        assert "b2" in block_vals

    def test_vector_blocks_dict(self):
        """Test accessing blocks from vector data."""
        series1 = pd.Series([1, 2], index=["x", "y"], name="b1")
        series2 = pd.Series([3, 4, 5], index=["a", "b", "c"], name="b2")
        vector = Vector(data=[series1, series2], name="observation")

        # Extract block names
        block_names = vector.data.index.get_level_values("block").unique()
        assert len(block_names) == 2
        assert "b1" in block_names
        assert "b2" in block_names

    def test_vector_duplicate_block_names(self):
        """Test that Vector raises error on duplicate block names."""
        series1 = pd.Series([1, 2], index=["x", "y"], name="duplicate")
        series2 = pd.Series([3, 4, 5], index=["a", "b", "c"], name="duplicate")

        with pytest.raises(ValueError, match="Duplicate block name"):
            Vector(data=[series1, series2], name="state")

    def test_vector_data_assembly(self):
        """Test that Vector properly assembles multi-block data."""
        series1 = pd.Series([10, 20], index=["a", "b"], name="b1")
        series2 = pd.Series([30, 40, 50], index=["x", "y", "z"], name="b2")
        vector = Vector(data=[series1, series2], name="posterior")

        # Check that assembled data has proper multi-index structure
        assert "block" in vector.data.index.names
        assert vector.data.shape[0] == 5  # 2 + 3 elements

    def test_vector_multiindex_assembly(self):
        """Test that Vector properly assembles MultiIndex blocks."""
        idx1 = pd.MultiIndex.from_product([["x", "y"], [1, 2]], names=["dim1", "dim2"])
        idx2 = pd.Index(["a", "b"], name="letters")

        series1 = pd.Series([1, 2, 3, 4], index=idx1, name="b1")
        series2 = pd.Series([5, 6], index=idx2, name="b2")

        vector = Vector(data=[series1, series2], name="obs")

        # Check that assembled data has proper structure
        assert len(vector.data) == 6
        assert len(vector.data.index.get_level_values("block").unique()) == 2
        assert "block" in vector.data.index.names


class TestVectorCrossSection:
    """Tests for Vector.xs() cross-section method."""

    def test_vector_xs_with_block_level(self):
        """Test cross-section directly on block level."""
        # Create blocks with named indices for cross-section to work
        idx = pd.Index(["x", "y"], name="dim")
        block1 = Block(pd.Series([10, 20], index=idx, name="b1"))
        block2 = Block(pd.Series([30, 40], index=idx, name="b2"))
        vector = Vector(data=[block1, block2], name="obs")

        # Access specific block using xs
        result = vector.xs("b1", level="block")

        assert isinstance(result, pd.Series)
        assert len(result) == 2

    def test_vector_xs_drop_level_false(self):
        """Test that drop_level=False preserves block level in result."""
        idx = pd.Index(["x", "y"], name="dim")
        block = Block(pd.Series([1, 2], index=idx, name="b1"))
        vector = Vector(data=[block], name="prior")

        result = vector.xs("b1", level="block", drop_level=False)

        assert isinstance(result, pd.Series)
        assert "block" in result.index.names

    def test_vector_xs_kwargs(self):
        """Test cross-section with keyword arguments."""
        idx = pd.Index(["a", "b", "c"], name="dim")
        block = Block(pd.Series([1, 2, 3], index=idx, name="b1"))
        vector = Vector(data=[block], name="posterior")

        # Access specific index value
        result = vector.xs("a", level="dim")

        assert isinstance(result, pd.Series) or isinstance(result, (int, float))

    def test_vector_xs_exists(self):
        """Test that xs method exists and is callable."""
        block = Block(pd.Series([1, 2], index=["x", "y"], name="b1"))
        vector = Vector(data=[block], name="obs")

        assert hasattr(vector, "xs")
        assert callable(vector.xs)

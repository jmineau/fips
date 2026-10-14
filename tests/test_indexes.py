"""Tests for index validation and manipulation utilities."""

import pandas as pd

from fips.indexes import apply_to_index, overlaps, round_index


def test_apply_to_index():
    """Test apply_to_index decorator for applying functions to each index level."""

    @apply_to_index
    def add_one(idx):
        return idx + 1

    # Test single index
    idx = pd.Index([1, 2, 3], name="a")
    res = add_one(idx)
    pd.testing.assert_index_equal(res, pd.Index([2, 3, 4], name="a"))

    # Test MultiIndex
    midx = pd.MultiIndex.from_arrays([[1, 2], [3, 4]], names=["a", "b"])
    res = add_one(midx)
    expected = pd.MultiIndex.from_arrays([[2, 3], [4, 5]], names=["a", "b"])
    pd.testing.assert_index_equal(res, expected)


def test_overlaps():
    """Test checking overlap between two indices."""
    idx1 = pd.Index([1, 2, 3])
    idx2 = pd.Index([2, 3, 4])
    idx3 = pd.Index([4, 5, 6])
    idx4 = pd.Index([1, 2])

    assert overlaps(idx4, idx1) is True  # target is fully in available
    assert overlaps(idx1, idx2) == "partial"  # target is partially in available
    assert overlaps(idx1, idx3) is False  # target is not in available


def test_round_index():
    """Test rounding numeric values in an index."""
    # Float index
    idx = pd.Index([1.123, 2.456], name="a")
    res = round_index(idx, 1)
    pd.testing.assert_index_equal(res, pd.Index([1.1, 2.5], name="a"))

    # Non-float index
    idx_str = pd.Index(["a", "b"], name="str")
    res_str = round_index(idx_str, 1)
    pd.testing.assert_index_equal(res_str, idx_str)

    # MultiIndex
    midx = pd.MultiIndex.from_arrays([[1.123, 2.456], ["a", "b"]], names=["f", "s"])
    res_midx = round_index(midx, 1)
    expected = pd.MultiIndex.from_arrays([[1.1, 2.5], ["a", "b"]], names=["f", "s"])
    pd.testing.assert_index_equal(res_midx, expected)

import numpy as np
import pandas as pd
import pytest

from fips.indexes import (
    apply_to_index,
    assign_block,
    outer_align_levels,
    overlaps,
    resolve_axes,
    round_index,
    to_numeric,
)


def test_apply_to_index():
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


def test_assign_block():
    # Single index
    idx = pd.Index([1, 2, 3], name="a")
    res = assign_block(idx, "block1")
    expected = pd.MultiIndex.from_arrays(
        [["block1"] * 3, [1, 2, 3]], names=["block", "a"]
    )
    pd.testing.assert_index_equal(res, expected)

    # MultiIndex without block
    midx = pd.MultiIndex.from_arrays([[1, 2], [3, 4]], names=["a", "b"])
    res = assign_block(midx, "block2")
    expected = pd.MultiIndex.from_arrays(
        [["block2"] * 2, [1, 2], [3, 4]], names=["block", "a", "b"]
    )
    pd.testing.assert_index_equal(res, expected)

    # MultiIndex with existing block
    midx_block = pd.MultiIndex.from_arrays([["old"] * 2, [1, 2]], names=["block", "a"])
    res = assign_block(midx_block, "new")
    expected = pd.MultiIndex.from_arrays([["new"] * 2, [1, 2]], names=["block", "a"])
    pd.testing.assert_index_equal(res, expected)


def test_outer_align_levels():
    df1 = pd.DataFrame({"val": [1, 2]}, index=pd.Index([1, 2], name="a"))
    df2 = pd.DataFrame({"val": [3, 4]}, index=pd.Index([3, 4], name="b"))

    aligned = outer_align_levels([df1, df2], axis=0)

    assert len(aligned) == 2

    # df1 should have levels ['a', 'b']
    expected_idx1 = pd.MultiIndex.from_arrays(
        [[1, 2], [np.nan, np.nan]], names=["a", "b"]
    )
    pd.testing.assert_index_equal(aligned[0].index, expected_idx1)

    # df2 should have levels ['a', 'b']
    expected_idx2 = pd.MultiIndex.from_arrays(
        [[np.nan, np.nan], [3, 4]], names=["a", "b"]
    )
    pd.testing.assert_index_equal(aligned[1].index, expected_idx2)


def test_overlaps():
    idx1 = pd.Index([1, 2, 3])
    idx2 = pd.Index([2, 3, 4])
    idx3 = pd.Index([4, 5, 6])
    idx4 = pd.Index([1, 2])

    assert overlaps(idx4, idx1) is True  # target is fully in available
    assert overlaps(idx1, idx2) == "partial"  # target is partially in available
    assert overlaps(idx1, idx3) is False  # target is not in available


def test_resolve_axes():
    assert resolve_axes(0) == (0,)
    assert resolve_axes("index") == (0,)
    assert resolve_axes(1) == (1,)
    assert resolve_axes("columns") == (1,)
    assert resolve_axes("both") == (0, 1)

    with pytest.raises(ValueError):
        resolve_axes("invalid")


def test_round_index():
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


def test_to_numeric():
    # String index that can be numeric
    idx = pd.Index(["1", "2", "3"], name="a")
    res = to_numeric(idx)
    pd.testing.assert_index_equal(res, pd.Index([1, 2, 3], name="a"))

    # String index that cannot be numeric
    idx_str = pd.Index(["a", "b", "c"], name="a")
    res_str = to_numeric(idx_str)
    pd.testing.assert_index_equal(res_str, idx_str)

    # MultiIndex
    midx = pd.MultiIndex.from_arrays([["1", "2"], ["a", "b"]], names=["n", "s"])
    res_midx = to_numeric(midx)
    expected = pd.MultiIndex.from_arrays([[1, 2], ["a", "b"]], names=["n", "s"])
    pd.testing.assert_index_equal(res_midx, expected)


def test_to_numeric_datetime():
    # DatetimeIndex should remain DatetimeIndex
    idx = pd.Index(pd.to_datetime(["2020-01-01", "2020-01-02"]), name="date")
    res = to_numeric(idx)
    pd.testing.assert_index_equal(res, idx)

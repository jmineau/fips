import numpy as np
import pandas as pd
import pytest

from fips.base import (
    MultiBlockMixin,
    Pickleable,
    SingleBlockMixin,
    Structure,
    Structure1D,
    Structure2D,
    xselect,
)


def test_xselect():
    midx = pd.MultiIndex.from_product([["A", "B"], [1, 2]], names=["L1", "L2"])
    df = pd.DataFrame({"val": [10, 20, 30, 40]}, index=midx)

    # Select L1="A"
    res = xselect(df, key="A", axis=0, level="L1", drop_level=True)
    expected = pd.DataFrame({"val": [10, 20]}, index=pd.Index([1, 2], name="L2"))
    pd.testing.assert_frame_equal(res, expected)

    # Select L2=1
    res2 = xselect(df, key=1, axis=0, level="L2", drop_level=True)
    expected2 = pd.DataFrame({"val": [10, 30]}, index=pd.Index(["A", "B"], name="L1"))
    pd.testing.assert_frame_equal(res2, expected2)


class DummyPickleable(Pickleable):
    def __init__(self, val):
        self.val = val

    def __getstate__(self):
        return {"val": self.val}

    def __setstate__(self, state):
        self.val = state["val"]


def test_pickleable(tmp_path):
    obj = DummyPickleable(42)
    file_path = tmp_path / "test.pkl"

    obj.to_file(file_path)
    assert file_path.exists()

    loaded = DummyPickleable.from_file(file_path)
    assert loaded.val == 42

    # Test invalid extension
    with pytest.raises(ValueError):
        obj.to_file(tmp_path / "test.txt")

    with pytest.raises(ValueError):
        DummyPickleable.from_file(tmp_path / "test.txt")

    with pytest.raises(FileNotFoundError):
        DummyPickleable.from_file(tmp_path / "nonexistent.pkl")


class DummyStructure(Structure):
    def __init__(self, data, name=None):
        self.data = data
        self.name = name
        self._validate()
        self._sanitize()

    def __getstate__(self):
        return {"data": self.data, "name": self.name}

    def __setstate__(self, state):
        self.data = state["data"]
        self.name = state["name"]


def test_structure_validate_sanitize():
    # Test NaN validation
    with pytest.raises(ValueError, match="Data contains NaN values."):
        DummyStructure(pd.Series([1.0, np.nan], index=pd.Index([0, 1], name="idx")))

    # Test sanitize (to_numeric)
    s = pd.Series([1, 2], index=pd.Index(["1", "2"], name="a"))
    struct = DummyStructure(s)
    pd.testing.assert_index_equal(struct.index, pd.Index([1, 2], name="a"))


def test_structure_properties():
    s = pd.Series([1, 2], index=pd.Index([1, 2], name="a"))
    struct = DummyStructure(s)

    assert struct.shape == (2,)
    np.testing.assert_array_equal(struct.values, np.array([1, 2]))


def test_structure_reindex():
    s = pd.Series([1, 2], index=pd.Index([1, 2], name="a"))
    struct = DummyStructure(s)
    new_idx = pd.Index([2, 3], name="a")
    struct.reindex(new_idx, fill_value=0.0)

    midx = pd.MultiIndex.from_arrays([[1, 2]], names=["a"])
    struct_midx = DummyStructure(pd.Series([1, 2], index=midx))

    new_midx = pd.MultiIndex.from_arrays([[2, 3]], names=["a"])
    res = struct_midx.reindex(new_midx, fill_value=0.0)

    expected = pd.Series([2.0, 0.0], index=new_midx)
    pd.testing.assert_series_equal(res.data, expected)


def test_structure_round_index():
    s = pd.Series([1, 2], index=pd.Index([1.123, 2.456], name="a"))
    struct = DummyStructure(s)

    res = struct.round_index(1)
    expected_idx = pd.Index([1.1, 2.5], name="a")
    pd.testing.assert_index_equal(res.index, expected_idx)


def test_structure1d():
    s = pd.Series([1, 2], name="test", index=pd.Index([1, 2], name="a"))
    struct = Structure1D(s)

    assert struct.name == "test"
    pd.testing.assert_series_equal(struct.to_series(), s)


def test_structure2d():
    df = pd.DataFrame(
        [[1, 2], [3, 4]],
        index=pd.Index([1, 2], name="a"),
        columns=pd.Index([1, 2], name="b"),
    )
    struct = Structure2D(df, name="test")

    assert struct.name == "test"
    pd.testing.assert_frame_equal(struct.to_frame(), df)
    pd.testing.assert_index_equal(struct.columns, pd.Index([1, 2], name="b"))

    # Test symmetric assumption
    struct_sym = Structure2D(
        np.array([[1, 2], [3, 4]]), index=pd.Index([1, 2], name="a")
    )
    pd.testing.assert_index_equal(struct_sym.columns, pd.Index([1, 2], name="a"))


class DummySingleBlock(SingleBlockMixin, DummyStructure):
    pass


def test_single_block_mixin():
    # Missing name
    with pytest.raises(ValueError, match="must have a name property"):
        DummySingleBlock(pd.Series([1, 2], index=pd.Index([0, 1], name="idx")))

    # Has block level
    midx = pd.MultiIndex.from_arrays([["b1", "b1"], [1, 2]], names=["block", "a"])
    with pytest.raises(ValueError, match="should not have a 'block' level"):
        DummySingleBlock(pd.Series([1, 2], index=midx), name="test")

    # Valid
    valid = DummySingleBlock(
        pd.Series([1, 2], index=pd.Index([1, 2], name="a")), name="test"
    )
    assert valid.name == "test"


class DummyMultiBlock(MultiBlockMixin, DummyStructure):
    pass


def test_multi_block_mixin():
    # Missing block level
    with pytest.raises(ValueError, match="must have a 'block' level"):
        DummyMultiBlock(
            pd.Series([1, 2], index=pd.Index([1, 2], name="a")), name="test"
        )

    # Valid
    midx = pd.MultiIndex.from_arrays([["b1", "b1"], [1, 2]], names=["block", "a"])
    valid = DummyMultiBlock(pd.Series([1, 2], index=midx), name="test")
    assert valid.name == "test"

    # Sanitize reorders block to first level
    midx_wrong_order = pd.MultiIndex.from_arrays(
        [[1, 2], ["b1", "b1"]], names=["a", "block"]
    )
    valid2 = DummyMultiBlock(pd.Series([1, 2], index=midx_wrong_order), name="test")
    assert valid2.index.names == ["block", "a"]

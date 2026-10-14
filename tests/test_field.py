"""Tests for fips.field: values on a space."""

import pickle

import numpy as np
import pandas as pd
import pytest

from fips import Field, Space


@pytest.fixture
def space():
    """Integer cells and string sites, in two blocks."""
    return Space(
        {
            "flux": pd.Index([1, 2, 3], name="cell"),
            "bias": pd.Index(["A", "B"], name="site"),
        }
    )


@pytest.fixture
def field(space):
    """Values 1 to 5 on the space."""
    return Field([1.0, 2.0, 3.0, 4.0, 5.0], space, name="prior")


class TestConstruction:
    def test_values_space_name(self, field, space):
        np.testing.assert_array_equal(field.values, [1, 2, 3, 4, 5])
        assert field.values.dtype == np.float64
        assert field.space is space
        assert field.name == "prior"
        assert field.index.equals(space.index)
        assert len(field) == 5

    def test_values_are_copied_and_read_only(self, space):
        values = np.arange(5.0)
        f = Field(values, space)
        values[0] = 99.0
        assert f.values[0] == 0.0
        with pytest.raises(ValueError, match="read-only"):
            f.values[0] = 1.0

    def test_name_becomes_a_string(self, space):
        assert Field(np.zeros(5), space, name=2020).name == "2020"
        assert Field(np.zeros(5), space).name is None

    def test_wrong_length(self, space):
        with pytest.raises(ValueError, match="Got 4 values for a space of 5 labels"):
            Field(np.zeros(4), space)

    def test_not_1d(self, space):
        with pytest.raises(ValueError, match="1D"):
            Field(np.zeros((5, 1)), space)

    def test_nan_names_its_labels(self, space):
        with pytest.raises(ValueError, match=r"NaN at 1 labels: \('bias', nan, 'B'\)"):
            Field([1.0, 2.0, 3.0, 4.0, np.nan], space)

    def test_a_series_is_refused(self, space):
        """A Series' own labels would be ignored, so the constructor refuses it."""
        series = pd.Series(np.arange(5.0))
        with pytest.raises(TypeError, match="Field.from_series"):
            Field(series, space)

    def test_space_must_be_a_space(self, space):
        with pytest.raises(TypeError, match="must be a Space"):
            Field(np.zeros(5), space.index)


class TestFromSeries:
    def test_with_block_level(self, field):
        back = Field.from_series(field.to_series())
        assert back.space == field.space
        np.testing.assert_array_equal(back.values, field.values)
        assert back.name == "prior"

    def test_rows_of_a_block_are_put_together(self):
        index = pd.MultiIndex.from_tuples(
            [("a", 1), ("b", 9), ("a", 2)], names=["block", "cell"]
        )
        f = Field.from_series(pd.Series([10.0, 90.0, 20.0], index=index))
        assert f.space.names == ("a", "b")
        np.testing.assert_array_equal(f.values, [10.0, 20.0, 90.0])
        assert f.index.tolist() == [("a", 1), ("a", 2), ("b", 9)]

    def test_without_block_level(self):
        series = pd.Series(
            [410.2, 412.8], index=pd.Index(["001", "002"], name="site"), name="co2"
        )
        f = Field.from_series(series, block="obs")
        assert f.space.names == ("obs",)
        assert f.space.select("obs").tolist() == ["001", "002"]  # not 1, 2 (#77)
        assert f.name == "co2"

    def test_without_block_level_needs_block(self):
        with pytest.raises(ValueError, match="needs `block`"):
            Field.from_series(pd.Series([1.0], index=pd.Index([1], name="cell")))

    def test_block_level_refuses_block(self, field):
        with pytest.raises(ValueError, match="leave `block` out"):
            Field.from_series(field.to_series(), block="flux")

    def test_duplicates_raise(self):
        series = pd.Series([1.0, 2.0], index=pd.Index(["a", "a"], name="site"))
        with pytest.raises(ValueError, match="Block 'obs' has duplicate labels: 'a'"):
            Field.from_series(series, block="obs")


class TestFromBlocks:
    def test_blocks_in_order(self, field):
        f = Field.from_blocks(
            {
                "flux": pd.Series(
                    [1.0, 2.0, 3.0], index=pd.Index([1, 2, 3], name="cell")
                ),
                "bias": pd.Series([4.0, 5.0], index=pd.Index(["A", "B"], name="site")),
            },
            name="prior",
        )
        assert f.space == field.space
        np.testing.assert_array_equal(f.values, field.values)
        assert f.name == "prior"

    def test_a_block_with_a_block_level_raises(self, field):
        with pytest.raises(ValueError, match="has a 'block' level"):
            Field.from_blocks({"x": field.to_series()})


class TestSelectAndConvert:
    def test_getitem_gives_a_field_of_one_block(self, field):
        flux = field["flux"]
        assert isinstance(flux, Field)
        assert flux.space.names == ("flux",)
        np.testing.assert_array_equal(flux.values, [1.0, 2.0, 3.0])
        # the block's labels as given: integers, not the padded floats (R1-18)
        pd.testing.assert_index_equal(
            flux.space.select("flux"), pd.Index([1, 2, 3], name="cell")
        )
        assert flux.name == "prior"

    def test_getitem_unknown_block(self, field):
        with pytest.raises(KeyError, match="No block named 'flx'"):
            field["flx"]

    def test_to_series_is_a_copy(self, field):
        series = field.to_series()
        assert series.index.equals(field.index)
        assert series.name == "prior"
        series.iloc[0] = 100.0
        assert field.values[0] == 1.0

    def test_align_reorders_to_a_space_with_the_same_labels(self, field):
        rng = np.random.default_rng(1)
        shuffled = Space.from_index(field.index[rng.permutation(len(field))])
        moved = field.align(shuffled)
        assert moved.space is shuffled
        pd.testing.assert_series_equal(
            moved.to_series().sort_index(), field.to_series().sort_index()
        )

    def test_align_raises_on_other_labels(self, field):
        other = Space(
            {
                "flux": pd.Index([1, 2, 4], name="cell"),
                "bias": field.space.select("bias"),
            }
        )
        with pytest.raises(ValueError, match="missing from the other space: 4"):
            field.align(other)

    def test_round_labels(self):
        space = Space({"s": pd.Index([0.1000001, 0.2], name="depth")})
        f = Field([1.0, 2.0], space).round_labels(3)
        assert f.space.select("s").tolist() == [0.1, 0.2]
        np.testing.assert_array_equal(f.values, [1.0, 2.0])

    def test_round_labels_refuses_a_merge(self):
        space = Space({"s": pd.Index([0.1000001, 0.1000002], name="depth")})
        with pytest.raises(ValueError, match="merges labels of block 's'"):
            Field([1.0, 2.0], space).round_labels(3)

    def test_repr(self, field):
        text = repr(field)
        assert text.startswith(
            "Field(name='prior', Space(flux: 3 x ['cell'], bias: 2 x ['site']))"
        )

    def test_pickle_keeps_it_read_only(self, field):
        back = pickle.loads(pickle.dumps(field))
        assert back.space == field.space
        np.testing.assert_array_equal(back.values, field.values)
        assert back.name == "prior"
        assert not back.values.flags.writeable

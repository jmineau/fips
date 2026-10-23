"""Tests for fips.vector: values on a space."""

import pickle

import numpy as np
import pandas as pd
import pytest

from fips import Space, Vector


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
def vector(space):
    """Values 1 to 5 on the space."""
    return Vector([1.0, 2.0, 3.0, 4.0, 5.0], space, name="prior")


class TestConstruction:
    def test_values_space_name(self, vector, space):
        np.testing.assert_array_equal(vector.values, [1, 2, 3, 4, 5])
        assert vector.values.dtype == np.float64
        assert vector.space is space
        assert vector.name == "prior"
        assert vector.index.equals(space.index)
        assert len(vector) == 5

    def test_values_are_copied_and_read_only(self, space):
        values = np.arange(5.0)
        f = Vector(values, space)
        values[0] = 99.0
        assert f.values[0] == 0.0
        with pytest.raises(ValueError, match="read-only"):
            f.values[0] = 1.0

    def test_name_becomes_a_string(self, space):
        assert Vector(np.zeros(5), space, name=2020).name == "2020"
        assert Vector(np.zeros(5), space).name is None

    def test_wrong_length(self, space):
        with pytest.raises(ValueError, match="Got 4 values for a space of 5 labels"):
            Vector(np.zeros(4), space)

    def test_not_1d(self, space):
        with pytest.raises(ValueError, match="1D"):
            Vector(np.zeros((5, 1)), space)

    def test_nan_names_its_labels(self, space):
        with pytest.raises(ValueError, match=r"NaN at 1 labels: \('bias', nan, 'B'\)"):
            Vector([1.0, 2.0, 3.0, 4.0, np.nan], space)

    def test_a_series_is_refused(self, space):
        """A Series' own labels would be ignored, so the constructor refuses it."""
        series = pd.Series(np.arange(5.0))
        with pytest.raises(TypeError, match="Vector.from_series"):
            Vector(series, space)

    def test_a_vector_is_refused(self, vector, space):
        """A Vector's labels would be ignored: it points to align and values."""
        reversed_space = Space({name: space.select(name)[::-1] for name in space.names})
        with pytest.raises(TypeError, match=r"v\.align\(space\).*v\.values"):
            Vector(vector, reversed_space)

    def test_an_integer_block_name_reads_back(self):
        """A block named 0 is kept as "0", and ``v[0]`` finds it."""
        v = Vector.from_blocks(
            {0: pd.Series([1.0, 2.0], index=pd.Index(["a", "b"], name="i"))}
        )
        np.testing.assert_array_equal(v[0].values, [1.0, 2.0])
        assert v[0].space == v["0"].space

    def test_space_must_be_a_space(self, space):
        with pytest.raises(TypeError, match="must be a Space"):
            Vector(np.zeros(5), space.index)


class TestFromSeries:
    def test_with_block_level(self, vector):
        back = Vector.from_series(vector.to_series())
        assert back.space == vector.space
        np.testing.assert_array_equal(back.values, vector.values)
        assert back.name == "prior"

    def test_rows_of_a_block_are_put_together(self):
        index = pd.MultiIndex.from_tuples(
            [("a", 1), ("b", 9), ("a", 2)], names=["block", "cell"]
        )
        f = Vector.from_series(pd.Series([10.0, 90.0, 20.0], index=index))
        assert f.space.names == ("a", "b")
        np.testing.assert_array_equal(f.values, [10.0, 20.0, 90.0])
        assert f.index.tolist() == [("a", 1), ("a", 2), ("b", 9)]

    def test_without_block_level(self):
        series = pd.Series(
            [410.2, 412.8], index=pd.Index(["001", "002"], name="site"), name="co2"
        )
        f = Vector.from_series(series, block="obs")
        assert f.space.names == ("obs",)
        assert f.space.select("obs").tolist() == ["001", "002"]  # not 1, 2 (#77)
        assert f.name == "co2"

    def test_without_block_level_needs_block(self):
        with pytest.raises(ValueError, match="needs `block`"):
            Vector.from_series(pd.Series([1.0], index=pd.Index([1], name="cell")))

    def test_block_level_refuses_block(self, vector):
        with pytest.raises(ValueError, match="leave `block` out"):
            Vector.from_series(vector.to_series(), block="flux")

    def test_duplicates_raise(self):
        series = pd.Series([1.0, 2.0], index=pd.Index(["a", "a"], name="site"))
        with pytest.raises(ValueError, match="Block 'obs' has duplicate labels: 'a'"):
            Vector.from_series(series, block="obs")


class TestFromBlocks:
    def test_blocks_in_order(self, vector):
        f = Vector.from_blocks(
            {
                "flux": pd.Series(
                    [1.0, 2.0, 3.0], index=pd.Index([1, 2, 3], name="cell")
                ),
                "bias": pd.Series([4.0, 5.0], index=pd.Index(["A", "B"], name="site")),
            },
            name="prior",
        )
        assert f.space == vector.space
        np.testing.assert_array_equal(f.values, vector.values)
        assert f.name == "prior"

    def test_a_block_with_a_block_level_raises(self, vector):
        with pytest.raises(ValueError, match="has a 'block' level"):
            Vector.from_blocks({"x": vector.to_series()})


class TestSelectAndConvert:
    def test_getitem_gives_a_vector_of_one_block(self, vector):
        flux = vector["flux"]
        assert isinstance(flux, Vector)
        assert flux.space.names == ("flux",)
        np.testing.assert_array_equal(flux.values, [1.0, 2.0, 3.0])
        # the block's labels as given: integers, not the padded floats (R1-18)
        pd.testing.assert_index_equal(
            flux.space.select("flux"), pd.Index([1, 2, 3], name="cell")
        )
        assert flux.name == "prior"

    def test_getitem_unknown_block(self, vector):
        with pytest.raises(KeyError, match="No block named 'flx'"):
            vector["flx"]

    def test_to_series_is_a_copy(self, vector):
        series = vector.to_series()
        assert series.index.equals(vector.index)
        assert series.name == "prior"
        series.iloc[0] = 100.0
        assert vector.values[0] == 1.0

    def test_to_series_of_a_block_is_on_its_own_labels(self, vector, space):
        """``to_series(block)`` is the block's values on its labels, as given, with no ``block`` level."""
        for block, values in (("flux", [1.0, 2.0, 3.0]), ("bias", [4.0, 5.0])):
            series = vector.to_series(block)
            pd.testing.assert_index_equal(series.index, space.select(block))
            np.testing.assert_array_equal(series.to_numpy(), values)
            assert series.name == "prior"
        assert vector.to_series("flux").index.dtype == np.int64

    def test_to_series_of_a_block_is_what_a_block_vector_gives(self, vector):
        """``v.to_series(b)`` equals ``v[b].to_series()`` without its ``block`` level."""
        pd.testing.assert_series_equal(
            vector.to_series("bias"),
            vector["bias"].to_series().droplevel("block"),
        )

    def test_to_series_of_a_block_round_trips(self, vector):
        """``Vector.from_series(v.to_series(b), block=b)`` is ``v[b]``."""
        back = Vector.from_series(vector.to_series("flux"), block="flux")
        assert back.space == vector["flux"].space
        np.testing.assert_array_equal(back.values, vector["flux"].values)

    def test_to_series_of_a_block_is_a_copy(self, vector):
        series = vector.to_series("flux")
        series.iloc[0] = 100.0
        assert vector.values[0] == 1.0

    def test_to_series_of_an_unknown_block_raises(self, vector):
        with pytest.raises(KeyError, match="No block named 'flx'"):
            vector.to_series("flx")

    def test_align_reorders_to_a_space_with_the_same_labels(self, vector):
        rng = np.random.default_rng(1)
        shuffled = Space.from_index(vector.index[rng.permutation(len(vector))])
        moved = vector.align(shuffled)
        assert moved.space is shuffled
        pd.testing.assert_series_equal(
            moved.to_series().sort_index(), vector.to_series().sort_index()
        )

    def test_align_raises_on_other_labels(self, vector):
        other = Space(
            {
                "flux": pd.Index([1, 2, 4], name="cell"),
                "bias": vector.space.select("bias"),
            }
        )
        with pytest.raises(ValueError, match="missing from the other space: 4"):
            vector.align(other)

    def test_round_labels(self):
        space = Space({"s": pd.Index([0.1000001, 0.2], name="depth")})
        f = Vector([1.0, 2.0], space).round_labels(3)
        assert f.space.select("s").tolist() == [0.1, 0.2]
        np.testing.assert_array_equal(f.values, [1.0, 2.0])

    def test_round_labels_refuses_a_merge(self):
        space = Space({"s": pd.Index([0.1000001, 0.1000002], name="depth")})
        with pytest.raises(ValueError, match="merges labels of block 's'"):
            Vector([1.0, 2.0], space).round_labels(3)

    def test_repr(self, vector):
        text = repr(vector)
        assert text.startswith(
            "Vector(name='prior', Space(flux: 3 x ['cell'], bias: 2 x ['site']))"
        )

    def test_pickle_keeps_it_read_only(self, vector):
        back = pickle.loads(pickle.dumps(vector))
        assert back.space == vector.space
        np.testing.assert_array_equal(back.values, vector.values)
        assert back.name == "prior"
        assert not back.values.flags.writeable

    def test_labels_are_not_converted(self):
        """Site codes stay strings (#77)."""
        series = pd.Series(
            [1.0, 2.0, 3.0], index=pd.Index(["001", "002", "010"], name="site")
        )
        vector = Vector.from_series(series, block="obs")
        assert vector["obs"].space.select("obs").tolist() == ["001", "002", "010"]


class TestToXarray:
    def test_one_block_is_a_data_array_on_its_own_levels(self):
        index = pd.MultiIndex.from_product(
            [[40.0, 41.0], [-112.0, -111.0, -110.0]], names=["lat", "lon"]
        )
        values = np.arange(6.0)
        vector = Vector.from_series(pd.Series(values, index=index), block="flux")
        array = vector.to_xarray()
        assert array.name == "flux"
        assert array.dims == ("lat", "lon")
        np.testing.assert_array_equal(array.values, values.reshape(2, 3))

    def test_several_blocks_are_a_dataset(self, vector):
        dataset = vector.to_xarray()
        assert list(dataset.data_vars) == ["flux", "bias"]
        np.testing.assert_array_equal(dataset["flux"].values, [1.0, 2.0, 3.0])
        assert dataset["bias"].dims == ("site",)
        np.testing.assert_array_equal(dataset["bias"].values, [4.0, 5.0])


class TestFromBlocksWithVectors:
    """``from_blocks`` takes a vector of one block too, as ``offset_block`` gives it."""

    def test_a_vector_beside_a_series(self, vector):
        """The key names the block; the vector brings its labels and values."""
        series = pd.Series([7.0, 8.0], index=pd.Index(["x", "y"], name="cell"))
        both = Vector.from_blocks({"flux": series, "offset": vector["bias"]})
        assert both.space.names == ("flux", "offset")
        assert both.space.select("offset").equals(vector.space.select("bias"))
        np.testing.assert_array_equal(both.values, [7.0, 8.0, 4.0, 5.0])

    def test_a_vector_of_several_blocks_raises(self, vector):
        """A vector for one block must have one block."""
        with pytest.raises(ValueError, match="give one block"):
            Vector.from_blocks({"all": vector})


class TestCombineFirst:
    """``combine_first``: a smaller vector's values on a larger space."""

    def test_values_where_this_vector_has_the_label(self, vector):
        """The values of this vector at its labels (in any order), the other's elsewhere."""
        part = Vector(
            [30.0, 10.0],
            Space({"flux": pd.Index([3, 1], name="cell")}),
            name="estimate",
        )
        full = part.combine_first(vector)
        assert full.space == vector.space
        assert full.name == "estimate"
        np.testing.assert_array_equal(full.values, [10.0, 2.0, 30.0, 4.0, 5.0])
        np.testing.assert_array_equal(vector.values, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_labels_the_other_lacks_raise(self, vector):
        """A label or a block of this vector that the other lacks is named."""
        stray = Vector([1.0], Space({"flux": pd.Index([9], name="cell")}))
        with pytest.raises(
            ValueError, match="labels of block 'flux' not in the space: 9"
        ):
            stray.combine_first(vector)
        other_block = Vector([1.0], Space({"co2": pd.Index([1], name="cell")}))
        with pytest.raises(ValueError, match="blocks not in the space: 'co2'"):
            other_block.combine_first(vector)

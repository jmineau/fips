"""Tests for fips.vector: values on a space."""

import pickle

import numpy as np
import pandas as pd
import pytest

from fips import Covariance, Operator, Space, Vector


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


OPERATORS = {
    "+": (np.add, lambda a, b: a + b),
    "-": (np.subtract, lambda a, b: a - b),
    "*": (np.multiply, lambda a, b: a * b),
    "/": (np.true_divide, lambda a, b: a / b),
}


@pytest.fixture
def shuffled(space):
    """The same labels as `space` in another order (blocks and cells), values 10 to 50 by label."""
    other = Space(
        {
            "bias": pd.Index(["B", "A"], name="site"),
            "flux": pd.Index([3, 1, 2], name="cell"),
        }
    )
    # by label: flux 1, 2, 3 = 10, 20, 30; bias A, B = 40, 50
    return Vector([50.0, 40.0, 30.0, 10.0, 20.0], other, name="obs")


class TestArithmetic:
    """``+ - * /`` between vectors, numbers and arrays, aligned by label."""

    @pytest.mark.parametrize("symbol", OPERATORS)
    def test_two_vectors_in_one_order(self, vector, symbol):
        """The same space object: the values combine in place."""
        _, op = OPERATORS[symbol]
        twin = Vector([5.0, 4.0, 3.0, 2.0, 1.0], vector.space)
        result = op(vector, twin)
        np.testing.assert_array_equal(result.values, op(vector.values, twin.values))
        assert result.space is vector.space

    @pytest.mark.parametrize("symbol", OPERATORS)
    def test_two_vectors_in_another_order(self, vector, shuffled, symbol):
        """The right operand is matched to the left's labels, not its positions."""
        _, op = OPERATORS[symbol]
        aligned = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = op(vector, shuffled)
        np.testing.assert_array_equal(result.values, op(vector.values, aligned))
        assert result.space is vector.space
        # and the other way round, on the shuffled space
        back = op(shuffled, vector)
        np.testing.assert_array_equal(
            back.values, op(shuffled.values, vector.align(shuffled.space).values)
        )
        assert back.space is shuffled.space

    @pytest.mark.parametrize("symbol", OPERATORS)
    def test_a_number(self, vector, symbol):
        _, op = OPERATORS[symbol]
        for number in (2.0, 3, np.float64(2.0), np.int64(3)):
            np.testing.assert_array_equal(
                op(vector, number).values, op(vector.values, number)
            )
            np.testing.assert_array_equal(
                op(number, vector).values, op(number, vector.values)
            )

    @pytest.mark.parametrize("symbol", OPERATORS)
    def test_an_array_by_position(self, vector, symbol):
        _, op = OPERATORS[symbol]
        array = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        np.testing.assert_array_equal(
            op(vector, array).values, op(vector.values, array)
        )
        np.testing.assert_array_equal(
            op(array, vector).values, op(array, vector.values)
        )
        assert isinstance(op(array, vector), Vector)
        np.testing.assert_array_equal(
            op(vector, list(array)).values, op(vector.values, array)
        )

    def test_reflected_forms(self, vector):
        np.testing.assert_array_equal((2 - vector).values, [1, 0, -1, -2, -3])
        np.testing.assert_array_equal((1 / vector).values, 1 / vector.values)
        np.testing.assert_array_equal((2 + vector).values, [3, 4, 5, 6, 7])
        np.testing.assert_array_equal((2 * vector).values, [2, 4, 6, 8, 10])

    def test_negative(self, vector):
        result = -vector
        np.testing.assert_array_equal(result.values, -vector.values)
        assert result.space is vector.space
        assert result.name == "prior"

    def test_name_kept_when_both_agree(self, vector, shuffled):
        same = Vector(np.ones(5), vector.space, name="prior")
        assert (vector + same).name == "prior"
        assert (vector + shuffled).name is None  # "prior" and "obs"
        assert (vector + Vector(np.ones(5), vector.space)).name is None
        assert (shuffled + vector).name is None

    def test_name_kept_with_a_number_or_array(self, vector):
        assert (2 * vector).name == "prior"
        assert (vector - np.ones(5)).name == "prior"
        assert (2 * Vector(np.ones(5), vector.space)).name is None

    def test_result_is_read_only_and_new(self, vector):
        result = vector + 1.0
        assert result is not vector
        with pytest.raises(ValueError, match="read-only"):
            result.values[0] = 0.0
        np.testing.assert_array_equal(vector.values, [1, 2, 3, 4, 5])

    def test_result_owns_its_values(self, vector):
        """The result does not share memory with either operand."""
        result = vector * 1.0
        assert not np.shares_memory(result.values, vector.values)

    def test_a_missing_label_raises(self, vector):
        part = vector["flux"]
        with pytest.raises(
            ValueError, match="The two vectors do not have the same labels"
        ):
            vector + part
        with pytest.raises(
            ValueError, match=r"blocks missing from the other vector: 'bias'"
        ):
            vector - part

    def test_an_extra_label_raises(self, vector, space):
        wider = Vector(
            np.ones(6),
            Space(
                {
                    "flux": pd.Index([1, 2, 3, 4], name="cell"),
                    "bias": space.select("bias"),
                }
            ),
        )
        with pytest.raises(
            ValueError, match=r"labels of block 'flux' only in the other vector: 4"
        ):
            vector * wider

    def test_a_wrong_label_names_both_sides(self, vector, space):
        moved = Vector(
            np.ones(5),
            Space(
                {
                    "flux": pd.Index([1, 2, 9], name="cell"),
                    "bias": space.select("bias"),
                }
            ),
        )
        with pytest.raises(ValueError) as caught:
            vector + moved
        message = str(caught.value)
        assert "labels of block 'flux' missing from the other vector: 3" in message
        assert "labels of block 'flux' only in the other vector: 9" in message

    def test_a_wrong_length_array_raises(self, vector):
        with pytest.raises(ValueError, match="Got 4 values for a space of 5 labels"):
            vector + np.ones(4)
        with pytest.raises(ValueError, match="1D"):
            vector + np.ones((5, 1))

    def test_pandas_objects_are_refused(self, vector):
        series = vector.to_series()
        for other in (series, series.to_frame(), series.index):
            with pytest.raises(TypeError, match="Vector.from_series"):
                vector + other
            with pytest.raises(TypeError, match=r"\.values"):
                vector * other

    def test_an_operator_or_covariance_is_refused(self, vector):
        op = Operator(np.eye(5), vector.space, vector.space)
        cov = Covariance.diagonal(1.0, vector.space)
        for other in (op, cov):
            with pytest.raises(TypeError, match="Vector.from_series"):
                vector + other
            with pytest.raises(TypeError, match="applied with @"):
                vector * other

    def test_a_result_that_is_nan_names_its_labels(self, vector):
        zero = Vector([0.0, 1.0, 1.0, 1.0, 1.0], vector.space)
        with (
            np.errstate(invalid="ignore"),
            pytest.raises(ValueError, match=r"NaN at 1 labels: \('flux', 1, nan\)"),
        ):
            zero / zero

    def test_division_by_zero_is_infinite(self, vector):
        with np.errstate(divide="ignore"):
            result = vector / 0.0
        assert np.isinf(result.values).all()

    def test_a_multi_level_block_and_a_padded_level(self):
        """A block of (time, cell) beside a block with one level: the padded MultiIndex."""
        flux = pd.MultiIndex.from_product(
            [pd.to_datetime(["2020-01-01", "2020-02-01"]), [1, 2]],
            names=["time", "cell"],
        )
        bias = pd.Index(["A"], name="site")
        space = Space({"flux": flux, "bias": bias})
        v = Vector([1.0, 2.0, 3.0, 4.0, 5.0], space, name="x")
        reordered = Space({"bias": bias, "flux": flux[[3, 0, 2, 1]]})
        w = Vector([5.0, 40.0, 10.0, 30.0, 20.0], reordered)
        result = v + w
        np.testing.assert_array_equal(result.values, [11, 22, 33, 44, 10])
        assert result.space is space
        np.testing.assert_array_equal((v * w).values, [10, 40, 90, 160, 25])
        assert result["flux"].space.select("flux").equals(flux)

    def test_a_vector_with_levels_in_another_order(self):
        """A block's levels may come in another order, as well as its labels."""
        flux = pd.MultiIndex.from_product([[1, 2], ["a", "b"]], names=["cell", "kind"])
        other = pd.MultiIndex.from_tuples(
            [("b", 1), ("a", 1), ("b", 2), ("a", 2)], names=["kind", "cell"]
        )
        v = Vector([1.0, 2.0, 3.0, 4.0], Space({"flux": flux}))
        w = Vector([20.0, 10.0, 40.0, 30.0], Space({"flux": other}))
        np.testing.assert_array_equal((v + w).values, [11, 22, 33, 44])


class TestUfuncs:
    """NumPy's element-wise functions return vectors."""

    @pytest.mark.parametrize("function", [np.exp, np.log, np.sqrt, np.abs, np.negative])
    def test_unary(self, vector, function):
        result = function(vector)
        assert isinstance(result, Vector)
        np.testing.assert_array_equal(result.values, function(vector.values))
        assert result.space is vector.space
        assert result.name == "prior"

    def test_two_vectors_are_aligned(self, vector, shuffled):
        result = np.maximum(vector, shuffled)
        assert isinstance(result, Vector)
        np.testing.assert_array_equal(result.values, [10, 20, 30, 40, 50])
        assert result.space is vector.space
        assert result.name is None
        flipped = np.minimum(shuffled, vector)
        assert flipped.space is shuffled.space
        np.testing.assert_array_equal(flipped.values, [5, 4, 3, 1, 2])

    def test_numbers_and_arrays(self, vector):
        np.testing.assert_array_equal(
            np.maximum(vector, 2.5).values, [2.5, 2.5, 3, 4, 5]
        )
        np.testing.assert_array_equal(np.power(vector, 2).values, vector.values**2)
        np.testing.assert_array_equal(np.power(2.0, vector).values, 2.0**vector.values)
        array = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert isinstance(np.add(array, vector), Vector)
        np.testing.assert_array_equal(np.minimum(array, vector).values, [1, 2, 3, 2, 1])

    def test_numpy_operands_on_the_left_give_vectors(self, vector):
        array = np.ones(5)
        for result in (array + vector, np.float64(2.0) - vector, array * vector):
            assert isinstance(result, Vector)
            assert result.space is vector.space
        np.testing.assert_array_equal(
            (np.float64(2.0) - vector).values, [1, 0, -1, -2, -3]
        )

    def test_reductions_are_not_defined(self, vector):
        with pytest.raises(TypeError):
            np.sum(vector)
        with pytest.raises(TypeError):
            np.add.reduce(vector)
        with pytest.raises(TypeError):
            np.maximum.accumulate(vector)
        assert np.sum(vector.values) == 15.0

    def test_out_and_other_keywords_are_not_defined(self, vector):
        with pytest.raises(TypeError):
            np.exp(vector, out=np.empty(5))
        with pytest.raises(TypeError):
            np.exp(vector, where=np.ones(5, dtype=bool))

    def test_comparisons_and_several_outputs_are_not_defined(self, vector):
        with pytest.raises(TypeError):
            np.isnan(vector)
        with pytest.raises(TypeError):
            np.greater(vector, 2.0)
        with pytest.raises(TypeError):
            np.modf(vector)

    def test_no_array_conversion(self, vector):
        """``np.asarray(v)`` stays explicit: a vector is not an array."""
        assert not hasattr(vector, "__array__")

    def test_a_series_is_refused(self, vector):
        with pytest.raises(TypeError, match="Vector.from_series"):
            np.add(vector, vector.to_series())

    def test_a_mean_preserving_lognormal_draw(self, vector, space):
        """The one-line truth of the synthetic guide stays on the prior's space."""
        cov = Covariance.diagonal(0.04, space)
        rng = np.random.default_rng(0)
        truth = vector * np.exp(cov.sample(rng) - 0.5 * cov.variances)
        assert isinstance(truth, Vector)
        assert truth.space is vector.space
        assert (truth.values > 0).all()
        # the mean over many draws is the prior
        draws = np.mean(
            [
                (vector * np.exp(cov.sample(rng) - 0.5 * cov.variances)).values
                for _ in range(2000)
            ],
            axis=0,
        )
        np.testing.assert_allclose(draws, vector.values, rtol=0.02)

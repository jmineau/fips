"""Tests for fips.space: the labels of a state, the observations or a matrix axis."""

import pickle

import numpy as np
import pandas as pd
import pytest

from fips import Space


@pytest.fixture
def ragged():
    """Two blocks with different levels: integer cells, and sites."""
    return Space(
        {
            "flux": pd.Index([1, 2, 3], name="cell"),
            "bias": pd.Index(["A", "B"], name="site"),
        }
    )


@pytest.fixture
def gridded():
    """A block on (time, lat, lon) and one on site."""
    flux = pd.MultiIndex.from_product(
        [pd.to_datetime(["2020-01-01", "2020-02-01"]), [40.5, 40.6], [-112.0]],
        names=["time", "lat", "lon"],
    )
    return Space({"flux": flux, "bias": pd.Index(["A"], name="site")})


class TestConstruction:
    def test_index_has_block_first_and_pads_missing_levels(self, ragged):
        index = ragged.index
        assert list(index.names) == ["block", "cell", "site"]
        assert list(index.get_level_values("block")) == ["flux"] * 3 + ["bias"] * 2
        cell = index.get_level_values("cell")
        assert cell[:3].tolist() == [1, 2, 3] and cell[3:].isna().all()
        assert index.get_level_values("site")[:3].isna().all()

    def test_padding_keeps_each_level_dtype(self, ragged, gridded):
        """An integer level stays integer, a datetime level datetime (R1-18)."""
        assert ragged.index.levels[1].dtype == np.int64
        assert pd.api.types.is_datetime64_any_dtype(gridded.index.levels[1])

    def test_levels_are_sorted_so_the_index_sorts(self, ragged):
        """Each level is sorted, as from_arrays makes it; pandas 2.0's sort_index needs it."""
        assert all(level.is_monotonic_increasing for level in ragged.index.levels)
        series = pd.Series(np.arange(5.0), index=ragged.index)
        assert series.sort_index().index.get_level_values("block").tolist() == [
            "bias",
            "bias",
            "flux",
            "flux",
            "flux",
        ]

    def test_select_gives_the_block_as_given(self, ragged, gridded):
        pd.testing.assert_index_equal(
            ragged.select("flux"), pd.Index([1, 2, 3], name="cell")
        )
        assert gridded.select("flux").names == ["time", "lat", "lon"]

    def test_levels_in_order_of_first_appearance(self):
        space = Space(
            {
                "a": pd.MultiIndex.from_tuples([(1, "x")], names=["i", "j"]),
                "b": pd.MultiIndex.from_tuples([("y", 2)], names=["k", "i"]),
            }
        )
        assert space.levels == ("i", "j", "k")
        assert list(space.index.names) == ["block", "i", "j", "k"]

    def test_levels_argument_orders_the_index(self, ragged):
        space = Space(
            {name: ragged.select(name) for name in ragged}, levels=["site", "cell"]
        )
        assert list(space.index.names) == ["block", "site", "cell"]
        assert space == ragged  # the same blocks; only the padded order differs

    @pytest.mark.parametrize("levels", [["cell"], ["cell", "site", "cell"], ["x"]])
    def test_levels_must_name_each_level_once(self, ragged, levels):
        with pytest.raises(ValueError, match="levels must name each level"):
            Space({name: ragged.select(name) for name in ragged}, levels=levels)

    def test_offsets_slice_and_len(self, ragged):
        np.testing.assert_array_equal(ragged.offsets, [0, 3, 5])
        assert ragged.slice("flux") == slice(0, 3)
        assert ragged.slice("bias") == slice(3, 5)
        assert len(ragged) == 5
        assert ragged.names == ("flux", "bias")
        assert list(ragged) == ["flux", "bias"]
        assert "bias" in ragged and "conc" not in ragged

    def test_a_one_level_multiindex_is_a_plain_index(self):
        """A block labeled by a MultiIndex of one level is that level, as from_index gives it."""
        labels = pd.MultiIndex.from_arrays([[1, 2]], names=["cell"])
        space = Space({"flux": labels})
        assert not isinstance(space.select("flux"), pd.MultiIndex)
        assert space.select("flux").equals(pd.Index([1, 2], name="cell"))
        assert space == Space.from_index(space.index)

    def test_block_names_become_strings(self):
        space = Space({2020: pd.Index([1], name="cell")})
        assert space.names == ("2020",)

    def test_two_blocks_with_one_name_raise(self):
        with pytest.raises(ValueError, match="Two blocks are named '1'"):
            Space({1: pd.Index([1], name="a"), "1": pd.Index([2], name="a")})

    def test_unknown_block_names_the_known_ones(self, ragged):
        with pytest.raises(KeyError, match="No block named 'flx'.*'flux', 'bias'"):
            ragged.select("flx")

    def test_repr(self, ragged):
        assert repr(ragged) == "Space(flux: 3 x ['cell'], bias: 2 x ['site'])"


class TestRefusals:
    def test_no_blocks(self):
        with pytest.raises(ValueError, match="at least one block"):
            Space({})

    def test_not_an_index(self):
        with pytest.raises(TypeError, match="must be a pandas Index"):
            Space({"a": [1, 2]})

    def test_empty_block(self):
        with pytest.raises(ValueError, match="Block 'a' has no labels"):
            Space({"a": pd.Index([], name="cell")})

    def test_unnamed_level(self):
        with pytest.raises(ValueError, match="Every level of block 'a' must be named"):
            Space({"a": pd.Index([1, 2])})

    def test_block_level(self):
        idx = pd.MultiIndex.from_tuples([("x", 1)], names=["block", "cell"])
        with pytest.raises(ValueError, match="has a 'block' level"):
            Space({"a": idx})

    @pytest.mark.parametrize(
        "index",
        [
            pd.Index([1.0, np.nan], name="cell"),
            pd.MultiIndex.from_tuples([(1, "x"), (2, None)], names=["i", "j"]),
        ],
        ids=["flat", "multi"],
    )
    def test_missing_label(self, index):
        with pytest.raises(ValueError, match="missing \\(NaN\\) labels"):
            Space({"a": index})

    def test_duplicate_labels_are_named(self):
        """Duplicates raise and the message names them (#78)."""
        idx = pd.MultiIndex.from_tuples(
            [(1, "x"), (2, "y"), (1, "x"), (2, "y"), (3, "z")], names=["i", "j"]
        )
        with pytest.raises(
            ValueError,
            match=r"Block 'a' has duplicate labels: \(1, 'x'\), \(2, 'y'\)\.",
        ):
            Space({"a": idx})

    def test_labels_are_not_converted(self):
        """Strings of digits stay strings (#77)."""
        space = Space({"obs": pd.Index(["001", "010", "1e3"], name="site")})
        assert space.select("obs").tolist() == ["001", "010", "1e3"]
        assert space.index.get_level_values("site").tolist() == ["001", "010", "1e3"]


class TestFromIndex:
    def test_round_trip(self, ragged, gridded):
        for space in (ragged, gridded):
            back = Space.from_index(space.index)
            assert back == space
            for name in space:
                pd.testing.assert_index_equal(back.select(name), space.select(name))

    def test_float_padded_index(self):
        """An index padded by pandas (the level made float) gives the blocks back."""
        index = pd.MultiIndex.from_arrays(
            [["flux", "flux", "bias"], [1, 2, np.nan], [np.nan, np.nan, "A"]],
            names=["block", "cell", "site"],
        )
        space = Space.from_index(index)
        assert space.names == ("flux", "bias")
        assert space.select("flux").tolist() == [1.0, 2.0]
        assert space.select("bias").tolist() == ["A"]

    def test_block_level_anywhere_and_level_order_kept(self):
        index = pd.MultiIndex.from_tuples(
            [(1, "x", "a"), (2, "y", "a")], names=["cell", "site", "block"]
        )
        space = Space.from_index(index)
        assert space.levels == ("cell", "site")
        assert space.select("a").names == ["cell", "site"]

    def test_blocks_need_not_be_contiguous(self):
        index = pd.MultiIndex.from_tuples(
            [("a", 1), ("b", 9), ("a", 2)], names=["block", "cell"]
        )
        space = Space.from_index(index)
        assert space.names == ("a", "b")
        assert space.select("a").tolist() == [1, 2]

    def test_ragged_level_within_a_block_raises(self):
        index = pd.MultiIndex.from_arrays(
            [["a", "a"], [1.0, np.nan]], names=["block", "cell"]
        )
        with pytest.raises(ValueError, match="for some of its rows and not for others"):
            Space.from_index(index)

    def test_needs_a_block_level(self):
        with pytest.raises(ValueError, match="'block' level"):
            Space.from_index(pd.Index([1, 2], name="cell"))

    def test_a_label_without_block_raises(self):
        index = pd.MultiIndex.from_arrays(
            [["a", np.nan], [1, 2]], names=["block", "cell"]
        )
        with pytest.raises(ValueError, match="name its block"):
            Space.from_index(index)

    def test_a_block_with_no_other_level_raises(self):
        index = pd.MultiIndex.from_arrays(
            [["a", "b"], [1.0, np.nan]], names=["block", "cell"]
        )
        with pytest.raises(ValueError, match="Block 'b' has no level besides 'block'"):
            Space.from_index(index)

    def test_block_labels_with_one_string_raise(self):
        index = pd.MultiIndex.from_tuples([(1, 1), ("1", 2)], names=["block", "cell"])
        with pytest.raises(ValueError, match="Two blocks are named '1'"):
            Space.from_index(index)

    def test_duplicates_raise(self):
        index = pd.MultiIndex.from_tuples([("a", 1), ("a", 1)], names=["block", "cell"])
        with pytest.raises(ValueError, match="duplicate labels"):
            Space.from_index(index)


class TestAlign:
    def test_permutation(self, gridded):
        rng = np.random.default_rng(0)
        shuffled = gridded.index[rng.permutation(len(gridded))]
        other = Space.from_index(shuffled)
        positions = gridded.align(other)
        assert other.index[positions].equals(gridded.index)
        assert other.index.tolist() != gridded.index.tolist()  # it was shuffled

    def test_identity(self, ragged):
        np.testing.assert_array_equal(ragged.align(ragged), np.arange(len(ragged)))

    def test_levels_in_another_order(self):
        a = Space(
            {"x": pd.MultiIndex.from_tuples([(1, "p"), (2, "q")], names=["i", "j"])}
        )
        b = Space(
            {"x": pd.MultiIndex.from_tuples([("q", 2), ("p", 1)], names=["j", "i"])}
        )
        np.testing.assert_array_equal(a.align(b), [1, 0])

    def test_blocks_in_another_order(self, ragged):
        other = Space({"bias": ragged.select("bias"), "flux": ragged.select("flux")})
        np.testing.assert_array_equal(ragged.align(other), [2, 3, 4, 0, 1])

    def test_missing_and_extra_labels_are_named(self):
        a = Space({"x": pd.Index([1, 2, 3], name="cell")})
        b = Space({"x": pd.Index([2, 3, 4], name="cell")})
        with pytest.raises(ValueError) as excinfo:
            a.align(b)
        message = str(excinfo.value)
        assert "labels of block 'x' missing from the other space: 1" in message
        assert "labels of block 'x' only the other space has: 4" in message

    def test_missing_and_extra_blocks_are_named(self, ragged):
        other = Space({"flux": ragged.select("flux"), "conc": pd.Index([0], name="t")})
        with pytest.raises(
            ValueError, match="blocks missing from the other space: 'bias'"
        ):
            ragged.align(other)
        with pytest.raises(ValueError, match="blocks only the other space has: 'conc'"):
            ragged.align(other)

    def test_other_levels_raise(self):
        a = Space({"x": pd.Index([1], name="cell")})
        b = Space({"x": pd.Index([1], name="site")})
        with pytest.raises(
            ValueError, match="has levels \\['cell'\\] here and \\['site'\\]"
        ):
            a.align(b)

    def test_strings_do_not_match_numbers(self):
        """Labels that differ only in type are not joined: the exact join catches #77's mixups."""
        a = Space({"obs": pd.Index(["001", "002"], name="site")})
        b = Space({"obs": pd.Index([1, 2], name="site")})
        with pytest.raises(
            ValueError, match="missing from the other space: '001', '002'"
        ):
            a.align(b)


class TestRoundLabels:
    def test_rounds_float_levels_only(self, gridded):
        noisy = Space(
            {
                "flux": pd.MultiIndex.from_arrays(
                    [[40.5000001, 40.6000002], ["a", "b"]], names=["lat", "id"]
                )
            }
        )
        rounded = noisy.round_labels(4)
        assert rounded.select("flux").get_level_values("lat").tolist() == [40.5, 40.6]
        assert rounded.select("flux").get_level_values("id").tolist() == ["a", "b"]
        assert (
            noisy.select("flux").get_level_values("lat")[0] == 40.5000001
        )  # unchanged
        assert gridded.round_labels(6) == gridded

    def test_merged_labels_raise_and_are_named(self):
        space = Space({"s": pd.Index([10.1234567, 10.1234568, 20.0], name="depth")})
        with pytest.raises(ValueError) as excinfo:
            space.round_labels(6)
        message = str(excinfo.value)
        assert "Rounding to 6 decimals merges labels of block 's'" in message
        assert "10.1234567, 10.1234568 become 10.123457" in message


class TestEqualityAndPickle:
    def test_equality(self, ragged):
        same = Space(
            {
                "flux": pd.Index([1, 2, 3], name="cell"),
                "bias": pd.Index(["A", "B"], name="site"),
            }
        )
        assert ragged == same
        reordered = Space(
            {"bias": ragged.select("bias"), "flux": ragged.select("flux")}
        )
        assert ragged != reordered
        assert ragged != Space(
            {"flux": pd.Index([1, 2, 3], name="cell2"), "bias": ragged.select("bias")}
        )
        assert ragged.__eq__("not a space") is NotImplemented

    def test_pickle(self, gridded):
        back = pickle.loads(pickle.dumps(gridded))
        assert back == gridded
        assert back.index.equals(gridded.index)
        assert back.levels == gridded.levels

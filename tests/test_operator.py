"""
Tests of the labeled operator, :class:`fips.Operator`.

An operator is a matrix in its cheapest form (:mod:`fips.linalg`) with a
:class:`fips.Space` of labels on each axis. These tests cover the ways in
(an array with its spaces, a frame, blocks), the labels a block comes back
with, the exact join in products and sums, the operations that keep the form,
and the labeled product of a forward operator with a state.
"""

import pickle

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from fips import Covariance, Operator, Space, Vector
from fips.linalg import DenseOperator, DiagonalOperator, SparseOperator

# A scalar in each form a caller might pass: Python float and int, a NumPy scalar, a 0-d array
SCALARS = [5.0, 5, np.float64(5.0), np.array(5.0)]

# Row and column blocks that differ in their levels, so that each axis's
# index pads the levels a block lacks with NaN.
ROWS = {
    "o1": pd.MultiIndex.from_product([[0, 1], ["x"]], names=["t", "site"]),
    "o2": pd.Index([10, 11], name="k"),
}
COLS = {
    "s1": pd.MultiIndex.from_product([[0.5, 1.5], [7]], names=["lat", "lon"]),
    "s2": pd.Index(["p", "q"], name="param"),
}


def ragged_blocks() -> dict[tuple[str, str], pd.DataFrame]:
    """The four (row, column) blocks of the ragged spaces, each filled with its own value."""
    return {
        (r, c): pd.DataFrame(
            np.full((2, 2), 1.0 + 2 * i + j), index=ROWS[r], columns=COLS[c]
        )
        for i, r in enumerate(ROWS)
        for j, c in enumerate(COLS)
    }


@pytest.fixture
def state():
    """A state space of one block whose labels are not sorted."""
    return Space({"flux": pd.Index(["c", "a", "b"], name="cell")})


@pytest.fixture
def obs():
    """An observation space of one block."""
    return Space({"co2": pd.Index([1, 2], name="obs_id")})


@pytest.fixture
def other_state():
    """The state space with label 'b' replaced by 'z'."""
    return Space({"flux": pd.Index(["c", "a", "z"], name="cell")})


@pytest.fixture
def H(obs, state):
    """A dense 2 x 3 forward operator, named."""
    return Operator(np.arange(6.0).reshape(2, 3), obs, state, name="H")


def reordered(space: Space) -> Space:
    """The same labels as `space`, each block's labels reversed."""
    return Space({name: space.select(name)[::-1] for name in space.names})


def bare(op: Operator) -> pd.DataFrame:
    """The operator as a frame without its block levels, as a one-block matrix is given."""
    return op.to_frame().droplevel("block").droplevel("block", axis=1)


def by_label(op: Operator) -> pd.DataFrame:
    """The operator as a frame sorted by its labels, to compare operators whatever their order."""
    return op.to_frame().sort_index().sort_index(axis=1)


class TestConstructor:
    """Building an operator from a matrix and the spaces of its axes."""

    def test_spaces_and_shape(self, H, obs, state):
        """The spaces label the rows and columns, and give the shape."""
        assert H.rows is obs
        assert H.cols is state
        assert H.shape == (2, 3)
        assert H.name == "H"

    @pytest.mark.parametrize(
        ("data", "form"),
        [
            (np.ones((3, 3)), DenseOperator),
            (sp.csr_array(np.triu(np.ones((3, 3)))), SparseOperator),
            (sp.eye_array(3, format="csr"), DiagonalOperator),
        ],
        ids=["dense", "sparse", "sparse_diagonal"],
    )
    def test_form_follows_storage(self, data, form, state):
        """An array is dense, a sparse matrix stays sparse, a diagonal one is a diagonal."""
        op = Operator(data, state, state)

        assert type(op.data) is form
        np.testing.assert_array_equal(
            op.dense(), np.asarray(sp.csr_array(data).todense())
        )

    def test_form_is_kept(self, state):
        """A form of fips.linalg is the operator's data, as it is."""
        form = DiagonalOperator([1.0, 2.0, 3.0])

        assert Operator(form, state, state).data is form

    @pytest.mark.parametrize(
        "data",
        [
            pd.DataFrame(np.eye(3)),
            pd.Series([1.0, 2.0, 3.0]),
        ],
        ids=["frame", "series"],
    )
    def test_pandas_object_refused(self, data, state):
        """A pandas object would have its labels ignored, so it is refused, naming from_frame."""
        with pytest.raises(TypeError, match="from_frame"):
            Operator(data, state, state)

    def test_operator_refused(self, H, obs, state):
        """An Operator would be taken by position, its labels ignored: it points to align and data."""
        with pytest.raises(TypeError, match=r"op\.align\(rows, cols\).*op\.data"):
            Operator(H, reordered(obs), state)
        with pytest.raises(TypeError, match="not a labeled Covariance"):
            Operator(Covariance(np.eye(3), state), state, state)

    def test_spaces_must_be_spaces(self, state):
        """An index is not a Space."""
        with pytest.raises(TypeError, match="rows must be a Space"):
            Operator(np.eye(3), state.index, state)

    def test_shape_must_match_the_spaces(self, obs, state):
        """A matrix of the wrong shape raises, naming both."""
        with pytest.raises(ValueError, match=r"shape \(3, 2\).*2 rows and 3 columns"):
            Operator(np.ones((3, 2)), obs, state)

    @pytest.mark.parametrize(
        "data", [5.0, np.array([1.0, 2.0, 3.0])], ids=["scalar", "1d"]
    )
    def test_scalar_and_1d_refused(self, data, state):
        """A scalar or a 1D array is not a matrix."""
        with pytest.raises(ValueError, match="2D"):
            Operator(data, state, state)

    @pytest.mark.parametrize(
        "data",
        [
            np.array([[1.0, np.nan, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            sp.csr_array(([np.nan], ([0], [1])), shape=(3, 3)),
            DiagonalOperator([1.0, np.nan, 3.0]),
        ],
        ids=["dense", "sparse", "diagonal"],
    )
    def test_nan_refused(self, data, state):
        """A stored NaN raises, in every form."""
        with pytest.raises(ValueError, match="NaN entries"):
            Operator(data, state, state)

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("x", "x"),
            (5, "5"),
            (0, "0"),
            (1.5, "1.5"),
            ((1, 2), "(1, 2)"),
            (None, None),
        ],
        ids=["str", "int", "zero", "float", "tuple", "None"],
    )
    def test_name_is_a_string_or_none(self, name, expected, state):
        """A name that is not None becomes a string."""
        op = Operator(np.eye(3), state, state, name=name)

        assert op.name == expected
        assert type(op.name) is type(expected)

    def test_repr(self, H):
        """The repr gives the name, shape and form, then each space."""
        text = repr(H)

        assert "Operator(name='H', shape=(2, 3), form=DenseOperator)" in text
        assert "rows: Space(co2: 2 x ['obs_id'])" in text
        assert "cols: Space(flux: 3 x ['cell'])" in text


# What a caller hands an operator, and how to change it in place afterwards
GIVEN = {
    "dense": (lambda: np.arange(9.0).reshape(3, 3) + 1.0, lambda g: g.fill(-1.0)),
    "sparse": (
        lambda: sp.csr_array(np.triu(np.ones((3, 3)))),
        lambda g: g.data.fill(-1.0),
    ),
    "sparse_diagonal": (lambda: sp.eye_array(3, format="csr"), lambda g: g.data.fill(-1.0)),
}  # fmt: skip


@pytest.mark.parametrize("case", GIVEN)
class TestOwnsItsData:
    """An operator copies its matrix and makes the copy read-only, as a Vector does."""

    def test_editing_the_input_changes_nothing(self, case, state):
        """Writing into the array or matrix handed over leaves the operator as it was."""
        make, edit = GIVEN[case]
        given = make()
        op = Operator(given, state, state)
        before = op.dense()
        edit(given)
        np.testing.assert_array_equal(op.dense(), before)
        np.testing.assert_array_equal(np.asarray(op), before)

    def test_asarray_is_read_only_and_array_a_copy(self, case, state):
        """``np.asarray(op)`` cannot be written into (a dense matrix); ``np.array(op)`` and ``dense()`` are independent copies."""
        make, _ = GIVEN[case]
        op = Operator(make(), state, state)
        before = op.dense()
        if case == "dense":
            buffer = np.asarray(op)
            assert not buffer.flags.writeable
            with pytest.raises(ValueError, match="read-only"):
                buffer[0, 0] = 99.0
        for copy in (np.array(op), op.dense(), np.array(op.data)):
            assert copy.flags.writeable
            copy[0, 0] = 99.0
        np.testing.assert_array_equal(op.dense(), before)

    def test_what_comes_from_it_is_read_only_too(self, case, state):
        """A transpose, a product, a sum, a scalar multiple and a selection hold read-only arrays."""
        make, _ = GIVEN[case]
        op = Operator(make(), state, state)
        for derived in (op.T, op @ op, op + op, 2.0 * op, op["flux", "flux"]):
            held = derived.data
            if case == "dense" or isinstance(held, DenseOperator):
                assert not np.asarray(held).flags.writeable
            elif isinstance(held, SparseOperator):
                assert not held.matrix.data.flags.writeable
            else:
                assert not held.values.flags.writeable

    def test_a_pickle_is_read_only_too(self, case, state):
        """A pickle keeps the values, not the flag: the operator sets it again."""
        make, _ = GIVEN[case]
        op = Operator(make(), state, state)
        again = pickle.loads(pickle.dumps(op))
        held = again.data
        array = (
            held.array
            if isinstance(held, DenseOperator)
            else held.matrix.data
            if isinstance(held, SparseOperator)
            else held.values
        )
        assert not array.flags.writeable


class TestOwnsItsDataFromOtherWays:
    """The other ways in copy too: from_frame, from_blocks, and a form handed over."""

    def test_from_frame(self, obs, state):
        """Changing the frame afterwards leaves the operator as it was."""
        frame = pd.DataFrame(
            np.arange(6.0).reshape(2, 3), index=obs.index.droplevel("block"),
            columns=state.index.droplevel("block"),
        )  # fmt: skip
        op = Operator.from_frame(frame, row_block="co2", col_block="flux")
        before = op.dense()
        frame.iloc[:, :] = -1.0
        np.testing.assert_array_equal(op.dense(), before)
        assert not np.asarray(op).flags.writeable

    def test_from_blocks_unlabeled_and_tuple(self, obs, state):
        """An array given as a block, bare or with its labels, is copied."""
        bare = np.arange(6.0).reshape(2, 3)
        labeled = bare.copy()
        rows, cols = obs.select("co2"), state.select("flux")
        one = Operator.from_blocks({("co2", "flux"): bare}, rows=obs, cols=state)
        two = Operator.from_blocks({("co2", "flux"): (labeled, rows, cols)})
        before = one.dense()
        bare.fill(-1.0)
        labeled.fill(-1.0)
        np.testing.assert_array_equal(one.dense(), before)
        np.testing.assert_array_equal(two.dense(), before)

    def test_from_blocks_several_blocks(self, obs, state):
        """Blocks put together in one matrix: the blocks given can change afterwards."""
        a, b = np.ones((2, 3)), np.full((2, 1), 2.0)
        bias = Space({"bias": pd.Index(["x"], name="site")})
        cols = Space({"flux": state.select("flux"), "bias": bias.select("bias")})
        op = Operator.from_blocks(
            {("co2", "flux"): a, ("co2", "bias"): b}, rows=obs, cols=cols
        )
        before = op.dense()
        a.fill(-1.0)
        b.fill(-1.0)
        np.testing.assert_array_equal(op.dense(), before)
        assert not np.asarray(op).flags.writeable

    def test_a_form_is_kept_not_copied(self, state):
        """A form already holds its own copy, so an operator takes it as it is."""
        form = DenseOperator(np.eye(3))
        op = Operator(form, state, state)
        assert op.data is form and np.asarray(op) is form.array

    def test_a_form_made_from_an_operator_shares_its_array(self, H):
        """The transpose holds the parent's read-only array, not a copy."""
        assert np.shares_memory(H.T.data.array, H.data.array)
        assert H.round_labels(2).data is H.data


class TestFromFrame:
    """Reading an operator's labels from a DataFrame."""

    def test_axes_without_block_level_need_block_names(self, H):
        """Each axis without a block level is the one block it is named, its labels in their order."""
        op = Operator.from_frame(bare(H), row_block="co2", col_block="flux", name="G")

        assert op.rows == H.rows and op.cols == H.cols
        assert op.name == "G"
        np.testing.assert_array_equal(op.dense(), H.dense())

    def test_missing_block_name_raises(self, H):
        """An axis without a block level and without a block name raises, naming the argument."""
        with pytest.raises(ValueError, match="`col_block` must name"):
            Operator.from_frame(bare(H), row_block="co2")

    def test_block_name_with_a_block_level_raises(self, H):
        """A block level already names the blocks, so a block name is refused."""
        with pytest.raises(ValueError, match="leave `row_block` out"):
            Operator.from_frame(H.to_frame(), row_block="co2")

    def test_rows_of_each_block_are_put_together(self):
        """Blocks interleaved in the frame come out contiguous, in order of first appearance."""
        index = pd.MultiIndex.from_tuples(
            [("x", 1), ("y", 1), ("x", 2)], names=["block", "i"]
        )
        frame = pd.DataFrame(np.arange(9.0).reshape(3, 3), index=index, columns=index)

        op = Operator.from_frame(frame)

        assert op.rows.names == ("x", "y")
        np.testing.assert_array_equal(
            op.dense(), [[0.0, 2.0, 1.0], [6.0, 8.0, 7.0], [3.0, 5.0, 4.0]]
        )
        pd.testing.assert_frame_equal(
            by_label(op), frame.sort_index().sort_index(axis=1)
        )

    def test_round_trip(self):
        """A ragged operator read back from its frame has the same spaces and matrix."""
        op = Operator.from_blocks(ragged_blocks())

        again = Operator.from_frame(op.to_frame())

        assert again.rows == op.rows
        assert again.cols == op.cols
        np.testing.assert_array_equal(again.dense(), op.dense())

    def test_frame_is_copied(self, H):
        """Changing the frame later does not change the operator."""
        frame = bare(H)
        op = Operator.from_frame(frame, row_block="co2", col_block="flux")

        frame.iloc[0, 0] = 999.0

        assert op.dense()[0, 0] == 0.0

    @pytest.mark.parametrize("axis", ["index", "columns"])
    def test_duplicate_labels_refused(self, axis):
        """A label twice in a block raises, naming it (the Space refuses it)."""
        labels = pd.Index(["a", "b", "a"], name="cell")
        other = pd.Index(["u", "v", "w"], name="other")
        index, columns = (labels, other) if axis == "index" else (other, labels)
        frame = pd.DataFrame(np.eye(3), index=index, columns=columns)

        with pytest.raises(ValueError, match="duplicate labels: 'a'"):
            Operator.from_frame(frame, row_block="r", col_block="c")

    def test_nan_entry_refused(self, H):
        """A NaN in the frame raises."""
        frame = bare(H)
        frame.iloc[1, 2] = np.nan

        with pytest.raises(ValueError, match="NaN entries"):
            Operator.from_frame(frame, row_block="co2", col_block="flux")

    def test_not_a_frame_refused(self):
        """A Series is not a matrix."""
        with pytest.raises(TypeError, match="takes a DataFrame"):
            Operator.from_frame(pd.Series([1.0], index=pd.Index([0], name="i")))

    @pytest.mark.parametrize("fill", [0.0, np.nan])
    def test_sparse_columns_stay_sparse(self, fill):
        """Pandas-sparse columns are read as CSR; the entries they do not store are zero."""
        labels = pd.Index(["a", "b", "c"], name="cell")
        dense = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
        frame = pd.DataFrame(dense, index=labels, columns=labels).astype(
            pd.SparseDtype(float, fill)
        )

        op = Operator.from_frame(frame, row_block="x", col_block="x")

        assert type(op.data) is SparseOperator
        np.testing.assert_array_equal(op.dense(), dense)

    def test_sparse_fill_other_than_zero_refused(self):
        """A sparse column filled with another value would not be read as stored, so it raises."""
        labels = pd.Index(["a", "b"], name="cell")
        frame = pd.DataFrame(np.eye(2), index=labels, columns=labels).astype(
            pd.SparseDtype(float, 1.0)
        )

        with pytest.raises(ValueError, match="fill value 1.0"):
            Operator.from_frame(frame, row_block="x", col_block="x")


class TestFromBlocks:
    """Assembling an operator from its blocks, each named by its row and column block."""

    def test_ragged_blocks_keep_their_levels(self):
        """Blocks with different levels make spaces whose blocks keep their own labels."""
        op = Operator.from_blocks(ragged_blocks())

        assert op.rows.names == ("o1", "o2")
        assert op.cols.names == ("s1", "s2")
        for name, labels in ROWS.items():
            pd.testing.assert_index_equal(op.rows.select(name), labels)
        for name, labels in COLS.items():
            pd.testing.assert_index_equal(op.cols.select(name), labels)
        expected = np.block(
            [
                [np.full((2, 2), 1.0), np.full((2, 2), 2.0)],
                [np.full((2, 2), 3.0), np.full((2, 2), 4.0)],
            ]
        )
        np.testing.assert_array_equal(op.dense(), expected)

    def test_missing_pairs_are_zero(self):
        """A pair that is not given is a block of zeros."""
        blocks = ragged_blocks()
        del blocks["o1", "s2"], blocks["o2", "s1"]

        op = Operator.from_blocks(blocks)

        np.testing.assert_array_equal(op["o1", "s2"].dense(), np.zeros((2, 2)))
        np.testing.assert_array_equal(op["o2", "s1"].dense(), np.zeros((2, 2)))
        np.testing.assert_array_equal(op["o2", "s2"].dense(), np.full((2, 2), 4.0))

    def test_pair_given_twice_raises(self, H):
        """Keys that name the same pair once converted to strings are one pair."""
        with pytest.raises(ValueError, match=r"\('co2', '1'\) is given twice"):
            Operator.from_blocks({("co2", 1): bare(H), ("co2", "1"): bare(H)})

    def test_unlabeled_blocks_take_the_given_spaces(self, obs, state):
        """Arrays, sparse matrices and forms take their labels from `rows` and `cols`."""
        bias = Space({"bias": pd.Index(["A"], name="site")})
        cols = Space({"flux": state.select("flux"), "bias": bias.select("bias")})

        op = Operator.from_blocks(
            {
                ("co2", "flux"): np.ones((2, 3)),
                ("co2", "bias"): sp.csr_array(np.ones((2, 1))),
            },
            rows=obs,
            cols=cols,
        )

        assert op.rows is obs and op.cols is cols
        np.testing.assert_array_equal(op.dense(), np.ones((2, 4)))

    def test_unlabeled_block_takes_labels_from_its_row_and_column(self, H):
        """An array's labels can come from labeled blocks in the same row and column blocks."""
        bias = pd.DataFrame(
            2.0,
            index=pd.Index([9], name="obs_id"),
            columns=pd.Index(["A"], name="site"),
        )

        op = Operator.from_blocks(
            {
                ("co2", "flux"): bare(H),
                ("ch4", "bias"): bias,
                ("co2", "bias"): np.full((2, 1), 3.0),
            }
        )

        np.testing.assert_array_equal(op["co2", "bias"].dense(), np.full((2, 1), 3.0))
        np.testing.assert_array_equal(op["ch4", "flux"].dense(), np.zeros((1, 3)))

    def test_unlabeled_block_without_labels_raises(self, state):
        """A block whose labels are found nowhere raises, naming it."""
        with pytest.raises(ValueError, match="Block 'co2' has no labels"):
            Operator.from_blocks({("co2", "flux"): np.ones((2, 3))}, cols=state)

    def test_unlabeled_block_of_the_wrong_shape_raises(self, obs, state):
        """An array must have one row and one column per label of its blocks."""
        with pytest.raises(ValueError, match=r"has shape \(3, 2\)"):
            Operator.from_blocks(
                {("co2", "flux"): np.ones((3, 2))}, rows=obs, cols=state
            )

    def test_labeled_blocks_are_put_in_the_given_order(self, H, obs, state):
        """A frame is aligned by label to the given spaces."""
        cols = reordered(state)

        op = Operator.from_blocks({("co2", "flux"): bare(H)}, rows=obs, cols=cols)

        assert op.cols is cols
        np.testing.assert_array_equal(op.dense(), H.dense()[:, ::-1])

    def test_one_block_with_two_sets_of_labels_raises(self, H, other_state):
        """Two frames that give one block different labels do not join."""
        ch4 = pd.DataFrame(
            1.0, index=pd.Index([9], name="obs_id"), columns=other_state.select("flux")
        )

        with pytest.raises(ValueError, match="'z'"):
            Operator.from_blocks({("co2", "flux"): bare(H), ("ch4", "flux"): ch4})

    def test_operator_on_other_blocks_raises(self, H):
        """An Operator must be on the one row block and one column block its key names."""
        with pytest.raises(ValueError, match="must be on the one block its key names"):
            Operator.from_blocks({("co2", "bias"): H})

    def test_operator_blocks(self, H):
        """Operators are blocks too, and keep their values."""
        doubled = Operator(
            2.0 * H.dense(), H.rows, Space({"bias": H.cols.select("flux")})
        )

        op = Operator.from_blocks({("co2", "flux"): H, ("co2", "bias"): doubled})

        np.testing.assert_array_equal(
            op.dense(), np.hstack([H.dense(), 2.0 * H.dense()])
        )

    @pytest.mark.parametrize(
        ("bias", "form"),
        [(DiagonalOperator([2.0]), SparseOperator), (np.array([[2.0]]), DenseOperator)],
        ids=["all_sparse", "one_dense"],
    )
    def test_sparse_only_when_every_block_is_sparse(self, bias, form, state):
        """Sparse and diagonal blocks make a sparse operator; one dense block, a dense one."""
        space = Space(
            {"flux": state.select("flux"), "bias": pd.Index(["A"], name="site")}
        )
        flux = sp.csr_array(np.triu(np.ones((3, 3))))

        op = Operator.from_blocks(
            {("flux", "flux"): flux, ("bias", "bias"): bias}, rows=space, cols=space
        )

        assert type(op.data) is form
        expected = np.zeros((4, 4))
        expected[:3, :3], expected[3, 3] = np.triu(np.ones((3, 3))), 2.0
        np.testing.assert_array_equal(op.dense(), expected)

    def test_a_labeled_block_on_part_of_its_labels_is_placed(self, obs, state):
        """A frame or tuple with some of a block's labels: the others are zero."""
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        part = pd.Index(["b", "c"], name="cell")
        for block, sparse in (
            (pd.DataFrame(values, index=obs.select("co2"), columns=part), False),
            ((sp.csr_array(values), obs.select("co2"), part), True),
            ((values, obs.select("co2"), part), False),
        ):
            op = Operator.from_blocks({("co2", "flux"): block}, rows=obs, cols=state)
            expected = np.zeros((2, 3))
            expected[:, [2, 0]] = values  # the state is c, a, b
            np.testing.assert_array_equal(op.dense(), expected)
            assert isinstance(op.data, SparseOperator) == sparse

    def test_a_label_the_space_lacks_raises(self, obs, state):
        """A labeled block may cover part of its block, never more: the extra label is named."""
        block = (np.ones((2, 1)), obs.select("co2"), pd.Index(["z"], name="cell"))
        with pytest.raises(ValueError, match="labels the block does not have: 'z'"):
            Operator.from_blocks({("co2", "flux"): block}, rows=obs, cols=state)

    def test_a_tuple_of_another_length_raises(self, obs, state):
        with pytest.raises(ValueError, match=r"\(matrix, row_labels, col_labels\)"):
            Operator.from_blocks(
                {("co2", "flux"): (np.ones((2, 3)),)}, rows=obs, cols=state
            )

    def test_a_reordered_block_keeps_its_form(self, state):
        """A labeled block that covers its labels in another order keeps its form."""
        form = DiagonalOperator([1.0, 2.0, 3.0])
        given = Operator(form, reordered(state), reordered(state))
        op = Operator.from_blocks({("flux", "flux"): given}, rows=state, cols=state)
        assert isinstance(op.data, DiagonalOperator)
        np.testing.assert_array_equal(op.dense(), np.diag([3.0, 2.0, 1.0]))

    def test_one_block_keeps_its_form(self, state):
        """A single block that covers both spaces is the operator's form, as given."""
        form = DiagonalOperator([1.0, 2.0, 3.0])

        op = Operator.from_blocks(
            {("flux", "flux"): form}, rows=state, cols=state, name="S"
        )

        assert op.data is form
        assert op.name == "S"


class TestBlockSelection:
    """A block taken back out with ``op[r, c]`` has the labels it was given."""

    @pytest.mark.parametrize("r", list(ROWS))
    @pytest.mark.parametrize("c", list(COLS))
    def test_block_keeps_its_own_labels(self, r, c):
        """The levels that are NaN-only for a block are not on it; values and dtypes are as given."""
        given = ragged_blocks()[r, c]
        op = Operator.from_blocks(ragged_blocks())

        block = op[r, c]

        assert block.rows.names == (r,) and block.cols.names == (c,)
        pd.testing.assert_index_equal(block.rows.select(r), given.index)
        pd.testing.assert_index_equal(block.cols.select(c), given.columns)
        pd.testing.assert_frame_equal(bare(block), given)

    def test_padded_index_keeps_each_level_dtype(self):
        """The whole operator's index pads levels with NaN, and each level keeps its dtype."""
        frame = Operator.from_blocks(ragged_blocks()).to_frame()

        assert frame.index.names == ["block", "t", "site", "k"]
        assert frame.columns.names == ["block", "lat", "lon", "param"]
        levels = dict(zip(frame.index.names, frame.index.levels, strict=True))
        assert levels["t"].dtype == np.int64 and levels["k"].dtype == np.int64
        columns = dict(zip(frame.columns.names, frame.columns.levels, strict=True))
        assert columns["lat"].dtype == np.float64 and columns["lon"].dtype == np.int64
        assert frame.index.get_level_values("k")[:2].isna().all()
        assert frame.columns.get_level_values("param")[:2].isna().all()

    def test_unknown_block_raises(self, H):
        """A block the space lacks raises a KeyError naming the blocks it has."""
        with pytest.raises(KeyError, match="the blocks are 'flux'"):
            H["co2", "bias"]

    def test_covariance_blocks(self, state):
        """A covariance's diagonal block is a Covariance; an off-diagonal block an Operator."""
        space = Space(
            {"flux": state.select("flux"), "bias": pd.Index(["A"], name="site")}
        )
        cov = Covariance(np.eye(4), space)

        assert type(cov["flux", "flux"]) is Covariance
        assert type(cov["flux", "bias"]) is Operator


class TestToFrame:
    """The matrix as a dense array or a labeled frame."""

    def test_block_level_first(self, H):
        """Both axes have the block level first, then the block's levels."""
        frame = H.to_frame()

        assert frame.index.names == ["block", "obs_id"]
        assert frame.columns.names == ["block", "cell"]
        pd.testing.assert_index_equal(frame.index, H.rows.index)
        pd.testing.assert_index_equal(frame.columns, H.cols.index)
        np.testing.assert_array_equal(frame.to_numpy(), H.dense())

    def test_dense_is_a_new_array(self, H):
        """Changing what dense() returns does not change the operator."""
        values = H.dense()
        values[0, 0] = 999.0

        assert H.dense()[0, 0] == 0.0

    def test_asarray(self, state):
        """NumPy takes an operator as its dense matrix."""
        op = Operator(DiagonalOperator([1.0, 2.0, 3.0]), state, state)

        np.testing.assert_array_equal(np.asarray(op), np.diag([1.0, 2.0, 3.0]))
        assert np.trace(op) == 6.0


class TestMatmul:
    """The product ``op @ other`` joins op's columns and other's rows exactly, by label."""

    def test_product_by_label(self, H, state):
        """The other's rows may be in any order; the product is by label."""
        other_cols = Space({"out": pd.Index(["u", "v"], name="w")})
        G_frame = pd.DataFrame(
            np.arange(6.0).reshape(3, 2),
            index=state.select("flux"),
            columns=other_cols.select("out"),
        )
        G = Operator.from_frame(
            G_frame.iloc[[2, 0, 1]], row_block="flux", col_block="out"
        )

        product = H @ G

        assert isinstance(product, Operator)
        assert product.rows is H.rows and product.cols == other_cols
        np.testing.assert_array_equal(product.dense(), H.dense() @ G_frame.to_numpy())

    @pytest.mark.parametrize(
        ("labels", "named"),
        [
            ({"flux": pd.Index(["c", "a", "z"], name="cell")}, ["'z'", "'b'"]),
            ({"bias": pd.Index(["c", "a", "b"], name="cell")}, ["'bias'", "'flux'"]),
        ],
        ids=["labels", "blocks"],
    )
    def test_mismatch_raises_naming_it(self, labels, named, H):
        """A label or block on one side only raises, naming what is missing and what is extra."""
        G = Operator(
            np.ones((3, 1)), Space(labels), Space({"out": pd.Index([0], name="w")})
        )

        with pytest.raises(ValueError, match="do not have the same labels") as error:
            H @ G
        assert all(name in str(error.value) for name in named)

    def test_vector_in_any_order(self, H, state):
        """A vector is aligned to the columns, and the result is a vector on the rows."""
        x = Vector([1.0, 2.0, 3.0], state)
        flipped = Vector([3.0, 2.0, 1.0], reordered(state))

        for vector in (x, flipped):
            y = H @ vector
            assert isinstance(y, Vector)
            assert y.space is H.rows
            np.testing.assert_array_equal(y.values, H.dense() @ x.values)

    def test_array_refused(self, H):
        """An array has no labels to check, so it is refused, naming op.data."""
        with pytest.raises(TypeError, match=r"op\.data @ x"):
            H @ np.ones(3)

    def test_a_vector_missing_a_column_raises(self, H, state):
        """A vector must have exactly the columns' labels: a missing one is named, not zero."""
        block = state.names[0]
        short = Space({block: state.select(block)[:2]})
        with pytest.raises(ValueError, match="missing from the other space"):
            H @ Vector([1.0, 2.0], short)

    def test_multiindex_block_labels(self):
        """A block labeled by several levels is matched on all of them."""
        labels = pd.MultiIndex.from_tuples(
            [("sub1", "a"), ("sub1", "b"), ("sub2", "a")], names=["subblock", "element"]
        )
        obs = Space({"obs": pd.Index(["o1", "o2"], name="obs")})
        H = Operator(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), obs, Space({"state": labels})
        )
        x = Vector([30.0, 20.0, 10.0], Space({"state": labels[::-1]}))
        np.testing.assert_array_equal((H @ x).values, [140.0, 320.0])


class TestSum:
    """Sums and differences join both spaces exactly, by label."""

    def test_sum_and_difference_by_label(self, H, obs, state):
        """The other operator may have its labels in any order."""
        other = H.align(reordered(obs), reordered(state)) * 2.0

        np.testing.assert_array_equal((H + other).dense(), 3.0 * H.dense())
        np.testing.assert_array_equal((H - other).dense(), -H.dense())
        assert (H + other).rows is H.rows

    def test_mismatched_labels_raise(self, H, obs, other_state):
        """A missing or extra label raises, naming it."""
        with pytest.raises(ValueError, match="'z'"):
            H - Operator(np.ones((2, 3)), obs, other_state)

    @pytest.mark.parametrize("other", [1.0, np.ones((2, 3))], ids=["scalar", "array"])
    def test_only_operators_add(self, H, other):
        """A scalar or array has no labels to join."""
        with pytest.raises(TypeError):
            H + other
        with pytest.raises(TypeError, match="adds an Operator"):
            H - other

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            ("S", "S", "S"),
            ("S", "T", None),
            ("S", None, None),
            (None, None, None),
            (5, "5", "5"),
        ],
        ids=["same", "different", "one_unnamed", "unnamed", "string_compared"],
    )
    def test_sum_name(self, left, right, expected, state):
        """A sum keeps a name both share, as strings, and is unnamed otherwise."""
        a = Operator(np.eye(3), state, state, name=left)
        b = Operator(np.eye(3), state, state, name=right)

        assert (a + b).name == expected

    def test_covariances_sum_to_a_covariance(self, state):
        """Two covariances add up to a covariance; a covariance and an operator to an operator."""
        cov = Covariance.diagonal(1.0, state)

        assert type(cov + cov) is Covariance
        assert type(cov + Operator(np.eye(3), state, state)) is Operator


class TestScalarArithmetic:
    """Multiplying, dividing and negating keep the form, the labels and the name."""

    FORMS = [
        pytest.param(np.ones((3, 3)), DenseOperator, id="dense"),
        pytest.param(
            sp.csr_array(np.triu(np.ones((3, 3)))), SparseOperator, id="sparse"
        ),
        pytest.param(
            DiagonalOperator([1.0, 1.0, 1.0]), DiagonalOperator, id="diagonal"
        ),
    ]

    @pytest.mark.parametrize(("data", "form"), FORMS)
    @pytest.mark.parametrize("scalar", SCALARS, ids=lambda s: type(s).__name__)
    def test_operations_keep_the_form(self, data, form, scalar, state):
        """Multiplying, dividing and negating keep the form, labels and name, and leave op as it was."""
        op = Operator(data, state, state, name="S")
        before = op.dense()

        for result, factor in ((op * scalar, 5.0), (op / scalar, 0.2), (-op, -1.0)):
            assert type(result.data) is form
            assert result.rows is state and result.cols is state
            assert result.name == "S"
            np.testing.assert_allclose(result.dense(), factor * before)
        np.testing.assert_array_equal(op.dense(), before)

    @pytest.mark.parametrize("scalar", [5.0, 5], ids=["float", "int"])
    def test_scalar_on_the_left(self, scalar, H):
        """A Python scalar on the left gives the same operator."""
        result = scalar * H

        assert isinstance(result, Operator)
        np.testing.assert_array_equal(result.dense(), 5.0 * H.dense())

    @pytest.mark.parametrize(
        "scalar", [np.float64(5.0), np.array(5.0)], ids=["float64", "0-d"]
    )
    def test_numpy_scalar_on_the_left(self, scalar, H):
        """A NumPy scalar on the left gives an operator, as on the right."""
        assert isinstance(scalar * H, Operator)

    def test_array_plus_operator_refused(self, H):
        """An array on the left of + or @ has no labels to join, so it is refused."""
        with pytest.raises(TypeError):
            np.ones((2, 3)) + H
        with pytest.raises(TypeError):
            np.ones((2, 2)) @ H

    def test_covariance_stays_a_covariance(self, state):
        """Scaling or negating a covariance gives a covariance."""
        cov = Covariance.diagonal(2.0, state, name="S")

        for result in (cov * 3.0, 3.0 * cov, cov / 2.0, -cov):
            assert type(result) is Covariance
            assert type(result.data) is DiagonalOperator
            assert result.name == "S"

    def test_array_factor_refused(self, H):
        """An array is not a scalar: '*' and '/' refuse it."""
        with pytest.raises(TypeError, match="use '@'"):
            H * np.ones(3)
        with pytest.raises(TypeError, match="only be divided by a scalar"):
            H / np.ones(3)


class TestSparseStaysSparse:
    """Transposes, products, sums and block selections of sparse operators stay sparse."""

    @pytest.mark.parametrize(
        "operation",
        [lambda A: A.T, lambda A: A @ A, lambda A: A + A, lambda A: A["x", "x"]],
        ids=["transpose", "product", "sum", "block"],
    )
    def test_form(self, operation):
        """A sparse operator's results are sparse, with the numbers of the dense ones."""
        space = Space({"x": pd.RangeIndex(10, name="i")})
        A = Operator(sp.eye_array(10, k=1, format="csr"), space, space)
        dense = Operator(A.dense(), space, space)

        result = operation(A)

        assert type(result.data) is SparseOperator
        np.testing.assert_array_equal(result.dense(), operation(dense).dense())


class TestTranspose:
    """The transpose swaps the spaces with the matrix."""

    def test_spaces_swap(self, H):
        """Rows become columns, labels and all; the name stays."""
        T = H.T

        assert T.rows is H.cols and T.cols is H.rows
        assert T.name == "H"
        np.testing.assert_array_equal(T.dense(), H.dense().T)
        pd.testing.assert_frame_equal(T.to_frame(), H.to_frame().T)


class TestAlign:
    """Putting an operator on spaces with the same labels in another order."""

    def test_reorders_by_label(self, H, obs, state):
        """The matrix follows its labels, and the name stays."""
        aligned = H.align(reordered(obs), reordered(state))

        assert aligned.rows == reordered(obs) and aligned.cols == reordered(state)
        assert aligned.name == "H"
        np.testing.assert_array_equal(aligned.dense(), H.dense()[::-1, ::-1])
        pd.testing.assert_frame_equal(by_label(aligned), by_label(H))

    def test_other_labels_raise(self, H, obs, other_state):
        """A space with other labels raises, naming the missing and the extra ones."""
        with pytest.raises(ValueError, match="'z'") as error:
            H.align(obs, other_state)
        assert "'b'" in str(error.value)

    def test_covariance_on_one_space(self, state):
        """A covariance aligned to one space is a covariance; to two different ones, an operator."""
        cov = Covariance(
            np.array([[2.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 3.0]]), state
        )
        flipped = reordered(state)

        assert type(cov.align(flipped, flipped)) is Covariance
        assert type(cov.align(flipped, state)) is Operator


class TestRoundLabels:
    """Rounding the float labels of both axes."""

    def test_rounds_both_axes(self, state):
        """Float levels are rounded on rows and columns; other levels are kept."""
        floats = Space({"obs": pd.Index([0.11, 0.26], name="obs_id")})
        op = Operator(np.ones((2, 3)), floats, state, name="H")

        rounded = op.round_labels(1)

        assert rounded.rows.select("obs").to_list() == [0.1, 0.3]
        assert rounded.cols == state and rounded.name == "H"
        assert op.rows is floats
        np.testing.assert_array_equal(rounded.dense(), op.dense())

    def test_merged_labels_raise(self):
        """Rounding two labels of a block to one raises."""
        space = Space({"obs": pd.Index([0.11, 0.12], name="obs_id")})

        with pytest.raises(ValueError, match="merges labels"):
            Operator(np.eye(2), space, space).round_labels(1)

    def test_covariance_stays_a_covariance(self):
        """A covariance's rounded rows are its rounded columns."""
        space = Space({"flux": pd.Index([0.1234, 0.5678], name="lat")})

        rounded = Covariance(np.eye(2), space).round_labels(2)

        assert type(rounded) is Covariance
        assert rounded.rows is rounded.cols


class TestPickle:
    """An operator pickles with its form, spaces and name."""

    @pytest.mark.parametrize(
        "data",
        [
            np.arange(9.0).reshape(3, 3),
            sp.csr_array(np.triu(np.ones((3, 3)))),
            DiagonalOperator([1.0, 2.0, 3.0]),
        ],
        ids=["dense", "sparse", "diagonal"],
    )
    def test_round_trip(self, data, state):
        """The operator comes back with its form, labels and name."""
        op = Operator(data, state, reordered(state), name="S")

        again = pickle.loads(pickle.dumps(op))

        assert type(again.data) is type(op.data)
        assert again.rows == op.rows and again.cols == op.cols
        assert again.name == "S"
        np.testing.assert_array_equal(again.dense(), op.dense())

    def test_covariance_round_trip(self, state):
        """A covariance comes back a covariance, in its form."""
        cov = Covariance.diagonal([1.0, 2.0, 3.0], state, name="S")

        again = pickle.loads(pickle.dumps(cov))

        assert type(again) is Covariance and type(again.data) is DiagonalOperator
        assert again.space == state and again.name == "S"

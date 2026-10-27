"""
Tests for sparse matrices: pandas-sparse frames read in, SciPy matrices kept through the work.

A sparse operator holds a SciPy sparse matrix (:class:`fips.linalg.SparseOperator`)
or a diagonal (:class:`fips.linalg.DiagonalOperator`). A pandas-sparse DataFrame
is a way in: :meth:`fips.Operator.from_frame` reads it as CSR, the entries a
sparse column does not store being zeros.
"""

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import fips
from fips import Covariance, Operator, Problem, Space, Vector
from fips.linalg import (
    BlockDiagonalOperator,
    DenseOperator,
    DiagonalOperator,
    SparseOperator,
)

# ---------------------------------------------------------------------------
# Pandas-sparse frames
# ---------------------------------------------------------------------------

_PAIR = pd.Index(["a", "b"], name="site")
_CELLS = np.array([[1.0, 5.0], [1.0, 1.0]])


def _from_spmatrix(matrix, index, columns) -> pd.DataFrame:
    """
    Build a sparse frame the ordinary pandas way.

    pandas 3's ``from_spmatrix`` fills a float frame with NaN
    (pandas-dev/pandas#59212), not with the zeros SciPy means.
    """
    return pd.DataFrame.sparse.from_spmatrix(matrix, index=index, columns=columns)


def _filled_frame(fills) -> pd.DataFrame:
    """
    Build `_CELLS` as a frame whose column j is sparse with fill ``fills[j]``.

    A fill of None leaves the column dense. A column stores the entries that
    differ from its fill, so with a fill of 1.0 only the 5.0 is stored.
    """
    data = {
        j: _CELLS[:, j]
        if fill is None
        else pd.arrays.SparseArray(_CELLS[:, j], fill_value=fill)
        for j, fill in enumerate(fills)
    }
    frame = pd.DataFrame(data, index=_PAIR)
    frame.columns = _PAIR
    return frame


def _nan_filled_frame(stored_nan: bool) -> pd.DataFrame:
    """
    Build a sparse frame with ``fill_value=NaN`` that stores rows 0 and 1.

    Column "x" can store a NaN in row 1 when `stored_nan` is set; row 2 is
    implicit. Pandas drops a NaN while building a NaN-filled array, so each
    column is built with fill value 0, which stores rows 0 and 1 whatever they
    hold (NaN included), and then relabeled with ``fill_value=NaN``: a
    SparseArray built from a SparseArray keeps its stored positions and values.
    """

    def column(stored: list[float]) -> pd.arrays.SparseArray:
        zero_filled = pd.arrays.SparseArray([*stored, 0.0], fill_value=0.0)
        return pd.arrays.SparseArray(zero_filled, fill_value=np.nan)

    frame = pd.DataFrame(
        {
            "x": column([1.0, np.nan] if stored_nan else [1.0, 2.0]),
            "y": column([4.0, 5.0]),
        },
        index=pd.Index(["o0", "o1", "o2"], name="obs_id"),
    )
    frame.columns.name = "cell"
    return frame


def _read(frame: pd.DataFrame) -> Operator:
    """Read a frame without block levels as one block on each axis."""
    return Operator.from_frame(frame, row_block="r", col_block="c")


def _read_covariance(frame: pd.DataFrame) -> Covariance:
    """Read a square frame without block levels as a covariance of one block."""
    return Covariance.from_frame(frame, row_block="s", col_block="s")


class TestFromFrame:
    """``Operator.from_frame`` reads a pandas-sparse frame as CSR, implicit entries as zeros."""

    def test_from_spmatrix_frame(self):
        """A frame from ``from_spmatrix`` is read as the matrix it was built from, NaN fill and all."""
        matrix = sp.random(30, 20, density=0.05, format="csr", random_state=0)
        frame = _from_spmatrix(
            matrix,
            pd.Index([f"o{i}" for i in range(30)], name="obs_id"),
            pd.Index([f"s{j}" for j in range(20)], name="cell"),
        )
        assert all(isinstance(dt, pd.SparseDtype) for dt in frame.dtypes)
        op = _read(frame)
        assert isinstance(op.data, SparseOperator)
        assert op.data.matrix.nnz == matrix.nnz
        np.testing.assert_array_equal(op.dense(), matrix.toarray())

    @pytest.mark.parametrize(
        "fills",
        [(0.0, 0.0), (0, 0), (np.nan, np.nan), (np.nan, 0.0), (0.0, np.nan)],
        ids=["zero", "int-zero", "nan", "nan-then-zero", "zero-then-nan"],
    )
    def test_zero_and_nan_fills_are_zeros(self, fills):
        """A fill of 0 or NaN stands for zeros: the same cells either way."""
        op = _read(_filled_frame(fills))
        assert isinstance(op.data, SparseOperator)
        np.testing.assert_array_equal(op.dense(), _CELLS)

    def test_implicit_nan_entries_are_zeros(self):
        """The NaN fill means "no entry", next to a zero fill and a dense column."""
        frame = _nan_filled_frame(stored_nan=False)
        frame["z"] = pd.arrays.SparseArray([0.0, 7.0, 0.0], fill_value=0.0)
        frame["w"] = [0.0, 0.0, 8.0]
        op = _read(frame)
        assert isinstance(op.data, SparseOperator)
        np.testing.assert_array_equal(
            op.dense(),
            [[1.0, 4.0, 0.0, 0.0], [2.0, 5.0, 7.0, 0.0], [0.0, 0.0, 0.0, 8.0]],
        )
        assert op.data.matrix.nnz == 6  # the zeros of the dense column are not stored

    def test_a_column_that_stores_every_row(self):
        """A NaN-filled column with no implicit entry reads like the others."""
        frame = pd.DataFrame(
            {
                "a": pd.arrays.SparseArray([1.0, 3.0], fill_value=np.nan),
                "b": pd.arrays.SparseArray([np.nan, 4.0], fill_value=np.nan),
            },
            index=_PAIR,
        )
        frame.columns.name = "cell"
        np.testing.assert_array_equal(_read(frame).dense(), [[1.0, 0.0], [3.0, 4.0]])

    @pytest.mark.parametrize(
        "fills",
        [
            (1.0, 1.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (np.nan, 1.0),
            (1.0, np.nan),
            (-3.0, -3.0),
        ],
        ids=[
            "all-one",
            "zero-then-one",
            "one-then-zero",
            "nan-then-one",
            "one-then-nan",
            "negative",
        ],
    )
    def test_nonzero_fill_raises(self, fills):
        """A fill that is neither 0 nor NaN stands for entries that are not zeros: refused."""
        bad = next(f for f in fills if not np.isnan(f) and f != 0)
        frame = _filled_frame(fills)
        for build in (
            _read,
            _read_covariance,
            lambda f: Operator.from_blocks({("r", "c"): f}),
        ):
            with pytest.raises(ValueError, match=rf"fill value {bad!r}.*must be 0"):
                build(frame)

    def test_mixed_sparse_and_dense_columns(self):
        """A dense column beside a sparse one is read with it; a nonzero fill still raises."""
        frame = _filled_frame((None, 0.0))
        assert [str(dt) for dt in frame.dtypes] == ["float64", "Sparse[float64, 0.0]"]
        op = _read(frame)
        assert isinstance(op.data, SparseOperator)
        np.testing.assert_array_equal(op.dense(), _CELLS)
        with pytest.raises(ValueError, match="fill value 1.0"):
            _read(_filled_frame((None, 1.0)))

    def test_stored_nan_raises(self):
        """A NaN a sparse column stores is data, not an implicit zero: refused, as on the dense path."""
        with pytest.raises(ValueError, match="NaN entries"):
            _read(_nan_filled_frame(stored_nan=True))
        dense = np.diag([1.0, np.nan, 3.0])
        sites = pd.Index(["a", "b", "c"], name="site")
        for frame in (
            pd.DataFrame(dense, index=sites, columns=sites),
            _from_spmatrix(sp.csr_matrix(dense), sites, sites),
        ):
            with pytest.raises(ValueError, match="NaN entries"):
                _read_covariance(frame)
        mixed = _filled_frame((None, 0.0))
        mixed.iloc[0, 0] = np.nan  # in the dense column of a sparse frame
        with pytest.raises(ValueError, match="NaN entries"):
            _read(mixed)

    def test_diagonal_frame_is_a_diagonal(self):
        """A square sparse frame that stores its diagonal only is a DiagonalOperator."""
        sites = pd.Index(["a", "b", "c"], name="site")
        frame = _from_spmatrix(sp.diags_array([1.0, 2.0, 3.0]), sites, sites)
        op = _read(frame)
        assert isinstance(op.data, DiagonalOperator)
        np.testing.assert_array_equal(op.data.values, [1.0, 2.0, 3.0])
        # A covariance with its columns in another order is put in the rows' order
        S = _read_covariance(frame.iloc[:, ::-1])
        assert isinstance(S.data, DiagonalOperator)
        np.testing.assert_array_equal(S.variances.values, [1.0, 2.0, 3.0])

    def test_rows_of_a_block_are_put_together(self):
        """With a ``block`` level, a sparse frame's rows and columns are put in block order, still sparse."""
        rows = pd.MultiIndex.from_tuples(
            [("co2", 1), ("ch4", 1), ("co2", 2), ("ch4", 2), ("co2", 3)],
            names=["block", "obs_id"],
        )
        cols = pd.MultiIndex.from_tuples(
            [("bias", 0), ("flux", 0), ("flux", 1)], names=["block", "cell"]
        )
        values = np.array(
            [[0, 1, 0], [2, 0, 0], [0, 0, 3], [0, 4, 0], [5, 0, 0]], dtype=float
        )
        frame = _from_spmatrix(sp.csr_array(values), rows, cols)
        op = Operator.from_frame(frame)
        assert isinstance(op.data, SparseOperator)
        assert op.rows.names == ("co2", "ch4") and op.cols.names == ("bias", "flux")
        dense = pd.DataFrame(values, index=rows, columns=cols)
        np.testing.assert_array_equal(
            op.dense(), dense.loc[op.rows.index, op.cols.index].to_numpy()
        )


# ---------------------------------------------------------------------------
# Sparse forms stay sparse through Operator arithmetic and from_blocks
# ---------------------------------------------------------------------------

N, M = 80, 60
STATE = Space({"flux": pd.Index([f"c{k}" for k in range(N)], name="cell")})
OBS = Space({"co2": pd.Index([f"o{k}" for k in range(M)], name="obs_id")})


def _jacobian() -> np.ndarray:
    """A sparse ``H``: observation i sees cells i and i + 20 (``H S H^T`` fills about a tenth)."""
    H = np.zeros((M, N))
    for i in range(M):
        H[i, i], H[i, i + 20] = 1.0 + i / M, 0.5
    return H


def _banded(n: int, seed: int = 0) -> np.ndarray:
    """A diagonally dominant (so positive-definite) tridiagonal matrix."""
    rng = np.random.default_rng(seed)
    off = rng.uniform(-1, 1, n - 1)
    return np.diag(rng.uniform(3, 4, n)) + np.diag(off, 1) + np.diag(off, -1)


def _reversed(space: Space) -> Space:
    """The space of one block, its labels in reverse order."""
    (name,) = space.names
    return Space({name: space.select(name)[::-1]})


def _sparse_operators():
    """A sparse ``H``, a sparse prior error and a diagonal mismatch, with the arrays of the first two."""
    H, S_0 = _jacobian(), _banded(N)
    return (
        Operator(sp.csr_array(H), OBS, STATE),
        Covariance(sp.csr_array(S_0), STATE),
        Covariance.diagonal(0.5, OBS),
        H,
        S_0,
    )


class TestArithmetic:
    """Products, sums, scaling, transposes, selections and pickling keep a sparse form."""

    def test_products_and_sums(self):
        """``H S``, ``H S H^T + S_z`` and ``S + S`` stay sparse and hold the dense numbers."""
        H, S_0, S_z, Ha, Sa = _sparse_operators()
        HS = H @ S_0
        assert isinstance(HS.data, SparseOperator)
        np.testing.assert_allclose(HS.dense(), Ha @ Sa)
        G = H @ S_0 @ H.T + S_z
        assert isinstance(G.data, SparseOperator)
        assert G.rows == OBS and G.cols == OBS
        np.testing.assert_allclose(G.dense(), Ha @ Sa @ Ha.T + 0.5 * np.eye(M))
        for total, expected in [(S_0 + S_0, 2 * Sa), (S_0 - S_0 * 0.5, 0.5 * Sa)]:
            assert isinstance(total, Covariance)
            assert isinstance(total.data, SparseOperator)
            np.testing.assert_allclose(total.dense(), expected)
        # A dense side makes the sum dense
        assert isinstance((S_0 + Covariance(np.eye(N), STATE)).data, DenseOperator)

    def test_scaling_transpose_and_selection(self):
        """Scalars, ``.T``, ``align``, a block and pickling keep the sparse form."""
        H, S_0, _, Ha, Sa = _sparse_operators()
        for op, expected in [
            (S_0 * 2.0, 2.0 * Sa),
            (2.0 * S_0, 2.0 * Sa),
            (S_0 / 4.0, Sa / 4.0),
            (-S_0, -Sa),
            (H.T, Ha.T),
            (S_0["flux", "flux"], Sa),
            (pickle.loads(pickle.dumps(H)), Ha),
        ]:
            assert isinstance(op.data, SparseOperator)
            np.testing.assert_array_equal(op.dense(), expected)
        moved = H.align(_reversed(OBS), _reversed(STATE))
        assert isinstance(moved.data, SparseOperator)
        np.testing.assert_array_equal(moved.dense(), Ha[::-1, ::-1])

    def test_exact_join_in_another_order(self):
        """An operator whose labels are in another order is aligned, and the product stays sparse."""
        H, S_0, _, Ha, Sa = _sparse_operators()
        moved = S_0.align(_reversed(STATE), _reversed(STATE))
        product = H @ moved
        assert isinstance(product.data, SparseOperator)
        assert product.cols == moved.cols
        np.testing.assert_allclose(product.align(OBS, STATE).dense(), Ha @ Sa)

    def test_block_diag_of_sparse_covariances(self):
        """Sparse and diagonal blocks side by side keep each form, and a product with ``H`` is sparse."""
        _, S_0, _, Ha, Sa = _sparse_operators()
        bias = Covariance.diagonal(2.0, Space({"bias": pd.Index(["A"], name="site")}))
        S = Covariance.block_diag(S_0, bias)
        assert isinstance(S.data, BlockDiagonalOperator)
        H = Operator(sp.csr_array(np.hstack([Ha, np.ones((M, 1))])), OBS, S.space)
        HS = H @ S
        assert isinstance(HS.data, SparseOperator)
        expected = np.hstack([Ha @ Sa, np.full((M, 1), 2.0)])
        np.testing.assert_allclose(HS.dense(), expected)

    def test_from_blocks(self):
        """Blocks that are all sparse (frames, matrices, diagonals, operators) assemble a sparse matrix."""
        obs, cells = OBS.select("co2"), STATE.select("flux")
        bias = Space({"bias": pd.Index(["A", "B"], name="site")})
        ch4 = Space({"ch4": pd.Index(["p0", "p1"], name="obs_id")})
        Ha, corner = _jacobian(), np.array([[0.0, 3.0], [4.0, 0.0]])
        blocks = {
            ("co2", "flux"): _from_spmatrix(sp.csr_array(Ha), obs, cells),
            ("co2", "bias"): sp.csr_array(np.eye(M, 2)),
            ("ch4", "bias"): Operator(sp.csr_array(corner), ch4, bias),
            ("ch4", "ch4"): DiagonalOperator([1.0, 2.0]),
        }
        rows = Space({"co2": obs, "ch4": ch4.select("ch4")})
        cols = Space(
            {"flux": cells, "bias": bias.select("bias"), "ch4": ch4.select("ch4")}
        )
        op = Operator.from_blocks(blocks, rows=rows, cols=cols)
        assert isinstance(op.data, SparseOperator)
        expected = np.zeros((M + 2, N + 4))
        expected[:M, :N], expected[:M, N : N + 2] = Ha, np.eye(M, 2)
        expected[M:, N : N + 2], expected[M:, N + 2 :] = corner, np.diag([1.0, 2.0])
        np.testing.assert_array_equal(op.dense(), expected)
        # One dense block makes it dense
        blocks[("co2", "bias")] = np.eye(M, 2)
        assert isinstance(Operator.from_blocks(blocks, rows, cols).data, DenseOperator)

    def test_from_blocks_single_block_keeps_its_form(self):
        """One block covering both spaces keeps its form: a diagonal stays a diagonal."""
        sites = pd.Index(["a", "b", "c"], name="site")
        frame = _from_spmatrix(sp.diags_array([1.0, 2.0, 3.0]), sites, sites)
        S = Covariance.from_blocks({("s", "s"): frame})
        assert isinstance(S, Covariance)
        assert isinstance(S.data, DiagonalOperator)


# ---------------------------------------------------------------------------
# A sparse problem solves as the dense one does
# ---------------------------------------------------------------------------


def _frames(sparse: bool, drop_cell: bool = False) -> dict:
    """The inputs of `_jacobian`'s problem, its matrices as pandas-sparse frames or dense frames."""
    rng = np.random.default_rng(1)
    state, obs = STATE.index, OBS.index
    H, S_0, S_z = _jacobian(), _banded(N, seed=2), np.diag(rng.uniform(0.1, 0.5, M))

    def frame(values, index, columns):
        """A frame of `values`, pandas-sparse or dense."""
        if sparse:
            return _from_spmatrix(sp.csr_array(values), index, columns)
        return pd.DataFrame(values, index=index, columns=columns)

    H_frame = frame(H, obs, state)
    if drop_cell:
        H_frame = H_frame.drop(columns=H_frame.columns[[3]])
    x_0 = rng.normal(5.0, 1.0, N)
    return {
        "obs": pd.Series(H @ x_0 + rng.normal(0, 1.0, M), index=obs, name="obs"),
        "prior": pd.Series(x_0, index=state, name="prior"),
        "forward": H_frame,
        "prior_cov": frame(S_0, state, state),
        "obs_cov": frame(S_z, obs, obs),
    }


def _problem(sparse: bool) -> Problem:
    """The problem of `_jacobian`, read from pandas-sparse frames or dense frames."""
    frames = _frames(sparse)
    return Problem(
        obs=Vector.from_series(frames["obs"]),
        prior=Vector.from_series(frames["prior"]),
        forward=Operator.from_frame(frames["forward"]),
        prior_cov=Covariance.from_frame(frames["prior_cov"]),
        obs_cov=Covariance.from_frame(frames["obs_cov"]),
    )


class TestProblem:
    """Sparse ``H``, ``S_0`` and ``S_z`` given as pandas-sparse frames stay sparse and solve as dense ones."""

    def test_forms(self):
        """The problem holds sparse frames as sparse forms, and dense frames as dense ones."""
        problem, dense = _problem(sparse=True), _problem(sparse=False)
        assert isinstance(problem.forward.data, SparseOperator)
        assert isinstance(problem.prior_cov.data, SparseOperator)
        assert isinstance(problem.obs_cov.data, DiagonalOperator)
        for name in ("forward", "prior_cov", "obs_cov"):
            assert isinstance(getattr(dense, name).data, DenseOperator)
            np.testing.assert_array_equal(
                getattr(problem, name).dense(), getattr(dense, name).dense()
            )

    def test_solve_matches_dense(self):
        """``fips.solve`` gives the dense problem's posterior."""
        post = fips.solve(_problem(sparse=True))
        ref = fips.solve(_problem(sparse=False))
        for name in ("estimate", "variances", "modeled_obs"):
            assert getattr(post, name).index.equals(getattr(ref, name).index)
            np.testing.assert_allclose(
                getattr(post, name).values, getattr(ref, name).values, rtol=1e-10
            )
        for got, want in ((post.cov(), ref.cov()), (post.kalman_gain, ref.kalman_gain)):
            np.testing.assert_allclose(
                got.dense(), want.dense(), rtol=1e-10, atol=1e-12
            )
        assert post.dofs == pytest.approx(ref.dofs, rel=1e-10)

    def test_a_cell_missing_from_h_raises(self):
        """A state cell missing from a sparse ``H`` raises, naming it: no zero fill."""
        frames = _frames(sparse=True, drop_cell=True)
        prior = Vector.from_series(frames["prior"])
        with pytest.raises(
            ValueError, match="missing from the columns of forward: 'c3'"
        ):
            Problem(
                obs=Vector.from_series(frames["obs"]),
                prior=prior,
                forward=Operator.from_frame(frames["forward"]),
                prior_cov=Covariance.from_frame(frames["prior_cov"]),
                obs_cov=Covariance.from_frame(frames["obs_cov"]),
            )

    def test_a_frame_reindexed_onto_the_state_stays_sparse(self):
        """``frame.reindex(columns=state, fill_value=0.0)`` puts a pandas-sparse ``H`` on the whole state, sparse."""
        frames = _frames(sparse=True, drop_cell=True)
        state = frames["prior"].index
        H = Operator.from_frame(
            frames["forward"].reindex(columns=state, fill_value=0.0)
        )
        assert isinstance(H.data, SparseOperator)
        assert not H.dense()[:, 3].any()
        full = Operator.from_frame(_frames(sparse=True)["forward"])
        expected = full.dense()
        expected[:, 3] = 0.0
        np.testing.assert_array_equal(H.dense(), expected)


# ---------------------------------------------------------------------------
# Aggregating a problem with a pandas-sparse frame and with a sparse Operator
# ---------------------------------------------------------------------------

DAYS, PER_DAY = 8, 4
_TIMES = pd.date_range("2023-01-01", periods=DAYS * PER_DAY, freq="6h", name="obs_time")
_OBS_IDX = pd.MultiIndex.from_arrays(
    [["co2"] * len(_TIMES), _TIMES], names=["block", "obs_time"]
)
_CELL_IDX = pd.MultiIndex.from_arrays(
    [["flux"] * len(_TIMES), [f"c{k}" for k in range(len(_TIMES))]],
    names=["block", "cell"],
)
_DAILY = pd.Grouper(level="obs_time", freq="D")


def _aggregator_inputs():
    """Return the obs, a sparse ``H`` (observation k sees cells k and k + 1) and the variances."""
    n = len(_TIMES)
    H = np.zeros((n, n))
    for k in range(n):
        H[k, k], H[k, (k + 1) % n] = 1.0, 0.5
    obs = pd.Series(np.arange(1.0, n + 1), index=_OBS_IDX, name="obs")
    return obs, H, np.linspace(1.0, 2.0, n)


def _expected(H, variances):
    """The daily means, ``W H`` and ``W S_z W^T``, written out."""
    W = np.kron(np.eye(DAYS), np.full((1, PER_DAY), 1.0 / PER_DAY))
    return W @ H, W @ np.diag(variances) @ W.T


def _frame(values, columns=_CELL_IDX, fill=None) -> pd.DataFrame:
    """A frame of `values` on the observations' rows, pandas-sparse with `fill` if given."""
    frame = pd.DataFrame(values, index=_OBS_IDX, columns=columns)
    return frame if fill is None else frame.astype(pd.SparseDtype(float, fill))


def _daily_problem(forward: Operator, obs_cov: Covariance) -> Problem:
    """The problem of `forward` and `obs_cov`, aggregated to daily means."""
    obs, _, _ = _aggregator_inputs()
    state = Space.from_index(_CELL_IDX)
    problem = Problem(
        prior=Vector(np.zeros(len(state)), state),
        prior_cov=Covariance.diagonal(1.0, state),
        obs=Vector.from_series(obs),
        obs_cov=obs_cov,
        forward=forward,
    )
    return problem.aggregate(fips.aggregate.mean(problem.obs.space, by=_DAILY))


class TestAggregate:
    """A pandas-sparse frame enters as a sparse Operator, and aggregating keeps it sparse."""

    @pytest.mark.parametrize("fill", [0.0, np.nan], ids=["zero-fill", "nan-fill"])
    def test_pandas_sparse_frame(self, fill):
        """A pandas-sparse ``H`` aggregates as its dense frame does, and stays sparse."""
        _, H, variances = _aggregator_inputs()
        S_z = Covariance.diagonal(variances, Space.from_index(_OBS_IDX))
        daily = _daily_problem(Operator.from_frame(_frame(H, fill=fill)), S_z)
        reference = _daily_problem(Operator.from_frame(_frame(H)), S_z)
        assert isinstance(daily.forward.data, SparseOperator)
        np.testing.assert_array_equal(daily.forward.dense(), reference.forward.dense())
        WH, WSW = _expected(H, variances)
        np.testing.assert_allclose(daily.forward.dense(), WH)
        np.testing.assert_allclose(daily.obs_cov.dense(), WSW)

    def test_from_spmatrix_frames(self):
        """``from_spmatrix`` frames (NaN fill on pandas 3) for ``H`` and ``S_z`` aggregate to the same numbers."""
        _, H, variances = _aggregator_inputs()
        frame = _from_spmatrix(sp.csr_array(H), _OBS_IDX, _CELL_IDX)
        S_z = _from_spmatrix(sp.diags_array(variances), _OBS_IDX, _OBS_IDX)
        daily = _daily_problem(Operator.from_frame(frame), Covariance.from_frame(S_z))
        WH, WSW = _expected(H, variances)
        np.testing.assert_allclose(daily.forward.dense(), WH)
        np.testing.assert_allclose(daily.obs_cov.dense(), WSW)

    def test_sparse_operator(self):
        """A sparse ``H`` and a diagonal ``S_z`` aggregate to a sparse Operator and Covariance."""
        _, H, variances = _aggregator_inputs()
        obs_space, state = Space.from_index(_OBS_IDX), Space.from_index(_CELL_IDX)
        H_op = Operator(sp.csr_array(H), obs_space, state, name="forward")
        S_z = Covariance.diagonal(variances, obs_space)
        daily = _daily_problem(H_op, S_z)
        H_agg, S_agg = daily.forward, daily.obs_cov
        assert type(H_agg) is Operator and isinstance(S_agg, Covariance)
        assert isinstance(H_agg.data, SparseOperator)
        assert isinstance(S_agg.data, (SparseOperator, DiagonalOperator))
        assert H_agg.name == "forward" and H_agg.cols == state
        assert H_agg.rows == S_agg.space and len(H_agg.rows) == DAYS
        WH, WSW = _expected(H, variances)
        np.testing.assert_allclose(H_agg.dense(), WH)
        np.testing.assert_allclose(S_agg.dense(), WSW)
        # The rows in another order give the same aggregate
        H_moved = H_op.align(_reversed(obs_space), state)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            again = _daily_problem(H_moved, S_z)
        assert isinstance(again.forward.data, SparseOperator)
        np.testing.assert_allclose(again.forward.dense(), WH)

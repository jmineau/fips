"""
Tests for sparse matrix support.

Sparse storage is available in Structure2D, Matrix, ForwardOperator,
CovarianceBuilder, and InverseProblem.
"""

import numpy as np
import pandas as pd
import pytest

from fips.covariance import (
    BlockDecayError,
    CovarianceBuilder,
    DiagonalError,
)
from fips.matrix import Matrix, MatrixBlock
from fips.operators import ForwardOperator
from fips.problem import InverseProblem
from fips.vector import Block, Vector

# ---------------------------------------------------------------------------
# Helpers / shared data
# ---------------------------------------------------------------------------


def _state_idx(n=4):
    return pd.Index([f"s{i}" for i in range(n)], name="state_id")


def _obs_idx(m=3):
    return pd.Index([f"o{i}" for i in range(m)], name="obs_id")


def _make_sparse_H(m=3, n=4, seed=0):
    """Forward operator with ~50% zeros — typical of a STILT Jacobian."""
    rng = np.random.default_rng(seed)
    H = rng.random((m, n))
    H[H < 0.5] = 0.0  # zero out half the entries
    return H


def _make_block(data, row_block, col_block, row_idx, col_idx):
    return MatrixBlock(
        data, row_block=row_block, col_block=col_block, index=row_idx, columns=col_idx
    )


# ---------------------------------------------------------------------------
# Structure2D / Matrix — sparse flag
# ---------------------------------------------------------------------------


class TestSparseMatrix:
    """Tests for sparse matrix creation and conversion."""

    def test_sparse_flag_creates_sparse_dataframe(self):
        """Test that sparse=True creates a sparse dataframe."""
        idx = _state_idx()
        block = _make_block(np.eye(4), "s", "s", idx, idx)
        M = Matrix([block], sparse=True)
        assert M.is_sparse

    def test_dense_flag_creates_dense_dataframe(self):
        """Test that sparse=False creates a dense dataframe."""
        idx = _state_idx()
        block = _make_block(np.eye(4), "s", "s", idx, idx)
        M = Matrix([block], sparse=False)
        assert not M.is_sparse

    def test_values_equal_after_sparsification(self):
        """Test that values remain equal after converting to sparse."""
        idx = _state_idx()
        data = np.diag([1.0, 0.0, 3.0, 0.0])
        block = _make_block(data, "s", "s", idx, idx)
        M_dense = Matrix([block])
        M_sparse = Matrix([block], sparse=True)
        assert np.allclose(M_dense.values, M_sparse.to_numpy())

    def test_to_sparse_roundtrip(self):
        """Test converting to sparse and back to dense preserves values."""
        idx = _state_idx()
        block = _make_block(np.eye(4), "s", "s", idx, idx)
        M = Matrix([block])
        assert not M.is_sparse
        Ms = M.to_sparse()
        assert Ms.is_sparse
        Md = Ms.to_dense()
        assert not Md.is_sparse
        assert np.allclose(M.values, Md.values)

    def test_to_sparse_with_threshold(self):
        """Test that to_sparse with threshold zeros out small values."""
        idx = _state_idx()
        data = np.array(
            [
                [1.0, 1e-12, 0.0, 0.0],
                [1e-12, 1.0, 1e-12, 0.0],
                [0.0, 1e-12, 1.0, 1e-12],
                [0.0, 0.0, 1e-12, 1.0],
            ]
        )
        block = _make_block(data, "s", "s", idx, idx)
        M = Matrix([block])
        Ms = M.to_sparse(threshold=1e-10)
        # Sub-threshold values should have been zeroed
        assert Ms.values[0, 1] == 0.0
        # Values above threshold should be unchanged
        assert Ms.values[0, 0] == 1.0

    def test_to_sparse_no_threshold_preserves_small_values(self):
        """Test that to_sparse without threshold preserves all values."""
        """Test that to_sparse without threshold preserves all values."""
        idx = _state_idx()
        data = np.eye(4)
        data[0, 1] = 1e-12
        block = _make_block(data, "s", "s", idx, idx)
        M = Matrix([block])
        Ms = M.to_sparse()  # no threshold
        assert Ms.values[0, 1] == pytest.approx(1e-12)

    def test_add_sparse_and_dense(self):
        """Adding sparse + dense should produce a valid (dense) result."""
        idx = _state_idx()
        block = _make_block(np.eye(4), "s", "s", idx, idx)
        Ms = Matrix([block], sparse=True)
        Md = Matrix([block])
        result = Ms + Md
        assert np.allclose(result.values, 2 * np.eye(4))

    def test_add_sparse_and_sparse(self):
        """Test adding two sparse matrices together."""
        idx = _state_idx()
        block = _make_block(np.eye(4), "s", "s", idx, idx)
        Ms1 = Matrix([block], sparse=True)
        Ms2 = Matrix([block], sparse=True)
        result = Ms1 + Ms2
        assert np.allclose(result.values, 2 * np.eye(4))

    def test_scale_preserves_sparsity(self):
        """Test that scaling a sparse matrix preserves sparsity."""
        idx = _state_idx()
        block = _make_block(np.diag([1.0, 0.0, 3.0, 0.0]), "s", "s", idx, idx)
        Ms = Matrix([block], sparse=True)
        scaled = Ms.scale(2.0)
        assert scaled.is_sparse
        assert scaled.to_numpy()[0, 0] == pytest.approx(2.0)
        assert scaled.to_numpy()[1, 1] == pytest.approx(0.0)

    def test_sparse_preserved_through_reindex(self):
        """Sparsity should survive a reindex call."""
        idx_full = _state_idx(4)
        block = _make_block(np.eye(4), "s", "s", idx_full, idx_full)
        Ms = Matrix([block], sparse=True)
        # Matrix stores a MultiIndex with a 'block' level; slice it directly from
        # the existing index so we don't need to reconstruct it manually.
        sub_index = Ms.index[:3]
        sub_cols = Ms.columns[:3]
        reindexed = Ms.reindex(index=sub_index, columns=sub_cols, fill_value=0.0)
        assert reindexed.is_sparse
        assert reindexed.shape == (3, 3)


# ---------------------------------------------------------------------------
# ForwardOperator — sparse convolve
# ---------------------------------------------------------------------------


class TestSparseForwardOperator:
    """Tests for sparse forward operator operations."""

    @pytest.fixture
    def H_data(self):
        """Provide sparse forward operator data."""
        return _make_sparse_H(m=3, n=4)

    @pytest.fixture
    def state_vec(self):
        """Provide state vector fixture."""
        idx = _state_idx(4)
        block = Block(pd.Series([1.0, 2.0, 3.0, 4.0], index=idx, name="state"))
        return Vector([block])

    def test_sparse_forward_operator_is_sparse(self, H_data):
        """Test that sparse forward operator is marked as sparse."""
        obs_idx = _obs_idx(3)
        state_idx = _state_idx(4)
        hblock = _make_block(H_data, "obs", "state", obs_idx, state_idx)
        H = ForwardOperator([hblock], sparse=True)
        assert H.is_sparse

    def test_sparse_convolve_matches_dense(self, H_data, state_vec):
        """Test that sparse convolve produces same result as dense."""
        obs_idx = _obs_idx(3)
        state_idx = _state_idx(4)
        hblock = _make_block(H_data, "obs", "state", obs_idx, state_idx)

        H_dense = ForwardOperator([hblock])
        H_sparse = ForwardOperator([hblock], sparse=True)

        y_dense = H_dense.convolve(state_vec)
        y_sparse = H_sparse.convolve(state_vec)

        assert np.allclose(y_dense.to_numpy(), y_sparse.to_numpy())

    def test_sparse_convolve_returns_series(self, H_data, state_vec):
        """Test that sparse convolve returns a Series."""
        obs_idx = _obs_idx(3)
        state_idx = _state_idx(4)
        hblock = _make_block(H_data, "obs", "state", obs_idx, state_idx)
        H = ForwardOperator([hblock], sparse=True)
        y = H.convolve(state_vec)
        assert isinstance(y, pd.Series)
        assert len(y) == 3


# ---------------------------------------------------------------------------
# CovarianceBuilder — sparse build path
# ---------------------------------------------------------------------------


class TestSparseCovarianceBuilder:
    """Tests for building sparse covariance matrices."""

    @pytest.fixture
    def midx(self):
        """Provide multi-index fixture."""
        return pd.MultiIndex.from_tuples(
            [(f"loc{i}", t) for i in range(3) for t in range(4)],
            names=["location", "time"],
        )

    def test_sparse_build_returns_sparse_dataframe(self, midx):
        """Test that sparse build returns a sparse DataFrame."""
        # sparse=True lives on CovarianceBuilder, not ErrorComponent directly
        builder = CovarianceBuilder([DiagonalError("diag", variances=1.0)])
        df = builder.build(midx, sparse=True)
        assert any(isinstance(dt, pd.SparseDtype) for dt in df.dtypes)

    def test_sparse_and_dense_build_are_numerically_equal(self, midx):
        """Test that sparse and dense builds produce numerically equal results."""
        builder = DiagonalError("d1", variances=1.0) + DiagonalError(
            "d2", variances=0.5
        )
        df_dense = builder.build(midx, sparse=False)
        df_sparse = builder.build(midx, sparse=True)
        assert np.allclose(
            df_dense.to_numpy(),
            df_sparse.sparse.to_dense().to_numpy(),
        )

    def test_single_component_sparse_build(self, midx):
        """Test building sparse covariance from a single component."""
        builder = CovarianceBuilder([DiagonalError("diag", variances=2.0)])
        df = builder.build(midx, sparse=True)
        assert any(isinstance(dt, pd.SparseDtype) for dt in df.dtypes)
        assert np.allclose(np.diag(df.sparse.to_dense().to_numpy()), 2.0)


# ---------------------------------------------------------------------------
# InverseProblem — end-to-end with sparse ForwardOperator
# ---------------------------------------------------------------------------


class TestSparseInverseProblem:
    """Tests for inverse problems with sparse matrices."""

    @pytest.fixture
    def sparse_problem(self):
        """Fixture providing an InverseProblem with a sparse forward operator."""
        n_state = 6
        n_obs = 4
        rng = np.random.default_rng(42)

        state_idx = pd.Index([f"s{i}" for i in range(n_state)], name="state_id")
        obs_idx = pd.Index([f"o{i}" for i in range(n_obs)], name="obs_id")

        # Sparse Jacobian: each obs only sees 2 state elements
        H_data = np.zeros((n_obs, n_state))
        for i in range(n_obs):
            H_data[i, i % n_state] = rng.random()
            H_data[i, (i + 1) % n_state] = rng.random()

        prior = pd.Series(rng.random(n_state), index=state_idx, name="prior")
        obs = pd.Series(rng.random(n_obs), index=obs_idx, name="obs")
        H = pd.DataFrame(H_data, index=obs_idx, columns=state_idx)
        S_0 = pd.DataFrame(np.eye(n_state), index=state_idx, columns=state_idx)
        S_z = pd.DataFrame(np.eye(n_obs) * 0.1, index=obs_idx, columns=obs_idx)

        return InverseProblem(
            prior=prior,
            obs=obs,
            forward_operator=H,
            prior_error=S_0,
            modeldata_mismatch=S_z,
        )

    @pytest.fixture
    def dense_problem(self, sparse_problem):
        """Fixture providing the same problem with a dense forward operator."""
        # Re-expose using the same underlying data
        return sparse_problem

    def test_forward_operator_can_be_made_sparse(self, sparse_problem):
        """Test that forward operator can be converted to sparse representation."""
        sparse_problem.forward_operator = sparse_problem.forward_operator.to_sparse(
            threshold=1e-10
        )
        assert sparse_problem.forward_operator.is_sparse

    def test_sparse_solve_produces_finite_posterior(self, sparse_problem):
        """Test that solving with sparse operators produces finite posterior values."""
        sparse_problem.forward_operator = sparse_problem.forward_operator.to_sparse(
            threshold=1e-10
        )
        sparse_problem.solve("bayesian")
        assert np.isfinite(sparse_problem.estimator.x_hat).all()
        assert np.isfinite(sparse_problem.estimator.S_hat).all()

    def test_sparse_and_dense_solve_agree(self, sparse_problem):
        """Sparse and dense forward operators must produce the same posterior."""
        # Dense solve
        sparse_problem.solve("bayesian")
        x_dense = sparse_problem.estimator.x_hat.copy()

        # Sparse solve (same problem, sparsified H)
        sparse_problem.forward_operator = sparse_problem.forward_operator.to_sparse(
            threshold=1e-10
        )
        sparse_problem.solve("bayesian")
        x_sparse = sparse_problem.estimator.x_hat

        assert np.allclose(x_dense, x_sparse, rtol=1e-10)

    def test_posterior_shape_with_sparse_operator(self, sparse_problem):
        """Test that posterior has correct shape when using sparse operators."""
        sparse_problem.forward_operator = sparse_problem.forward_operator.to_sparse(
            threshold=1e-10
        )
        sparse_problem.solve("bayesian")
        assert sparse_problem.posterior.shape[0] == sparse_problem.n_state
        assert sparse_problem.posterior_error.shape == (
            sparse_problem.n_state,
            sparse_problem.n_state,
        )


# ---------------------------------------------------------------------------
# fill_value normalization (fips#2)
# ---------------------------------------------------------------------------


def _pandas_native_sparse(m=6, n=5, density=0.3, seed=0):
    """
    Build a sparse frame the ordinary pandas way.

    On pandas >= 3 `from_spmatrix` gives float frames `fill_value=NaN`
    (pandas-dev/pandas#59212), losing scipy's definition that unstored
    entries are zero.
    """
    sp = pytest.importorskip("scipy.sparse")
    matrix = sp.random(m, n, density=density, format="csr", random_state=seed)
    frame = pd.DataFrame.sparse.from_spmatrix(
        matrix,
        index=pd.Index([f"o{i}" for i in range(m)], name="obs_id"),
        columns=pd.Index([f"s{i}" for i in range(n)], name="state_id"),
    )
    return frame, matrix


class TestSparseFillValue:
    """Sparse blocks must fill with 0.0, whatever pandas hands back."""

    def test_normalize_sets_fill_value_to_zero(self):
        """Normalizing must leave no NaN and a 0.0 fill."""
        from fips._sparse import normalize_fill_value

        frame, _ = _pandas_native_sparse()
        out = normalize_fill_value(frame)
        assert all(dt.fill_value == 0.0 for dt in out.dtypes)
        assert not out.isna().to_numpy().any()

    def test_normalize_preserves_values_and_sparsity(self):
        """Normalizing must not densify: the point of sparse storage."""
        from fips._sparse import normalize_fill_value

        frame, matrix = _pandas_native_sparse(m=200, n=100, density=0.01)
        out = normalize_fill_value(frame)
        np.testing.assert_allclose(out.sparse.to_dense().to_numpy(), matrix.toarray())
        # the point of sparse storage: normalizing must not densify
        assert out.sparse.density == pytest.approx(frame.sparse.density)

    def test_normalize_is_a_no_op_when_already_zero_filled(self):
        """An already zero-filled frame is returned untouched."""
        from fips._sparse import normalize_fill_value

        frame = pd.DataFrame(np.eye(3)).astype(pd.SparseDtype(float, fill_value=0.0))
        assert normalize_fill_value(frame) is frame

    def test_normalize_passes_dense_frames_through(self):
        """Dense frames are returned untouched."""
        from fips._sparse import normalize_fill_value

        frame = pd.DataFrame(np.eye(3))
        assert normalize_fill_value(frame) is frame

    def test_structure_accepts_pandas_native_sparse_frame(self):
        """A user-supplied sparse frame must not be rejected as containing NaN."""
        frame, matrix = _pandas_native_sparse()
        block = MatrixBlock(frame, row_block="o", col_block="s")
        assert block.is_sparse
        assert all(dt.fill_value == 0.0 for dt in block.data.dtypes)
        values = block.values
        dense = values.toarray() if hasattr(values, "toarray") else np.asarray(values)  # pyright: ignore[reportAttributeAccessIssue]
        np.testing.assert_allclose(dense, matrix.toarray())
        assert not np.isnan(dense).any()

    def test_covariance_builder_sparse_has_zero_fill(self):
        """CovarianceBuilder.build(sparse=True) — the path named in fips#2."""
        idx = pd.MultiIndex.from_product([["a"], range(4)], names=["block", "state_id"])
        built = CovarianceBuilder([DiagonalError("e", 2.0)]).build(idx, sparse=True)
        assert all(dt.fill_value == 0.0 for dt in built.dtypes)
        assert not built.isna().to_numpy().any()
        np.testing.assert_allclose(built.sparse.to_dense().to_numpy(), np.eye(4) * 2.0)

    def test_covariance_sparse_matches_dense(self):
        """Sparse and dense builds must hold the same numbers."""
        idx = pd.MultiIndex.from_product([["a"], range(4)], names=["block", "state_id"])
        builder = CovarianceBuilder([DiagonalError("e", 2.0)])
        np.testing.assert_allclose(
            builder.build(idx, sparse=True).sparse.to_dense().to_numpy(),
            builder.build(idx, sparse=False).to_numpy(),
        )


# ---------------------------------------------------------------------------
# A NaN stored in a sparse matrix is data, not an implicit zero
# ---------------------------------------------------------------------------

_SITES = pd.MultiIndex.from_product(
    [["state"], ["a", "b", "c"]], names=["block", "site"]
)


def _nan_filled_frame(stored_nan: bool) -> pd.DataFrame:
    """
    Build a sparse frame with ``fill_value=NaN`` that stores rows 0 and 1.

    pandas 3's ``from_spmatrix`` builds frames like this; pandas 2's fills
    with 0.0, so the columns are built on a sparse index directly. Column "x"
    stores a NaN in row 1 when `stored_nan` is set; row 2 is implicit.
    """
    stored = pd.arrays.SparseArray([1.0, 2.0, 0.0], fill_value=0.0).sp_index
    x = [1.0, np.nan] if stored_nan else [1.0, 2.0]
    return pd.DataFrame(
        {
            "x": pd.arrays.SparseArray(x, sparse_index=stored, fill_value=np.nan),
            "y": pd.arrays.SparseArray(
                [4.0, 5.0], sparse_index=stored, fill_value=np.nan
            ),
        }
    )


def _aggregator_inputs(stored_nan: bool):
    """Return obs, a sparse forward operator and a covariance, as bare frames."""
    from fips.indexes import assign_block

    times = pd.DatetimeIndex(
        [
            "2023-01-01 00:00",
            "2023-01-01 12:00",
            "2023-01-02 00:00",
            "2023-01-02 12:00",
        ],
        name="obs_time",
    )
    obs_idx = assign_block(times, "obs")
    state_idx = assign_block(pd.Index(["s0", "s1"], name="state_id"), "state")
    # s0 is seen on the first day only and s1 on the second: W @ H is diagonal
    H = np.array([[0.5, 0.0], [0.5, 0.0], [0.0, 0.5], [0.0, 0.5]])
    if stored_nan:
        H[1, 0] = np.nan
    H_df = pd.DataFrame(H, index=obs_idx, columns=state_idx).astype(
        pd.SparseDtype(float, fill_value=0.0)
    )
    obs = pd.Series([1.0, 2.0, 3.0, 4.0], index=obs_idx, name="obs")
    S_z = pd.DataFrame(np.eye(4) * 2.0, index=obs_idx, columns=obs_idx)
    return obs, H_df, S_z


def _daily(idx):
    """Group obs_time to daily frequency."""
    return idx.get_level_values("obs_time").floor("D")


class TestStoredNaN:
    """A NaN stored in a sparse matrix is refused, as on the dense path."""

    def test_normalize_raises_on_stored_nan(self):
        """Filling would read the stored NaN as a zero."""
        from fips._sparse import normalize_fill_value

        with pytest.raises(ValueError, match="Data contains NaN values."):
            normalize_fill_value(_nan_filled_frame(stored_nan=True))

    def test_normalize_fills_implicit_entries_only(self):
        """Without a stored NaN the implicit entries become zeros, nothing else."""
        from fips._sparse import normalize_fill_value

        frame = _nan_filled_frame(stored_nan=False)
        out = normalize_fill_value(frame)
        assert all(dt.fill_value == 0.0 for dt in out.dtypes)
        np.testing.assert_array_equal(
            out.sparse.to_dense().to_numpy(), [[1.0, 4.0], [2.0, 5.0], [0.0, 0.0]]
        )
        assert out.sparse.density == pytest.approx(frame.sparse.density)

    def test_covariance_builder_sparse(self):
        """A NaN variance is refused on the sparse build, as on the dense one."""
        from fips.covariance import CovarianceMatrix

        builder = CovarianceBuilder([DiagonalError("e", np.nan)])
        with pytest.raises(ValueError, match="Data contains NaN values."):
            CovarianceMatrix(builder.build(_SITES))
        with pytest.raises(ValueError, match="Data contains NaN values."):
            CovarianceMatrix(builder.build(_SITES, sparse=True))

    def test_pandas_native_sparse_frame(self):
        """A frame built with from_spmatrix from a matrix that stores a NaN."""
        sp = pytest.importorskip("scipy.sparse")
        dense = np.diag([1.0, np.nan, 3.0])
        frame = pd.DataFrame.sparse.from_spmatrix(
            sp.csr_matrix(dense), index=_SITES, columns=_SITES
        )
        with pytest.raises(ValueError, match="Data contains NaN values."):
            Matrix(pd.DataFrame(dense, index=_SITES, columns=_SITES))
        with pytest.raises(ValueError, match="Data contains NaN values."):
            Matrix(frame)

    @pytest.mark.parametrize("sparse", [False, True])
    def test_diagonal(self, sparse):
        """A NaN on a diagonal is refused, sparse or dense."""
        with pytest.raises(ValueError, match="Data contains NaN values."):
            Matrix.diagonal([1.0, np.nan, 3.0], index=_SITES, sparse=sparse)

    def test_obs_aggregator_sparse_operator(self):
        """A NaN in a sparse forward operator does not come out of W @ H as a zero."""
        from fips.aggregators import ObsAggregator

        obs, H, S_z = _aggregator_inputs(stored_nan=True)
        with pytest.raises(ValueError, match="Data contains NaN values."):
            ForwardOperator(ObsAggregator(by=_daily).apply(obs, H, S_z)[1])

    def test_obs_aggregator_sparse_operator_without_nan(self):
        """The sparse aggregated operator holds the dense one's numbers."""
        from fips.aggregators import ObsAggregator

        obs, H, S_z = _aggregator_inputs(stored_nan=False)
        agg = ObsAggregator(by=_daily)
        H_sparse = agg.apply(obs, H, S_z)[1]
        H_dense = agg.apply(obs, H.sparse.to_dense(), S_z)[1]
        dense = H_sparse.sparse.to_dense().to_numpy()
        np.testing.assert_allclose(dense, [[0.5, 0.0], [0.0, 0.5]])
        np.testing.assert_allclose(dense, H_dense.to_numpy())
        assert ForwardOperator(H_sparse).is_sparse


# ---------------------------------------------------------------------------
# A column that stores every row has no implicit entry to fill
# ---------------------------------------------------------------------------

_PAIR = pd.MultiIndex.from_product([["state"], ["a", "b"]], names=["block", "site"])


def _full_column_frame() -> pd.DataFrame:
    """
    Build a sparse frame with ``fill_value=NaN`` whose first column stores every row.

    pandas 3's ``from_spmatrix`` builds this from ``[[1, 0], [3, 4]]``; pandas 2's
    fills with 0.0, so the columns are built directly. Column "a" has no implicit
    entry, so it holds no NaN; column "b" has one, in row 0.
    """
    frame = pd.DataFrame(
        {
            "a": pd.arrays.SparseArray([1.0, 3.0], fill_value=np.nan),
            "b": pd.arrays.SparseArray([np.nan, 4.0], fill_value=np.nan),
        },
        index=_PAIR,
    )
    frame.columns = _PAIR
    return frame


class TestFullColumn:
    """A column that stores every row is re-based like the others: one fill, no warning."""

    def test_normalize_rebases_the_column(self):
        """No column keeps a NaN fill, and no cell changes."""
        from fips._sparse import normalize_fill_value

        frame = _full_column_frame()
        out = normalize_fill_value(frame)
        assert [dt.fill_value for dt in out.dtypes] == [0.0, 0.0]
        np.testing.assert_array_equal(
            out.sparse.to_dense().to_numpy(), [[1.0, 0.0], [3.0, 4.0]]
        )
        assert out.sparse.density == pytest.approx(frame.sparse.density)
        pd.testing.assert_index_equal(out.index, frame.index)
        pd.testing.assert_index_equal(out.columns, frame.columns)

    def test_normalize_leaves_the_input_alone(self):
        """The frame handed in keeps its NaN fills."""
        from fips._sparse import normalize_fill_value

        frame = _full_column_frame()
        normalize_fill_value(frame)
        assert all(np.isnan(dt.fill_value) for dt in frame.dtypes)

    def test_structure_takes_the_frame(self):
        """A mix of fills would warn when the structure checks the data for NaN."""
        matrix = Matrix(_full_column_frame())
        assert [dt.fill_value for dt in matrix.data.dtypes] == [0.0, 0.0]
        np.testing.assert_array_equal(matrix.values.toarray(), [[1.0, 0.0], [3.0, 4.0]])

    def test_frame_from_spmatrix(self):
        """The helper every sparse build goes through."""
        sp = pytest.importorskip("scipy.sparse")
        from fips._sparse import frame_from_spmatrix

        frame = frame_from_spmatrix(
            sp.csr_matrix(np.array([[1.0, 0.0], [3.0, 4.0]])), _PAIR, _PAIR
        )
        assert [dt.fill_value for dt in frame.dtypes] == [0.0, 0.0]
        np.testing.assert_array_equal(
            frame.sparse.to_dense().to_numpy(), [[1.0, 0.0], [3.0, 4.0]]
        )

    def test_covariance_builder_sparse(self):
        """A covariance whose columns hold no zero, built sparse."""
        builder = CovarianceBuilder(
            [
                BlockDecayError(
                    "e", 2.0, ["block"], lambda group: np.full((len(group),) * 2, 0.5)
                )
            ]
        )
        built = builder.build(_PAIR, sparse=True)
        assert [dt.fill_value for dt in built.dtypes] == [0.0, 0.0]
        np.testing.assert_allclose(built.sparse.to_dense().to_numpy(), np.ones((2, 2)))

    def test_obs_aggregator_sparse_operator(self):
        """An aggregated operator whose columns hold no zero."""
        from fips.aggregators import ObsAggregator

        obs, H, S_z = _aggregator_inputs(stored_nan=False)
        H = pd.DataFrame(
            np.full(H.shape, 0.5), index=H.index, columns=H.columns
        ).astype(pd.SparseDtype(float, fill_value=0.0))
        H_agg = ObsAggregator(by=_daily).apply(obs, H, S_z)[1]
        assert [dt.fill_value for dt in H_agg.dtypes] == [0.0, 0.0]
        np.testing.assert_allclose(
            H_agg.sparse.to_dense().to_numpy(), np.full((2, 2), 0.5)
        )
        assert ForwardOperator(H_agg).is_sparse

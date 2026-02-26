"""Tests for sparse matrix support across Structure2D, Matrix, ForwardOperator,
CovarianceBuilder, and InverseProblem."""

import numpy as np
import pandas as pd
import pytest

from fips.covariance import (
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
    def test_sparse_flag_creates_sparse_dataframe(self):
        idx = _state_idx()
        block = _make_block(np.eye(4), "s", "s", idx, idx)
        M = Matrix([block], sparse=True)
        assert M.is_sparse

    def test_dense_flag_creates_dense_dataframe(self):
        idx = _state_idx()
        block = _make_block(np.eye(4), "s", "s", idx, idx)
        M = Matrix([block], sparse=False)
        assert not M.is_sparse

    def test_values_equal_after_sparsification(self):
        idx = _state_idx()
        data = np.diag([1.0, 0.0, 3.0, 0.0])
        block = _make_block(data, "s", "s", idx, idx)
        M_dense = Matrix([block])
        M_sparse = Matrix([block], sparse=True)
        assert np.allclose(M_dense.values, M_sparse.to_numpy())

    def test_to_sparse_roundtrip(self):
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
        idx = _state_idx()
        block = _make_block(np.eye(4), "s", "s", idx, idx)
        Ms1 = Matrix([block], sparse=True)
        Ms2 = Matrix([block], sparse=True)
        result = Ms1 + Ms2
        assert np.allclose(result.values, 2 * np.eye(4))

    def test_scale_preserves_sparsity(self):
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
    @pytest.fixture
    def H_data(self):
        return _make_sparse_H(m=3, n=4)

    @pytest.fixture
    def state_vec(self):
        idx = _state_idx(4)
        block = Block(pd.Series([1.0, 2.0, 3.0, 4.0], index=idx, name="state"))
        return Vector([block])

    def test_sparse_forward_operator_is_sparse(self, H_data):
        obs_idx = _obs_idx(3)
        state_idx = _state_idx(4)
        hblock = _make_block(H_data, "obs", "state", obs_idx, state_idx)
        H = ForwardOperator([hblock], sparse=True)
        assert H.is_sparse

    def test_sparse_convolve_matches_dense(self, H_data, state_vec):
        obs_idx = _obs_idx(3)
        state_idx = _state_idx(4)
        hblock = _make_block(H_data, "obs", "state", obs_idx, state_idx)

        H_dense = ForwardOperator([hblock])
        H_sparse = ForwardOperator([hblock], sparse=True)

        y_dense = H_dense.convolve(state_vec)
        y_sparse = H_sparse.convolve(state_vec)

        assert np.allclose(y_dense.to_numpy(), y_sparse.to_numpy())

    def test_sparse_convolve_returns_series(self, H_data, state_vec):
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
    @pytest.fixture
    def midx(self):
        return pd.MultiIndex.from_tuples(
            [(f"loc{i}", t) for i in range(3) for t in range(4)],
            names=["location", "time"],
        )

    def test_sparse_build_returns_sparse_dataframe(self, midx):
        # sparse=True lives on CovarianceBuilder, not ErrorComponent directly
        builder = CovarianceBuilder([DiagonalError("diag", variances=1.0)])
        df = builder.build(midx, sparse=True)
        assert any(isinstance(dt, pd.SparseDtype) for dt in df.dtypes)

    def test_sparse_and_dense_build_are_numerically_equal(self, midx):
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
        builder = CovarianceBuilder([DiagonalError("diag", variances=2.0)])
        df = builder.build(midx, sparse=True)
        assert any(isinstance(dt, pd.SparseDtype) for dt in df.dtypes)
        assert np.allclose(np.diag(df.sparse.to_dense().to_numpy()), 2.0)


# ---------------------------------------------------------------------------
# InverseProblem — end-to-end with sparse ForwardOperator
# ---------------------------------------------------------------------------


class TestSparseInverseProblem:
    @pytest.fixture
    def sparse_problem(self):
        """InverseProblem with a sparse forward operator."""
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
        """Same problem but reconstructed with a dense forward operator."""
        # Re-expose using the same underlying data
        return sparse_problem

    def test_forward_operator_can_be_made_sparse(self, sparse_problem):
        sparse_problem.forward_operator = sparse_problem.forward_operator.to_sparse(
            threshold=1e-10
        )
        assert sparse_problem.forward_operator.is_sparse

    def test_sparse_solve_produces_finite_posterior(self, sparse_problem):
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
        sparse_problem.forward_operator = sparse_problem.forward_operator.to_sparse(
            threshold=1e-10
        )
        sparse_problem.solve("bayesian")
        assert sparse_problem.posterior.shape[0] == sparse_problem.n_state
        assert sparse_problem.posterior_error.shape == (
            sparse_problem.n_state,
            sparse_problem.n_state,
        )

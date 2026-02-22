"""Test suite for fips.problem module."""

import warnings

import numpy as np
import pandas as pd
import pytest

from fips.covariance import CovarianceMatrix
from fips.operators import ForwardOperator
from fips.problem import InverseProblem
from fips.vector import Block, Vector
from tests.generate_data import generate_test_data


class TestInverseProblemCreation:
    """Tests for InverseProblem initialization."""

    def test_inverse_problem_basic_creation(
        self,
        simple_prior,
        simple_obs,
        simple_forward_operator,
        simple_prior_error,
        simple_modeldata_mismatch,
    ):
        """Test basic InverseProblem creation."""
        problem = InverseProblem(
            prior=simple_prior,
            obs=simple_obs,
            forward_operator=simple_forward_operator,
            prior_error=simple_prior_error,
            modeldata_mismatch=simple_modeldata_mismatch,
        )

        assert problem.prior is not None
        assert problem.obs is not None
        assert problem.forward_operator is not None

    def test_inverse_problem_with_vector_objects(self):
        """Test InverseProblem with Vector objects instead of Series."""
        # Use consistent dimension naming - both use 'id' as dimension name
        block_prior = Block(
            pd.Series([1.0, 2.0], index=pd.Index(["x", "y"], name="id"), name="state")
        )
        prior = Vector(data=[block_prior], name="prior")

        obs = pd.Series(
            [1.0, 2.0], index=pd.Index(["obs_0", "obs_1"], name="id"), name="obs"
        )

        # Create simple matrices - use 'id' for dimension name to match vectors
        state_labels = ["x", "y"]
        obs_labels = ["obs_0", "obs_1"]

        H = pd.DataFrame(
            [[1, 0.5], [0.5, 1]],
            index=pd.Index(obs_labels, name="id"),
            columns=pd.Index(state_labels, name="id"),
        )

        S_0 = pd.DataFrame(
            np.eye(2),
            index=pd.Index(state_labels, name="id"),
            columns=pd.Index(state_labels, name="id"),
        )
        S_z = pd.DataFrame(
            np.eye(2),
            index=pd.Index(obs_labels, name="id"),
            columns=pd.Index(obs_labels, name="id"),
        )

        problem = InverseProblem(
            obs=obs,
            prior=prior,
            forward_operator=H,
            modeldata_mismatch=S_z,
            prior_error=S_0,
        )

        assert isinstance(problem.prior, Vector)
        assert isinstance(problem.obs, Vector)

    def test_inverse_problem_stores_matrices(
        self,
        simple_prior,
        simple_obs,
        simple_forward_operator,
        simple_prior_error,
        simple_modeldata_mismatch,
    ):
        """Test that InverseProblem stores all matrices."""
        problem = InverseProblem(
            prior=simple_prior,
            obs=simple_obs,
            forward_operator=simple_forward_operator,
            prior_error=simple_prior_error,
            modeldata_mismatch=simple_modeldata_mismatch,
        )

        assert hasattr(problem, "forward_operator")
        assert hasattr(problem, "prior_error")
        assert hasattr(problem, "modeldata_mismatch")

    def test_inverse_problem_stores_vectors(
        self,
        simple_prior,
        simple_obs,
        simple_forward_operator,
        simple_prior_error,
        simple_modeldata_mismatch,
    ):
        """Test that InverseProblem stores all vectors."""
        problem = InverseProblem(
            prior=simple_prior,
            obs=simple_obs,
            forward_operator=simple_forward_operator,
            prior_error=simple_prior_error,
            modeldata_mismatch=simple_modeldata_mismatch,
        )

        assert hasattr(problem, "prior")
        assert hasattr(problem, "obs")


class TestInverseProblemProperties:
    """Tests for InverseProblem property accessors."""

    def test_prior_property(
        self,
        simple_prior,
        simple_obs,
        simple_forward_operator,
        simple_prior_error,
        simple_modeldata_mismatch,
    ):
        """Test prior property access."""
        problem = InverseProblem(
            prior=simple_prior,
            obs=simple_obs,
            forward_operator=simple_forward_operator,
            prior_error=simple_prior_error,
            modeldata_mismatch=simple_modeldata_mismatch,
        )

        prior = problem.prior
        assert isinstance(prior, Vector)
        assert prior.shape[0] == 3

    def test_obs_property(
        self,
        simple_prior,
        simple_obs,
        simple_forward_operator,
        simple_prior_error,
        simple_modeldata_mismatch,
    ):
        """Test obs property access."""
        problem = InverseProblem(
            prior=simple_prior,
            obs=simple_obs,
            forward_operator=simple_forward_operator,
            prior_error=simple_prior_error,
            modeldata_mismatch=simple_modeldata_mismatch,
        )

        obs = problem.obs
        assert isinstance(obs, Vector)
        assert obs.shape[0] == 4

    def test_forward_operator_property(
        self,
        simple_prior,
        simple_obs,
        simple_forward_operator,
        simple_prior_error,
        simple_modeldata_mismatch,
    ):
        """Test forward_operator property access."""
        problem = InverseProblem(
            prior=simple_prior,
            obs=simple_obs,
            forward_operator=simple_forward_operator,
            prior_error=simple_prior_error,
            modeldata_mismatch=simple_modeldata_mismatch,
        )

        H = problem.forward_operator
        from fips.operators import ForwardOperator

        assert isinstance(H, ForwardOperator)
        assert H.shape == (4, 3)

    def test_prior_error_property(
        self,
        simple_prior,
        simple_obs,
        simple_forward_operator,
        simple_prior_error,
        simple_modeldata_mismatch,
    ):
        """Test prior_error property access."""
        problem = InverseProblem(
            prior=simple_prior,
            obs=simple_obs,
            forward_operator=simple_forward_operator,
            prior_error=simple_prior_error,
            modeldata_mismatch=simple_modeldata_mismatch,
        )

        # Check that prior_error is accessible and correctly wrapped
        S_a = problem.prior_error
        assert isinstance(S_a, CovarianceMatrix)
        assert isinstance(S_a.data, pd.DataFrame)
        assert S_a.shape == (3, 3)

    def test_modeldata_mismatch_property(
        self,
        simple_prior,
        simple_obs,
        simple_forward_operator,
        simple_prior_error,
        simple_modeldata_mismatch,
    ):
        """Test modeldata_mismatch property access."""
        problem = InverseProblem(
            prior=simple_prior,
            obs=simple_obs,
            forward_operator=simple_forward_operator,
            prior_error=simple_prior_error,
            modeldata_mismatch=simple_modeldata_mismatch,
        )

        # Check that modeldata_mismatch is accessible and correctly wrapped
        S_z = problem.modeldata_mismatch
        assert isinstance(S_z, CovarianceMatrix)
        assert isinstance(S_z.data, pd.DataFrame)
        assert S_z.shape == (4, 4)


class TestInverseProblemMatrixHandling:
    """Tests for matrix handling in InverseProblem."""

    def test_matrix_wrapping_covariance(
        self,
        simple_prior,
        simple_obs,
        simple_forward_operator,
        simple_prior_error,
        simple_modeldata_mismatch,
    ):
        """Test that CovarianceMatrix objects are unwrapped."""
        from fips.matrix import MatrixBlock

        cov_prior = CovarianceMatrix(
            [MatrixBlock(simple_prior_error, row_block="state", col_block="state")]
        )
        cov_mismatch = CovarianceMatrix(
            [MatrixBlock(simple_modeldata_mismatch, row_block="obs", col_block="obs")]
        )

        problem = InverseProblem(
            prior=simple_prior,
            obs=simple_obs,
            forward_operator=simple_forward_operator,
            prior_error=cov_prior,
            modeldata_mismatch=cov_mismatch,
        )

        # Check that matrices are stored properly
        assert problem.prior_error is not None
        assert problem.modeldata_mismatch is not None

    def test_matrix_wrapping_forward_operator(
        self,
        simple_prior,
        simple_obs,
        simple_forward_operator,
        simple_prior_error,
        simple_modeldata_mismatch,
    ):
        """Test that ForwardOperator objects are unwrapped."""
        from fips.matrix import MatrixBlock

        forward = ForwardOperator(
            [MatrixBlock(simple_forward_operator, row_block="obs", col_block="state")]
        )

        problem = InverseProblem(
            prior=simple_prior,
            obs=simple_obs,
            forward_operator=forward,
            prior_error=simple_prior_error,
            modeldata_mismatch=simple_modeldata_mismatch,
        )

        assert problem.forward_operator is not None


class TestInverseProblemIndexAlignment:
    """Tests for index alignment in InverseProblem."""

    def test_mismatched_prior_indices_raises(self):
        """Test error when prior indices don't match."""
        prior = pd.Series(
            [1.0, 2.0], index=pd.Index(["a", "b"], name="id"), name="prior"
        )
        obs = pd.Series([1.0, 2.0], index=pd.Index(["o1", "o2"], name="id"), name="obs")

        # Forward operator uses different state indices
        H = pd.DataFrame(
            [[1, 0], [0, 1]],
            index=pd.Index(["o1", "o2"], name="id"),
            columns=pd.Index(["x", "y"], name="id"),
        )

        prior_idx = pd.Index(["a", "b"], name="id")
        obs_idx = pd.Index(["o1", "o2"], name="id")
        S_a = pd.DataFrame(np.eye(2), index=prior_idx, columns=prior_idx)
        S_z = pd.DataFrame(np.eye(2), index=obs_idx, columns=obs_idx)

        with pytest.raises(ValueError, match="does not overlap"):
            InverseProblem(
                obs=obs,
                prior=prior,
                forward_operator=H,
                modeldata_mismatch=S_z,
                prior_error=S_a,
            )

    def test_matrix_reindexing_fills_zeros(self):
        """Test that matrices are reindexed with zero-filling."""
        prior = pd.Series(
            [1.0, 2.0, 3.0], index=pd.Index(["a", "b", "c"], name="id"), name="prior"
        )
        obs = pd.Series([1.0, 2.0], index=pd.Index(["o1", "o2"], name="id"), name="obs")

        # Forward operator doesn't cover all state variables
        H = pd.DataFrame(
            [[1, 0.5, 0], [0.5, 1, 0.5]],
            index=pd.Index(["o1", "o2"], name="id"),
            columns=pd.Index(["a", "b", "c"], name="id"),
        )

        state_idx = pd.Index(["a", "b", "c"], name="id")
        obs_idx = pd.Index(["o1", "o2"], name="id")
        S_a = pd.DataFrame(np.eye(3), index=state_idx, columns=state_idx)
        S_z = pd.DataFrame(np.eye(2), index=obs_idx, columns=obs_idx)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            problem = InverseProblem(
                obs=obs,
                prior=prior,
                forward_operator=H,
                modeldata_mismatch=S_z,
                prior_error=S_a,
            )

            # H should still have the right shape
            assert problem.forward_operator.shape == (2, 3)


class TestInverseProblemFloatPrecision:
    """Tests for float precision handling."""

    def test_float_precision_rounding(self):
        """Test float precision rounding in InverseProblem."""
        prior = pd.Series(
            [1.0, 2.0], index=pd.Index([1.123456, 2.654321], name="id"), name="prior"
        )
        obs = pd.Series(
            [1.0, 2.0], index=pd.Index([3.111111, 4.222222], name="id"), name="obs"
        )

        H = pd.DataFrame(
            np.eye(2),
            index=pd.Index([3.111111, 4.222222], name="id"),
            columns=pd.Index([1.123456, 2.654321], name="id"),
        )

        state_idx = pd.Index([1.123456, 2.654321], name="id")
        obs_idx = pd.Index([3.111111, 4.222222], name="id")
        S_a = pd.DataFrame(np.eye(2), index=state_idx, columns=state_idx)
        S_z = pd.DataFrame(np.eye(2), index=obs_idx, columns=obs_idx)

        problem = InverseProblem(
            obs=obs,
            prior=prior,
            forward_operator=H,
            modeldata_mismatch=S_z,
            prior_error=S_a,
            round_index=2,
        )

        # Should have rounded indices
        assert problem.prior is not None


class TestInverseProblemDimensions:
    """Tests for dimension consistency."""

    def test_rectangular_problem(self):
        """Test with more observations than states (overdetermined)."""
        prior = pd.Series(
            np.ones(5),
            index=pd.Index([f"s{i}" for i in range(5)], name="id"),
            name="prior",
        )
        obs = pd.Series(
            np.ones(10),
            index=pd.Index([f"o{i}" for i in range(10)], name="id"),
            name="obs",
        )

        H = pd.DataFrame(
            np.ones((10, 5)),
            index=pd.Index([f"o{i}" for i in range(10)], name="id"),
            columns=pd.Index([f"s{i}" for i in range(5)], name="id"),
        )

        state_idx = pd.Index([f"s{i}" for i in range(5)], name="id")
        obs_idx = pd.Index([f"o{i}" for i in range(10)], name="id")
        S_a = pd.DataFrame(
            np.eye(5),
            index=state_idx,
            columns=state_idx,
        )
        S_z = pd.DataFrame(
            np.eye(10),
            index=obs_idx,
            columns=obs_idx,
        )

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            problem = InverseProblem(
                obs=obs,
                prior=prior,
                forward_operator=H,
                modeldata_mismatch=S_z,
                prior_error=S_a,
            )

            assert problem.forward_operator.shape == (10, 5)
            assert len(problem.prior.data) == 5
            assert len(problem.obs.data) == 10

    def test_square_problem(self):
        """Test with equal dimensions."""
        n = 7
        prior = pd.Series(
            np.ones(n),
            index=pd.Index([f"s{i}" for i in range(n)], name="id"),
            name="prior",
        )
        obs = pd.Series(
            np.ones(n),
            index=pd.Index([f"o{i}" for i in range(n)], name="id"),
            name="obs",
        )

        H = pd.DataFrame(
            np.eye(n),
            index=pd.Index([f"o{i}" for i in range(n)], name="id"),
            columns=pd.Index([f"s{i}" for i in range(n)], name="id"),
        )

        state_idx = pd.Index([f"s{i}" for i in range(n)], name="id")
        obs_idx = pd.Index([f"o{i}" for i in range(n)], name="id")
        S_a = pd.DataFrame(
            np.eye(n),
            index=state_idx,
            columns=state_idx,
        )
        S_z = pd.DataFrame(
            np.eye(n),
            index=obs_idx,
            columns=obs_idx,
        )

        problem = InverseProblem(
            obs=obs,
            prior=prior,
            forward_operator=H,
            modeldata_mismatch=S_z,
            prior_error=S_a,
        )

        assert problem.forward_operator.shape == (n, n)

    def test_underdetermined_problem(self):
        """Test with fewer observations than states (underdetermined)."""
        prior = pd.Series(
            np.ones(10),
            index=pd.Index([f"s{i}" for i in range(10)], name="id"),
            name="prior",
        )
        obs = pd.Series(
            np.ones(5),
            index=pd.Index([f"o{i}" for i in range(5)], name="id"),
            name="obs",
        )

        H = pd.DataFrame(
            np.ones((5, 10)),
            index=pd.Index([f"o{i}" for i in range(5)], name="id"),
            columns=pd.Index([f"s{i}" for i in range(10)], name="id"),
        )

        state_idx = pd.Index([f"s{i}" for i in range(10)], name="id")
        obs_idx = pd.Index([f"o{i}" for i in range(5)], name="id")
        S_a = pd.DataFrame(
            np.eye(10),
            index=state_idx,
            columns=state_idx,
        )
        S_z = pd.DataFrame(
            np.eye(5),
            index=obs_idx,
            columns=obs_idx,
        )

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            problem = InverseProblem(
                obs=obs,
                prior=prior,
                forward_operator=H,
                modeldata_mismatch=S_z,
                prior_error=S_a,
            )

            assert problem.forward_operator.shape == (5, 10)


class TestInverseProblemInitializerNone:
    """Tests for initializer being None."""

    def test_estimator_initially_none(
        self,
        simple_prior,
        simple_obs,
        simple_forward_operator,
        simple_prior_error,
        simple_modeldata_mismatch,
    ):
        """Test that estimator is initially None."""
        problem = InverseProblem(
            prior=simple_prior,
            obs=simple_obs,
            forward_operator=simple_forward_operator,
            prior_error=simple_prior_error,
            modeldata_mismatch=simple_modeldata_mismatch,
        )

        assert problem._estimator is None


class TestInverseProblemSolve:
    """Tests for solving an inverse problem end-to-end."""

    def test_solve_bayesian_with_generated_data(self):
        """Generate synthetic data, build problem, and solve with bayesian estimator."""
        data = generate_test_data(n_state=5, n_obs=8, seed=1, correlation_len=1.5)

        # Directly pass generated data (Series/DataFrame/CovarianceMatrix)
        problem = InverseProblem(
            prior=data["prior"],
            obs=data["obs"],
            forward_operator=data["forward_operator"],
            prior_error=data["prior_error"],
            modeldata_mismatch=data["modeldata_mismatch"],
        )

        problem.solve(estimator="bayesian")

        # Estimator is attached
        assert problem._estimator is not None

        # Check estimator has the results
        estimator = problem.estimator
        assert estimator.x_hat.shape[0] == len(data["prior"])
        assert estimator.y_hat.shape[0] == len(data["obs"])
        assert estimator.S_hat.shape == (
            len(data["prior"]),
            len(data["prior"]),
        )

        # No NaNs or infs in outputs
        assert np.isfinite(estimator.x_hat).all()
        assert np.isfinite(estimator.y_hat).all()
        assert np.isfinite(estimator.S_hat).all()

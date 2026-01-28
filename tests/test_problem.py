"""Test suite for fips.problem module."""

import warnings

import numpy as np
import pandas as pd

from fips.matrices import CovarianceMatrix, ForwardOperator
from fips.problem import InverseProblem
from fips.vectors import Block, Vector
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
        block_prior = Block(pd.Series([1.0, 2.0], index=["x", "y"], name="state"))
        prior = Vector(name="prior", blocks=[block_prior])

        obs = pd.Series([1.0, 2.0], index=["obs_0", "obs_1"])

        # Forward operator must have columns matching the prior's assembled index structure
        # Since prior is a Vector, its index will be promoted to include the block level
        prior_idx = prior.data.index

        H = pd.DataFrame(
            [[1, 0.5], [0.5, 1]], index=["obs_0", "obs_1"], columns=prior_idx
        )

        S_0 = pd.DataFrame(np.eye(2), index=prior_idx, columns=prior_idx)
        S_z = pd.DataFrame(
            np.eye(2), index=["obs_0", "obs_1"], columns=["obs_0", "obs_1"]
        )

        problem = InverseProblem(prior, obs, H, S_0, S_z)

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
        assert prior.n == 3

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
        assert obs.n == 4

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
        from fips.matrices import ForwardOperator

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
        cov_prior = CovarianceMatrix(simple_prior_error)
        cov_mismatch = CovarianceMatrix(simple_modeldata_mismatch)

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
        forward = ForwardOperator(simple_forward_operator)

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

    def test_mismatched_prior_indices_warning(self):
        """Test warning when prior indices don't match."""
        prior = pd.Series([1.0, 2.0], index=["a", "b"], name="prior")
        obs = pd.Series([1.0, 2.0], index=["o1", "o2"], name="obs")

        # Forward operator uses different state indices
        H = pd.DataFrame([[1, 0], [0, 1]], index=["o1", "o2"], columns=["x", "y"])

        S_a = pd.DataFrame(np.eye(2), index=["a", "b"], columns=["a", "b"])
        S_z = pd.DataFrame(np.eye(2), index=["o1", "o2"], columns=["o1", "o2"])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            InverseProblem(prior, obs, H, S_a, S_z)

            # Should generate warning about partial overlap
            warning_messages = [str(warn.message) for warn in w]
            assert any(
                "Partial overlap" in msg or "No overlap" in msg
                for msg in warning_messages
            )

    def test_matrix_reindexing_fills_zeros(self):
        """Test that matrices are reindexed with zero-filling."""
        prior = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"], name="prior")
        obs = pd.Series([1.0, 2.0], index=["o1", "o2"], name="obs")

        # Forward operator doesn't cover all state variables
        H = pd.DataFrame(
            [[1, 0.5, 0], [0.5, 1, 0.5]], index=["o1", "o2"], columns=["a", "b", "c"]
        )

        S_a = pd.DataFrame(np.eye(3), index=["a", "b", "c"], columns=["a", "b", "c"])
        S_z = pd.DataFrame(np.eye(2), index=["o1", "o2"], columns=["o1", "o2"])

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            problem = InverseProblem(prior, obs, H, S_a, S_z)

            # H should still have the right shape
            assert problem.forward_operator.shape == (2, 3)


class TestInverseProblemFloatPrecision:
    """Tests for float precision handling."""

    def test_float_precision_rounding(self):
        """Test float precision rounding in InverseProblem."""
        prior = pd.Series([1.0, 2.0], index=[1.123456, 2.654321], name="prior")
        obs = pd.Series([1.0, 2.0], index=[3.111111, 4.222222], name="obs")

        H = pd.DataFrame(
            np.eye(2), index=[3.111111, 4.222222], columns=[1.123456, 2.654321]
        )

        S_a = pd.DataFrame(
            np.eye(2), index=[1.123456, 2.654321], columns=[1.123456, 2.654321]
        )
        S_z = pd.DataFrame(
            np.eye(2), index=[3.111111, 4.222222], columns=[3.111111, 4.222222]
        )

        problem = InverseProblem(prior, obs, H, S_a, S_z, float_precision=2)

        # Should have rounded indices
        assert problem.prior is not None


class TestInverseProblemDimensions:
    """Tests for dimension consistency."""

    def test_rectangular_problem(self):
        """Test with more observations than states (overdetermined)."""
        prior = pd.Series(np.ones(5), index=[f"s{i}" for i in range(5)], name="prior")
        obs = pd.Series(np.ones(10), index=[f"o{i}" for i in range(10)], name="obs")

        H = pd.DataFrame(
            np.ones((10, 5)),
            index=[f"o{i}" for i in range(10)],
            columns=[f"s{i}" for i in range(5)],
        )

        S_a = pd.DataFrame(
            np.eye(5),
            index=[f"s{i}" for i in range(5)],
            columns=[f"s{i}" for i in range(5)],
        )
        S_z = pd.DataFrame(
            np.eye(10),
            index=[f"o{i}" for i in range(10)],
            columns=[f"o{i}" for i in range(10)],
        )

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            problem = InverseProblem(prior, obs, H, S_a, S_z)

            assert problem.forward_operator.shape == (10, 5)
            assert problem.prior.n == 5
            assert problem.obs.n == 10

    def test_square_problem(self):
        """Test with equal dimensions."""
        n = 7
        prior = pd.Series(np.ones(n), index=[f"s{i}" for i in range(n)], name="prior")
        obs = pd.Series(np.ones(n), index=[f"o{i}" for i in range(n)], name="obs")

        H = pd.DataFrame(
            np.eye(n),
            index=[f"o{i}" for i in range(n)],
            columns=[f"s{i}" for i in range(n)],
        )

        S_a = pd.DataFrame(
            np.eye(n),
            index=[f"s{i}" for i in range(n)],
            columns=[f"s{i}" for i in range(n)],
        )
        S_z = pd.DataFrame(
            np.eye(n),
            index=[f"o{i}" for i in range(n)],
            columns=[f"o{i}" for i in range(n)],
        )

        problem = InverseProblem(prior, obs, H, S_a, S_z)

        assert problem.forward_operator.shape == (n, n)

    def test_underdetermined_problem(self):
        """Test with fewer observations than states (underdetermined)."""
        prior = pd.Series(np.ones(10), index=[f"s{i}" for i in range(10)], name="prior")
        obs = pd.Series(np.ones(5), index=[f"o{i}" for i in range(5)], name="obs")

        H = pd.DataFrame(
            np.ones((5, 10)),
            index=[f"o{i}" for i in range(5)],
            columns=[f"s{i}" for i in range(10)],
        )

        S_a = pd.DataFrame(
            np.eye(10),
            index=[f"s{i}" for i in range(10)],
            columns=[f"s{i}" for i in range(10)],
        )
        S_z = pd.DataFrame(
            np.eye(5),
            index=[f"o{i}" for i in range(5)],
            columns=[f"o{i}" for i in range(5)],
        )

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            problem = InverseProblem(prior, obs, H, S_a, S_z)

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
            forward_operator=data["jacobian"],
            prior_error=data["prior_error"],
            modeldata_mismatch=data["modeldata_mismatch"],
        )

        result = problem.solve(estimator="bayesian")

        # Estimator is attached
        assert problem._estimator is not None

        # Result keys and shapes
        assert set(result.keys()) == {"posterior", "posterior_error", "posterior_obs"}
        assert result["posterior"].n == len(data["prior"])
        assert result["posterior_obs"].n == len(data["obs"])
        assert result["posterior_error"].values.shape == (
            len(data["prior"]),
            len(data["prior"]),
        )

        # No NaNs or infs in outputs
        assert np.isfinite(result["posterior"].values).all()
        assert np.isfinite(result["posterior_obs"].values).all()
        assert np.isfinite(result["posterior_error"].values).all()

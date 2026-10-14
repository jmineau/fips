"""Test suite for fips.problem module."""

import warnings

import numpy as np
import pandas as pd
import pytest

from fips.covariance import CovarianceMatrix
from fips.matrix import MatrixBlock
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

    @pytest.mark.parametrize("scalar", [0.5, 2, np.float32(0.5)])
    def test_scalar_covariances_are_diagonals(
        self, scalar, simple_prior, simple_obs, simple_forward_operator
    ):
        """A scalar covariance is that variance for every element, labeled by the vector."""
        problem = InverseProblem(
            prior=simple_prior,
            obs=simple_obs,
            forward_operator=simple_forward_operator,
            prior_error=scalar,
            modeldata_mismatch=scalar,
        )

        assert isinstance(problem.prior_error, CovarianceMatrix)
        pd.testing.assert_index_equal(problem.prior_error.index, problem.prior.index)
        pd.testing.assert_index_equal(
            problem.modeldata_mismatch.index, problem.obs.index
        )
        np.testing.assert_allclose(problem.prior_error.values, scalar * np.eye(3))
        np.testing.assert_allclose(
            problem.modeldata_mismatch.values, scalar * np.eye(4)
        )

    def test_scalar_covariances_solve_like_the_matrices(
        self, simple_prior, simple_obs, simple_forward_operator
    ):
        """A problem given scalars has the posterior of one given the diagonal matrices."""
        state = simple_prior.index
        obs = simple_obs.index
        common = {
            "prior": simple_prior,
            "obs": simple_obs,
            "forward_operator": simple_forward_operator,
        }

        from_scalars = InverseProblem(
            **common, prior_error=0.5, modeldata_mismatch=0.25
        ).solve("bayesian")
        from_matrices = InverseProblem(
            **common,
            prior_error=pd.DataFrame(0.5 * np.eye(3), index=state, columns=state),
            modeldata_mismatch=pd.DataFrame(0.25 * np.eye(4), index=obs, columns=obs),
        ).solve("bayesian")

        pd.testing.assert_series_equal(
            from_scalars.posterior.data, from_matrices.posterior.data
        )

    @pytest.mark.parametrize("wrap", [lambda s: s, Vector], ids=["series", "vector"])
    def test_labeled_variances_are_diagonals(
        self, wrap, simple_prior, simple_obs, simple_forward_operator
    ):
        """A Series or a Vector with a block level is the diagonal, matched by label."""
        prior = Vector(simple_prior)
        obs = Vector(simple_obs)
        # Given in another order than the vectors: the labels decide, not the position
        prior_var = pd.Series([3.0, 2.0, 1.0], index=prior.index[::-1], name="var")
        obs_var = pd.Series([4.0, 3.0, 2.0, 1.0], index=obs.index[::-1], name="var")

        problem = InverseProblem(
            prior=simple_prior,
            obs=simple_obs,
            forward_operator=simple_forward_operator,
            prior_error=wrap(prior_var),
            modeldata_mismatch=wrap(obs_var),
        )

        np.testing.assert_array_equal(
            problem.prior_error.values, np.diag([1.0, 2.0, 3.0])
        )
        np.testing.assert_array_equal(
            problem.modeldata_mismatch.values, np.diag([1.0, 2.0, 3.0, 4.0])
        )

    def test_1d_array_covariance_is_refused(
        self, simple_prior, simple_obs, simple_forward_operator
    ):
        """An unlabeled 1D array is not matched to the state by position."""
        with pytest.raises(ValueError, match="takes 2D data, got 1D data"):
            InverseProblem(
                prior=simple_prior,
                obs=simple_obs,
                forward_operator=simple_forward_operator,
                prior_error=np.array([1.0, 2.0, 3.0]),
                modeldata_mismatch=0.5,
            )

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

    @pytest.mark.parametrize("kind", ["none", "scalar", "series"])
    def test_constant_is_a_vector_or_none(
        self,
        simple_prior,
        simple_obs,
        simple_forward_operator,
        simple_prior_error,
        simple_modeldata_mismatch,
        kind,
    ):
        """The constant is stored as a Vector, whatever it was given as, or None."""
        constant = {
            "none": None,
            "scalar": 2.5,
            "series": simple_obs * 2,
        }[kind]

        problem = InverseProblem(
            prior=simple_prior,
            obs=simple_obs,
            forward_operator=simple_forward_operator,
            prior_error=simple_prior_error,
            modeldata_mismatch=simple_modeldata_mismatch,
            constant=constant,
        )

        if kind == "none":
            assert problem.constant is None
        else:
            assert isinstance(problem.constant, Vector)
            expected = [2.5] * 4 if kind == "scalar" else [3.0, 5.0, 7.0, 9.0]
            np.testing.assert_allclose(problem.constant.values, expected)


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

    def test_prior_obs_error_is_h_s0_ht_on_the_obs_index(self):
        """``prior_obs_error`` wraps H S_0 H^T as a covariance on the observations."""
        data = generate_test_data(n_state=5, n_obs=8, seed=2, correlation_len=1.5)
        problem = InverseProblem(
            prior=data["prior"],
            obs=data["obs"],
            forward_operator=data["forward_operator"],
            prior_error=data["prior_error"],
            modeldata_mismatch=data["modeldata_mismatch"],
        )
        with pytest.raises(RuntimeError, match="not been solved"):
            _ = problem.prior_obs_error
        problem.solve(estimator="bayesian")

        S_y0 = problem.prior_obs_error
        assert isinstance(S_y0, CovarianceMatrix)
        assert S_y0.index.equals(problem.obs_index)
        assert S_y0.columns.equals(problem.obs_index)
        est = problem.estimator
        expected = est.H @ est.S_0 @ est.H.T
        np.testing.assert_allclose(S_y0.to_numpy(), expected)
        np.testing.assert_allclose(S_y0.to_numpy(), S_y0.to_numpy().T)
        assert (np.diag(S_y0.to_numpy()) >= 0).all()


class TestInverseProblemLabelChecks:
    """Duplicate labels (#78) and unknown block names (#80) raise."""

    @staticmethod
    def _rounding_problem(round_index=6):
        """Build the problem of #78: two depths that differ past the 6th decimal."""
        state = pd.Index([10.1234567, 10.1234568, 20.0], name="depth_m")
        prior = pd.Series([1.0, 2.0, 3.0], index=state, name="density")
        obs_idx = pd.Index([0, 1], name="obs_id")
        obs = pd.Series([1.0, 2.0], index=obs_idx, name="gravity")
        H = pd.DataFrame(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]], index=obs_idx, columns=state
        )
        return InverseProblem(
            obs=obs,
            prior=prior,
            forward_operator=H,
            prior_error=1.0,
            modeldata_mismatch=1.0,
            round_index=round_index,
        )

    def test_issue_78_labels_merged_by_rounding_raise(self):
        """The reproduction of #78 raises and names the labels that merged."""
        with pytest.raises(ValueError) as excinfo:
            self._rounding_problem()
        message = str(excinfo.value)
        assert "round_index=6" in message
        assert "state index" in message
        assert "10.1234567" in message
        assert "10.1234568" in message
        assert "10.123457" in message

    def test_without_rounding_the_labels_stay_distinct(self):
        """With round_index=None the three labels of #78 are kept."""
        problem = self._rounding_problem(round_index=None)
        assert problem.n_state == 3
        assert not problem.state_index.has_duplicates

    def test_obs_labels_merged_by_rounding_raise(self):
        """Rounding that merges two observation labels raises too."""
        state = pd.Index(["a", "b"], name="id")
        obs_idx = pd.Index([0.5, 0.50000001], name="time")
        prior = pd.Series([1.0, 2.0], index=state, name="state")
        obs = pd.Series([1.0, 2.0], index=obs_idx, name="obs")
        H = pd.DataFrame(np.eye(2), index=obs_idx, columns=state)
        with pytest.raises(
            ValueError, match=r"round_index=6 merges .* observation index.*0\.5"
        ):
            InverseProblem(
                obs=obs,
                prior=prior,
                forward_operator=H,
                prior_error=1.0,
                modeldata_mismatch=1.0,
            )

    @pytest.mark.parametrize("round_index", [None, 6])
    @pytest.mark.parametrize("space", ["state", "observation"])
    def test_duplicate_input_labels_raise(self, space, round_index):
        """A label given twice raises, with or without rounding."""
        unique = pd.Index(["a", "b", "c"], name="id")
        repeated = pd.Index(["a", "b", "b"], name="id")
        state = repeated if space == "state" else unique
        obs_idx = repeated if space == "observation" else unique
        prior = pd.Series([1.0, 2.0, 3.0], index=state, name="state")
        obs = pd.Series([1.0, 2.0, 3.0], index=obs_idx, name="obs")
        H = pd.DataFrame(np.eye(3), index=obs_idx, columns=state)
        with pytest.raises(
            ValueError, match=rf"The {space} index has duplicate labels: \('\w+', 'b'\)"
        ):
            InverseProblem(
                obs=obs,
                prior=prior,
                forward_operator=H,
                prior_error=1.0,
                modeldata_mismatch=1.0,
                round_index=round_index,
            )

    @staticmethod
    def _blocks_problem(typo=None, cells=(1, 2)):
        """
        Build a problem with two state blocks and two obs blocks.

        ``typo`` is ``(argument, axis, block)``: in that argument's matrix,
        the block named ``block`` gets an ``x`` appended on ``axis`` ("row"
        or "column"). ``cells`` are the flux cells the Jacobian covers.
        """
        flux = pd.Index([1, 2], name="cell")
        bias = pd.Index(["A"], name="site")
        obs_a = pd.Index([0, 1], name="obs_id")
        obs_b = pd.Index([2], name="obs_id")
        prior = Vector(
            [
                Block(pd.Series([1.0, 2.0], index=flux, name="flux")),
                Block(pd.Series([0.0], index=bias, name="bias")),
            ]
        )
        obs = Vector(
            [
                Block(pd.Series([5.0, 6.0], index=obs_a, name="a")),
                Block(pd.Series([7.0], index=obs_b, name="b")),
            ]
        )
        covered = pd.Index(list(cells), name="cell")
        H = {
            ("a", "flux"): pd.DataFrame(1.0, index=obs_a, columns=covered),
            ("b", "flux"): pd.DataFrame(1.0, index=obs_b, columns=covered),
            ("a", "bias"): pd.DataFrame(1.0, index=obs_a, columns=bias),
            ("b", "bias"): pd.DataFrame(1.0, index=obs_b, columns=bias),
        }
        S_0 = {
            ("flux", "flux"): pd.DataFrame(np.eye(2), index=flux, columns=flux),
            ("bias", "bias"): pd.DataFrame([[1.0]], index=bias, columns=bias),
        }
        S_z = {
            ("a", "a"): pd.DataFrame(np.eye(2), index=obs_a, columns=obs_a),
            ("b", "b"): pd.DataFrame([[1.0]], index=obs_b, columns=obs_b),
        }
        matrices = {
            "forward_operator": H,
            "prior_error": S_0,
            "modeldata_mismatch": S_z,
        }
        if typo is not None:
            argument, axis, block = typo
            blocks = matrices[argument]
            for row, col in list(blocks):
                new_row = row + "x" if axis == "row" and row == block else row
                new_col = col + "x" if axis == "column" and col == block else col
                blocks[new_row, new_col] = blocks.pop((row, col))
        return InverseProblem(obs=obs, prior=prior, **matrices)

    def test_issue_80_unknown_col_block_raises(self):
        """The reproduction of #80 raises and names the unknown and known blocks."""
        cells = pd.Index([1, 2], name="cell")
        sites = pd.Index(["A"], name="site")
        oidx = pd.Index([0], name="obs_id")
        prior = Vector(
            [
                Block(pd.Series([1.0, 2.0], index=cells, name="flux")),
                Block(pd.Series([0.0], index=sites, name="bias")),
            ]
        )
        obs = pd.Series([5.0], index=oidx, name="conc")
        H = [
            MatrixBlock(
                pd.DataFrame([[1.0, 1.0]], index=oidx, columns=cells),
                row_block="conc",
                col_block="fluxx",
            ),
            MatrixBlock(
                pd.DataFrame([[1.0]], index=oidx, columns=sites),
                row_block="conc",
                col_block="bias",
            ),
        ]
        with pytest.raises(ValueError) as excinfo:
            InverseProblem(
                obs=obs,
                prior=prior,
                forward_operator=H,
                prior_error=1.0,
                modeldata_mismatch=1.0,
            )
        message = str(excinfo.value)
        assert "forward_operator has column blocks" in message
        assert "'fluxx'" in message
        assert "The known blocks are 'flux', 'bias'" in message
        assert "col_block" in message

    @pytest.mark.parametrize(
        ("argument", "axis", "block", "known"),
        [
            ("forward_operator", "row", "a", "'a', 'b'"),
            ("forward_operator", "column", "flux", "'flux', 'bias'"),
            ("prior_error", "row", "bias", "'flux', 'bias'"),
            ("prior_error", "column", "flux", "'flux', 'bias'"),
            ("modeldata_mismatch", "row", "b", "'a', 'b'"),
            ("modeldata_mismatch", "column", "a", "'a', 'b'"),
        ],
    )
    def test_unknown_block_name_raises(self, argument, axis, block, known):
        """A misspelled row_block or col_block raises for H, S_0 and S_z."""
        argument_name = "row_block" if axis == "row" else "col_block"
        with pytest.raises(ValueError) as excinfo:
            self._blocks_problem(typo=(argument, axis, block))
        message = str(excinfo.value)
        assert f"{argument} has {axis} blocks" in message
        assert f"'{block}x'" in message
        assert f"The known blocks are {known}" in message
        assert argument_name in message

    def test_known_blocks_build_without_warning(self):
        """A problem whose blocks all match builds silently, as before."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            problem = self._blocks_problem()
        assert problem.forward_operator.shape == (3, 3)
        assert problem.forward_operator.values.tolist() == [[1.0] * 3] * 3

    def test_labels_missing_within_a_known_block_are_zero_filled(self):
        """A Jacobian that covers only some cells of a block is still filled with zeros."""
        with pytest.warns(UserWarning, match="Partial overlap"):
            problem = self._blocks_problem(cells=(2,))
        H = problem.forward_operator
        assert H.shape == (3, 3)
        assert H["a", "flux"].to_numpy().tolist() == [[0.0, 1.0], [0.0, 1.0]]
        assert H["b", "flux"].to_numpy().tolist() == [[0.0, 1.0]]
        assert H["a", "bias"].to_numpy().tolist() == [[1.0], [1.0]]

"""Test suite for fips.estimators module."""

import numpy as np
import pytest

from fips.estimators import Estimator


class ConcreteEstimator(Estimator):
    """Concrete Estimator implementation for testing."""

    @property
    def x_hat(self) -> np.ndarray:
        """Posterior state estimate (dummy implementation)."""
        return self.x_0 + np.random.randn(self.n_x) * 0.01

    @property
    def S_hat(self) -> np.ndarray:
        """Posterior error covariance (dummy implementation)."""
        return self.S_0 * 0.5

    def cost(self, x: np.ndarray) -> float:
        """Cost function (dummy implementation)."""
        residual = self.residual(x)
        return float(np.sum(residual**2))


class TestEstimatorBasics:
    """Tests for basic Estimator functionality."""

    def test_estimator_initialization(self):
        """Test basic Estimator initialization."""
        z = np.array([1.0, 2.0, 3.0])
        x_0 = np.array([1.0, 2.0])
        H = np.array([[1.0, 0.5], [0.5, 1.0], [1.0, 1.0]])
        S_0 = np.eye(2)
        S_z = np.eye(3)

        estimator = ConcreteEstimator(z, x_0, H, S_0, S_z)

        assert estimator.n_z == 3
        assert estimator.n_x == 2
        assert isinstance(estimator.z, np.ndarray)

    def test_estimator_data_types(self):
        """Test that Estimator converts inputs to float."""
        z = np.array([1, 2, 3], dtype=int)
        x_0 = np.array([1, 2], dtype=int)
        H = np.array([[1, 0], [0, 1], [1, 1]], dtype=int)
        S_0 = np.eye(2, dtype=int)
        S_z = np.eye(3, dtype=int)

        estimator = ConcreteEstimator(z, x_0, H, S_0, S_z)

        assert estimator.z.dtype == float
        assert estimator.x_0.dtype == float
        assert estimator.H.dtype == float

    def test_estimator_with_constant_data(self):
        """Test Estimator with constant data."""
        z = np.array([1.0, 2.0])
        x_0 = np.array([1.0, 1.0])
        H = np.eye(2)
        S_0 = np.eye(2)
        S_z = np.eye(2)
        c = 0.5

        estimator = ConcreteEstimator(z, x_0, H, S_0, S_z, c=c)

        assert estimator.c.shape == (2,)
        assert np.allclose(estimator.c, [0.5, 0.5])

    def test_estimator_with_array_constant(self):
        """Test Estimator with array constant data."""
        z = np.array([1.0, 2.0])
        x_0 = np.array([1.0, 1.0])
        H = np.eye(2)
        S_0 = np.eye(2)
        S_z = np.eye(2)
        c = np.array([0.1, 0.2])

        estimator = ConcreteEstimator(z, x_0, H, S_0, S_z, c=c)

        assert np.allclose(estimator.c, [0.1, 0.2])

    def test_estimator_with_none_constant(self):
        """Test Estimator with None constant (defaults to 0)."""
        z = np.array([1.0, 2.0])
        x_0 = np.array([1.0, 1.0])
        H = np.eye(2)
        S_0 = np.eye(2)
        S_z = np.eye(2)

        estimator = ConcreteEstimator(z, x_0, H, S_0, S_z, c=None)

        assert np.allclose(estimator.c, [0.0, 0.0])

    def test_estimator_invalid_constant_type(self):
        """Test that invalid constant type raises error."""
        z = np.array([1.0, 2.0])
        x_0 = np.array([1.0, 1.0])
        H = np.eye(2)
        S_0 = np.eye(2)
        S_z = np.eye(2)

        with pytest.raises(TypeError):
            Estimator(z, x_0, H, S_0, S_z, c="invalid")


class TestEstimatorForward:
    """Tests for forward model calculation."""

    def test_forward_basic(self):
        """Test basic forward model calculation."""
        z = np.array([1.0, 2.0])
        x_0 = np.array([1.0, 2.0])
        H = np.array([[1.0, 2.0], [3.0, 4.0]])
        S_0 = np.eye(2)
        S_z = np.eye(2)

        estimator = ConcreteEstimator(z, x_0, H, S_0, S_z, c=0.0)

        # Forward of x_0: H @ x_0
        # [[1, 2], [3, 4]] @ [1, 2] = [5, 11]
        result = estimator.forward(x_0)

        expected = np.array([5.0, 11.0])
        assert np.allclose(result, expected)

    def test_forward_with_constant(self):
        """Test forward model with constant term."""
        z = np.array([1.0, 2.0])
        x_0 = np.array([1.0, 2.0])
        H = np.eye(2)
        S_0 = np.eye(2)
        S_z = np.eye(2)
        c = np.array([0.5, 1.0])

        estimator = ConcreteEstimator(z, x_0, H, S_0, S_z, c=c)
        result = estimator.forward(x_0)

        # I @ [1, 2] + [0.5, 1.0] = [1.5, 3.0]
        expected = np.array([1.5, 3.0])
        assert np.allclose(result, expected)

    def test_forward_output_shape(self):
        """Test that forward output has correct shape."""
        z = np.ones(10)
        x_0 = np.ones(5)
        H = np.ones((10, 5))
        S_0 = np.eye(5)
        S_z = np.eye(10)

        estimator = ConcreteEstimator(z, x_0, H, S_0, S_z)
        result = estimator.forward(x_0)

        assert result.shape == (10,)


class TestEstimatorResidual:
    """Tests for residual calculation."""

    def test_residual_basic(self):
        """Test basic residual calculation."""
        z = np.array([5.0, 11.0])
        x_0 = np.array([1.0, 2.0])
        H = np.array([[1.0, 2.0], [3.0, 4.0]])
        S_0 = np.eye(2)
        S_z = np.eye(2)

        estimator = ConcreteEstimator(z, x_0, H, S_0, S_z, c=0.0)

        # Forward of x_0: [5, 11]
        # Residual: z - forward = [5, 11] - [5, 11] = [0, 0]
        result = estimator.residual(x_0)

        assert np.allclose(result, [0.0, 0.0])

    def test_residual_with_misfit(self):
        """Test residual with actual misfit."""
        z = np.array([6.0, 12.0])
        x_0 = np.array([1.0, 2.0])
        H = np.array([[1.0, 2.0], [3.0, 4.0]])
        S_0 = np.eye(2)
        S_z = np.eye(2)

        estimator = ConcreteEstimator(z, x_0, H, S_0, S_z, c=0.0)
        result = estimator.residual(x_0)

        # Forward of x_0: [5, 11]
        # Residual: [6, 12] - [5, 11] = [1, 1]
        assert np.allclose(result, [1.0, 1.0])

    def test_residual_shape(self):
        """Test residual has correct shape."""
        z = np.random.randn(8)
        x_0 = np.random.randn(4)
        H = np.random.randn(8, 4)
        S_0 = np.eye(4)
        S_z = np.eye(8)

        estimator = ConcreteEstimator(z, x_0, H, S_0, S_z)
        result = estimator.residual(x_0)

        assert result.shape == z.shape


class TestEstimatorProperties:
    """Tests for Estimator properties and dimensions."""

    def test_estimator_dimensions(self):
        """Test that estimator correctly tracks dimensions."""
        z = np.ones(7)
        x_0 = np.ones(4)
        H = np.ones((7, 4))
        S_0 = np.eye(4)
        S_z = np.eye(7)

        estimator = ConcreteEstimator(z, x_0, H, S_0, S_z)

        assert estimator.n_z == 7
        assert estimator.n_x == 4

    def test_estimator_matrix_shapes(self):
        """Test that matrices have correct shapes."""
        z = np.ones(5)
        x_0 = np.ones(3)
        H = np.ones((5, 3))
        S_0 = np.ones((3, 3))
        S_z = np.ones((5, 5))

        estimator = ConcreteEstimator(z, x_0, H, S_0, S_z)

        assert estimator.H.shape == (5, 3)
        assert estimator.S_0.shape == (3, 3)
        assert estimator.S_z.shape == (5, 5)

    def test_estimator_with_rectangular_forward_operator(self):
        """Test with rectangular forward operator."""
        z = np.ones(20)  # More observations
        x_0 = np.ones(5)  # Fewer states
        H = np.ones((20, 5))
        S_0 = np.eye(5)
        S_z = np.eye(20)

        estimator = ConcreteEstimator(z, x_0, H, S_0, S_z)

        assert estimator.H.shape[0] > estimator.H.shape[1]


class TestEstimatorEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_estimator_with_unit_matrices(self):
        """Test with identity matrices."""
        z = np.array([1.0, 2.0, 3.0])
        x_0 = np.array([1.0, 2.0, 3.0])
        H = np.eye(3)
        S_0 = np.eye(3)
        S_z = np.eye(3)

        estimator = ConcreteEstimator(z, x_0, H, S_0, S_z)

        # Forward should be identity
        result = estimator.forward(x_0)
        assert np.allclose(result, x_0)

    def test_estimator_with_zero_observation(self):
        """Test with zero observations."""
        z = np.zeros(5)
        x_0 = np.ones(3)
        H = np.ones((5, 3))
        S_0 = np.eye(3)
        S_z = np.eye(5)

        estimator = ConcreteEstimator(z, x_0, H, S_0, S_z)

        assert len(estimator.z) == 5
        assert np.allclose(estimator.z, 0)

    def test_estimator_large_dimensions(self):
        """Test with large dimensions."""
        n_z, n_x = 100, 50
        z = np.random.randn(n_z)
        x_0 = np.random.randn(n_x)
        H = np.random.randn(n_z, n_x)
        S_0 = np.eye(n_x)
        S_z = np.eye(n_z)

        estimator = ConcreteEstimator(z, x_0, H, S_0, S_z)

        assert estimator.n_z == n_z
        assert estimator.n_x == n_x

    def test_estimator_with_small_values(self):
        """Test numerical stability with small values."""
        z = np.array([1e-6, 2e-6])
        x_0 = np.array([1e-6, 1e-6])
        H = np.eye(2)
        S_0 = np.eye(2) * 1e-12
        S_z = np.eye(2) * 1e-12

        estimator = ConcreteEstimator(z, x_0, H, S_0, S_z)

        result = estimator.forward(x_0)
        assert np.allclose(result, x_0)

    def test_estimator_with_large_values(self):
        """Test with large values."""
        z = np.array([1e6, 2e6])
        x_0 = np.array([1e6, 1e6])
        H = np.eye(2)
        S_0 = np.eye(2)
        S_z = np.eye(2)

        estimator = ConcreteEstimator(z, x_0, H, S_0, S_z)

        result = estimator.forward(x_0)
        assert np.allclose(result, x_0)


class TestEstimatorStatistics:
    """Tests for statistical properties of Estimator."""

    def test_leverage_exists(self):
        """Test that leverage method exists and is callable."""
        z = np.array([1.0, 2.0, 3.0])
        x_0 = np.array([1.0, 2.0])
        H = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        S_0 = np.eye(2)
        S_z = np.eye(3)

        estimator = ConcreteEstimator(z, x_0, H, S_0, S_z)

        # Leverage exists and is callable
        assert hasattr(estimator, "leverage")
        assert callable(estimator.leverage)

    def test_DOFS(self):
        """Test Degrees of Freedom for Signal (DOFS)."""
        z = np.array([1.0, 2.0, 3.0])
        x_0 = np.array([1.0, 2.0])
        H = np.array([[1.0, 0.5], [0.5, 1.0], [1.0, 1.0]])
        S_0 = np.eye(2)
        S_z = np.eye(3)

        estimator = ConcreteEstimator(z, x_0, H, S_0, S_z)

        dofs = estimator.DOFS
        assert isinstance(dofs, float)
        assert 0 <= dofs <= 2  # Should be between 0 and rank(H)

    def test_DOFS_identity_matrix(self):
        """Test DOFS with identity-like system."""
        z = np.array([1.0, 2.0])
        x_0 = np.array([1.0, 2.0])
        H = np.eye(2)
        S_0 = np.eye(2)
        S_z = np.eye(2)

        estimator = ConcreteEstimator(z, x_0, H, S_0, S_z)

        dofs = estimator.DOFS
        assert 0 < dofs <= 2

    def test_reduced_chi2(self):
        """Test reduced chi-squared statistic."""
        z = np.array([1.0, 2.0, 3.0])
        x_0 = np.array([1.0, 2.0])
        H = np.array([[1.0, 0.5], [0.5, 1.0], [1.0, 1.0]])
        S_0 = np.eye(2)
        S_z = np.eye(3)

        estimator = ConcreteEstimator(z, x_0, H, S_0, S_z)

        chi2 = estimator.reduced_chi2
        assert isinstance(chi2, float)
        assert chi2 >= 0

    def test_R2(self):
        """Test R-squared (coefficient of determination)."""
        z = np.array([1.0, 2.0, 3.0])
        x_0 = np.array([1.0, 2.0])
        H = np.array([[1.0, 0.5], [0.5, 1.0], [1.0, 1.0]])
        S_0 = np.eye(2)
        S_z = np.eye(3)

        estimator = ConcreteEstimator(z, x_0, H, S_0, S_z)

        r2 = estimator.R2
        assert isinstance(r2, float)
        assert 0 <= r2 <= 1

    def test_RMSE(self):
        """Test Root Mean Square Error."""
        z = np.array([1.0, 2.0, 3.0])
        x_0 = np.array([1.0, 2.0])
        H = np.array([[1.0, 0.5], [0.5, 1.0], [1.0, 1.0]])
        S_0 = np.eye(2)
        S_z = np.eye(3)

        estimator = ConcreteEstimator(z, x_0, H, S_0, S_z)

        rmse = estimator.RMSE
        assert isinstance(rmse, float)
        assert rmse >= 0

    def test_uncertainty_reduction(self):
        """Test uncertainty reduction metric."""
        z = np.array([1.0, 2.0, 3.0])
        x_0 = np.array([1.0, 2.0])
        H = np.array([[1.0, 0.5], [0.5, 1.0], [1.0, 1.0]])
        S_0 = np.eye(2)
        S_z = np.eye(3)

        estimator = ConcreteEstimator(z, x_0, H, S_0, S_z)

        u_red = estimator.uncertainty_reduction
        assert isinstance(u_red, float)
        # Should be between 0 (no improvement) and 1 (complete certainty)
        assert -1 <= u_red <= 1

    def test_statistics_consistency(self):
        """Test that statistics are consistent across multiple calls."""
        z = np.array([1.0, 2.0, 3.0])
        x_0 = np.array([1.0, 2.0])
        H = np.array([[1.0, 0.5], [0.5, 1.0], [1.0, 1.0]])
        S_0 = np.eye(2)
        S_z = np.eye(3)

        estimator = ConcreteEstimator(z, x_0, H, S_0, S_z)

        # Call multiple times
        rmse1 = estimator.RMSE
        rmse2 = estimator.RMSE

        assert rmse1 == rmse2


class TestLeverage:
    """Tests for Estimator.leverage() method."""

    def test_leverage_callable(self):
        """Leverage is a callable method."""
        z = np.array([1.0, 2.0, 3.0])
        x_0 = np.array([1.0, 2.0])
        H = np.array([[1.0, 0.5], [0.5, 1.0], [1.0, 1.0]])
        S_0 = np.eye(2)
        S_z = np.eye(3)
        estimator = ConcreteEstimator(z, x_0, H, S_0, S_z)
        assert callable(estimator.leverage)


class TestAvailableEstimators:
    """Tests for available_estimators() function."""

    def test_returns_list(self):
        """Test that available_estimators returns a list."""
        from fips.estimators import available_estimators

        result = available_estimators()
        assert isinstance(result, list)

    def test_contains_bayesian(self):
        """Test that 'bayesian' is in available estimators."""
        from fips.estimators import available_estimators

        assert "bayesian" in available_estimators()

    def test_exported_from_fips(self):
        """Test that available_estimators is exported from main package."""
        import fips

        assert hasattr(fips, "available_estimators")
        assert callable(fips.available_estimators)


class TestBayesianSolverRepr:
    """Tests for BayesianSolver.__repr__."""

    def test_repr_before_solve(self):
        """Test repr before solving shows solved=False."""
        from fips.estimators import BayesianSolver

        z = np.array([1.0, 2.0, 3.0])
        x_0 = np.array([1.0, 2.0])
        H = np.array([[1.0, 0.5], [0.5, 1.0], [1.0, 1.0]])
        S_0 = np.eye(2)
        S_z = np.eye(3)
        solver = BayesianSolver(z, x_0, H, S_0, S_z)
        r = repr(solver)
        assert "BayesianSolver" in r
        assert "n_x=2" in r
        assert "n_z=3" in r
        assert "solved=False" in r

    def test_repr_after_solve(self):
        """Test repr after solving shows solved=True."""
        from fips.estimators import BayesianSolver

        z = np.array([1.0, 2.0, 3.0])
        x_0 = np.array([1.0, 2.0])
        H = np.array([[1.0, 0.5], [0.5, 1.0], [1.0, 1.0]])
        S_0 = np.eye(2)
        S_z = np.eye(3)
        solver = BayesianSolver(z, x_0, H, S_0, S_z)
        _ = solver.x_hat  # trigger cached_property
        r = repr(solver)
        assert "solved=True" in r


class TestBayesianSolverRegularization:
    """Tests for BayesianSolver regularization factor."""

    @pytest.fixture
    def simple_problem(self):
        """Setup a simple inverse problem for testing."""
        np.random.seed(42)
        n_x = 3
        n_z = 4

        # True state
        x_true = np.array([1.0, 2.0, 3.0])

        # Forward operator
        H = np.random.randn(n_z, n_x)

        # Generate synthetic observations with noise
        y_true = H @ x_true
        obs_noise = np.random.randn(n_z) * 0.1
        z = y_true + obs_noise

        # Prior
        x_0 = np.zeros(n_x)
        S_0 = np.eye(n_x) * 2.0  # Prior uncertainty

        # Observation error covariance
        S_z = np.eye(n_z) * 0.01

        return {
            "z": z,
            "x_0": x_0,
            "H": H,
            "S_0": S_0,
            "S_z": S_z,
            "x_true": x_true,
        }

    def test_default_regularization_factor(self, simple_problem):
        """Test that default gamma=1.0 doesn't change S_z."""
        from fips.estimators import BayesianSolver

        solver = BayesianSolver(
            simple_problem["z"],
            simple_problem["x_0"],
            simple_problem["H"],
            simple_problem["S_0"],
            simple_problem["S_z"],
            gamma=1.0,
        )

        assert solver.gamma == 1.0
        # With gamma=1.0, S_z should equal the original (S_z / 1.0 = S_z)
        assert np.allclose(solver.S_z, simple_problem["S_z"])
        # _S_z_orig should store the unmodified original
        assert np.allclose(solver._S_z_orig, simple_problem["S_z"])

    def test_regularization_factor_scales_S_z(self, simple_problem):
        """Test that gamma scales S_z correctly (divides by gamma)."""
        from fips.estimators import BayesianSolver

        gamma = 2.5
        solver = BayesianSolver(
            simple_problem["z"],
            simple_problem["x_0"],
            simple_problem["H"],
            simple_problem["S_0"],
            simple_problem["S_z"],
            gamma=gamma,
        )

        # S_z should be scaled by 1/gamma (so S_z^{-1} is scaled by gamma)
        expected_S_z = simple_problem["S_z"] / gamma
        assert np.allclose(solver.S_z, expected_S_z)
        # Original should be unchanged
        assert np.allclose(solver._S_z_orig, simple_problem["S_z"])

    def test_high_regularization_factor_increases_uncertainty(self, simple_problem):
        """Test that gamma > 1 increases weight on data fitting (less regularization)."""
        from fips.estimators import BayesianSolver

        # Create solvers with different regularization factors
        solver_standard = BayesianSolver(
            simple_problem["z"],
            simple_problem["x_0"],
            simple_problem["H"],
            simple_problem["S_0"],
            simple_problem["S_z"],
            gamma=1.0,
        )

        solver_less_reg = BayesianSolver(
            simple_problem["z"],
            simple_problem["x_0"],
            simple_problem["H"],
            simple_problem["S_0"],
            simple_problem["S_z"],
            gamma=10.0,  # Less regularization (fits data more closely)
        )

        # Higher gamma increases weight on data, resulting in tighter posterior (smaller uncertainty)
        posterior_std_standard = np.sqrt(np.diag(solver_standard.S_hat))
        posterior_std_less_reg = np.sqrt(np.diag(solver_less_reg.S_hat))

        # With less regularization (higher gamma), posterior should be tighter
        assert np.all(posterior_std_less_reg <= posterior_std_standard + 1e-10)

    def test_low_regularization_factor_decreases_uncertainty(self, simple_problem):
        """Test that gamma < 1 decreases weight on data fitting (more regularization)."""
        from fips.estimators import BayesianSolver

        solver_standard = BayesianSolver(
            simple_problem["z"],
            simple_problem["x_0"],
            simple_problem["H"],
            simple_problem["S_0"],
            simple_problem["S_z"],
            gamma=1.0,
        )

        solver_more_reg = BayesianSolver(
            simple_problem["z"],
            simple_problem["x_0"],
            simple_problem["H"],
            simple_problem["S_0"],
            simple_problem["S_z"],
            gamma=0.1,  # More regularization (stays closer to prior)
        )

        # Lower gamma decreases weight on data, solution stays closer to prior
        # This results in larger posterior uncertainty
        posterior_std_standard = np.sqrt(np.diag(solver_standard.S_hat))
        posterior_std_more_reg = np.sqrt(np.diag(solver_more_reg.S_hat))

        # With more regularization (lower gamma), posterior should have more uncertainty
        assert np.all(posterior_std_more_reg >= posterior_std_standard - 1e-10)

    def test_regularization_affects_posterior_mean(self, simple_problem):
        """Test that regularization factor affects the posterior mean estimate."""
        from fips.estimators import BayesianSolver

        solver_less_reg = BayesianSolver(
            simple_problem["z"],
            simple_problem["x_0"],
            simple_problem["H"],
            simple_problem["S_0"],
            simple_problem["S_z"],
            gamma=100.0,  # Less regularization (fit data more)
        )

        solver_more_reg = BayesianSolver(
            simple_problem["z"],
            simple_problem["x_0"],
            simple_problem["H"],
            simple_problem["S_0"],
            simple_problem["S_z"],
            gamma=0.01,  # More regularization (stay closer to prior)
        )

        # With more regularization (low gamma), solution should be closer to prior
        distance_to_prior_less_reg = np.linalg.norm(
            solver_less_reg.x_hat - simple_problem["x_0"]
        )
        distance_to_prior_more_reg = np.linalg.norm(
            solver_more_reg.x_hat - simple_problem["x_0"]
        )

        assert distance_to_prior_more_reg < distance_to_prior_less_reg

    def test_regularization_in_cost_function(self, simple_problem):
        """Test that regularization factor is used in cost function."""
        from fips.estimators import BayesianSolver

        x_test = simple_problem["x_0"] + np.array([0.1, 0.2, 0.3])

        solver_rf1 = BayesianSolver(
            simple_problem["z"],
            simple_problem["x_0"],
            simple_problem["H"],
            simple_problem["S_0"],
            simple_problem["S_z"],
            gamma=1.0,
        )

        solver_rf2 = BayesianSolver(
            simple_problem["z"],
            simple_problem["x_0"],
            simple_problem["H"],
            simple_problem["S_0"],
            simple_problem["S_z"],
            gamma=2.0,
        )

        cost1 = solver_rf1.cost(x_test)
        cost2 = solver_rf2.cost(x_test)

        # Costs should be different due to different regularization
        assert not np.isclose(cost1, cost2)

    def test_kalman_gain_affected_by_regularization(self, simple_problem):
        """Test that Kalman gain matrix accounts for regularization."""
        from fips.estimators import BayesianSolver

        solver_rf1 = BayesianSolver(
            simple_problem["z"],
            simple_problem["x_0"],
            simple_problem["H"],
            simple_problem["S_0"],
            simple_problem["S_z"],
            gamma=1.0,
        )

        solver_rf5 = BayesianSolver(
            simple_problem["z"],
            simple_problem["x_0"],
            simple_problem["H"],
            simple_problem["S_0"],
            simple_problem["S_z"],
            gamma=5.0,
        )

        K1 = solver_rf1.K
        K5 = solver_rf5.K

        # Kalman gains should differ
        assert not np.allclose(K1, K5)

    def test_averaging_kernel_affected_by_regularization(self, simple_problem):
        """Test that averaging kernel matrix accounts for regularization."""
        from fips.estimators import BayesianSolver

        solver_rf1 = BayesianSolver(
            simple_problem["z"],
            simple_problem["x_0"],
            simple_problem["H"],
            simple_problem["S_0"],
            simple_problem["S_z"],
            gamma=1.0,
        )

        solver_rf5 = BayesianSolver(
            simple_problem["z"],
            simple_problem["x_0"],
            simple_problem["H"],
            simple_problem["S_0"],
            simple_problem["S_z"],
            gamma=5.0,
        )

        A1 = solver_rf1.A
        A5 = solver_rf5.A

        # Averaging kernels should differ
        assert not np.allclose(A1, A5)

    def test_dofs_affected_by_regularization(self, simple_problem):
        """Test that DOFS changes with regularization factor."""
        from fips.estimators import BayesianSolver

        solver_less_reg = BayesianSolver(
            simple_problem["z"],
            simple_problem["x_0"],
            simple_problem["H"],
            simple_problem["S_0"],
            simple_problem["S_z"],
            gamma=10.0,  # Less regularization (fit data more)
        )

        solver_more_reg = BayesianSolver(
            simple_problem["z"],
            simple_problem["x_0"],
            simple_problem["H"],
            simple_problem["S_0"],
            simple_problem["S_z"],
            gamma=0.1,  # More regularization (fit data less)
        )

        dofs_less_reg = solver_less_reg.DOFS
        dofs_more_reg = solver_more_reg.DOFS

        # DOFS should differ with different regularization
        assert not np.isclose(dofs_less_reg, dofs_more_reg)

        # With more regularization (lower gamma), DOFS should generally be lower
        # (less freedom to fit the data)
        assert dofs_more_reg < dofs_less_reg

    def test_uncertainty_reduction_affected_by_regularization(self, simple_problem):
        """Test that uncertainty reduction metrics work with regularization."""
        from fips.estimators import BayesianSolver

        solver = BayesianSolver(
            simple_problem["z"],
            simple_problem["x_0"],
            simple_problem["H"],
            simple_problem["S_0"],
            simple_problem["S_z"],
            gamma=2.0,
        )

        # Should be able to compute uncertainty reduction
        ur = solver.uncertainty_reduction
        U_red = solver.U_red

        assert isinstance(ur, float)
        assert 0.0 <= ur <= 1.0
        assert U_red.shape == (simple_problem["x_0"].shape[0],)

    def test_repr_includes_regularization_factor(self, simple_problem):
        """Test that __repr__ includes the regularization factor."""
        from fips.estimators import BayesianSolver

        solver = BayesianSolver(
            simple_problem["z"],
            simple_problem["x_0"],
            simple_problem["H"],
            simple_problem["S_0"],
            simple_problem["S_z"],
            gamma=3.5,
        )

        r = repr(solver)
        assert "gamma=3.5" in r

    def test_regularization_with_extreme_values(self, simple_problem):
        """Test that extreme regularization values don't break the solver."""
        from fips.estimators import BayesianSolver

        # Very small gamma (low weight on data fitting → more regularization → stay close to prior)
        solver_tiny = BayesianSolver(
            simple_problem["z"],
            simple_problem["x_0"],
            simple_problem["H"],
            simple_problem["S_0"],
            simple_problem["S_z"],
            gamma=1e-6,
        )

        # Very large gamma (high weight on data fitting → less regularization → fit data closely)
        solver_huge = BayesianSolver(
            simple_problem["z"],
            simple_problem["x_0"],
            simple_problem["H"],
            simple_problem["S_0"],
            simple_problem["S_z"],
            gamma=1e6,
        )

        # Should be able to compute solutions without errors
        x_hat_tiny = solver_tiny.x_hat
        x_hat_huge = solver_huge.x_hat

        assert x_hat_tiny.shape == simple_problem["x_0"].shape
        assert x_hat_huge.shape == simple_problem["x_0"].shape

        # Tiny gamma (more regularization) should keep solution very close to prior
        assert np.linalg.norm(x_hat_tiny - simple_problem["x_0"]) < 0.1

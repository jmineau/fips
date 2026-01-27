"""Test suite for fips.interfaces module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock

from fips.interfaces import EstimatorOutput, XR
from fips.matrices import CovarianceMatrix


class MockEstimator:
    """Mock Estimator for testing EstimatorOutput."""
    
    def __init__(self, n_x=5, n_z=10):
        self.n_x = n_x
        self.n_z = n_z
        self.x_hat = np.random.randn(n_x)
        self.S_hat = np.eye(n_x)
        self.y_hat = np.random.randn(n_z)
        self.y_0 = np.random.randn(n_z)
        self.U_red = np.ones(n_x) * 0.5


class MockInverseProblem(EstimatorOutput):
    """Mock InverseProblem for testing."""
    
    def __init__(self, n_x=5, n_z=10):
        self.n_x = n_x
        self.n_z = n_z
        self._estimator = MockEstimator(n_x, n_z)
        self._state_index = pd.Index([f'state_{i}' for i in range(n_x)], name='state')
        self._obs_index = pd.Index([f'obs_{i}' for i in range(n_z)], name='obs')
    
    @property
    def state_index(self):
        return self._state_index
    
    @property
    def obs_index(self):
        return self._obs_index
    
    @property
    def estimator(self):
        return self._estimator


class TestEstimatorOutputPosterior:
    """Tests for EstimatorOutput posterior properties."""

    def test_posterior_property(self):
        """Test posterior property returns correct Series."""
        problem = MockInverseProblem(n_x=5, n_z=10)
        posterior = problem.posterior
        
        assert isinstance(posterior, pd.Series)
        assert len(posterior) == 5
        assert posterior.name == "posterior"
        assert isinstance(posterior.index, pd.Index)

    def test_posterior_uses_estimator_x_hat(self):
        """Test that posterior uses estimator's x_hat."""
        problem = MockInverseProblem(n_x=3, n_z=5)
        estimator_x_hat = problem.estimator.x_hat.copy()
        
        posterior = problem.posterior
        
        assert np.allclose(posterior.values, estimator_x_hat)

    def test_posterior_index_matches_state_index(self):
        """Test that posterior index matches state index."""
        problem = MockInverseProblem(n_x=4, n_z=6)
        posterior = problem.posterior
        
        assert posterior.index.equals(problem.state_index)

    def test_posterior_cached_property(self):
        """Test that posterior is cached."""
        problem = MockInverseProblem()
        
        posterior1 = problem.posterior
        posterior2 = problem.posterior
        
        # Should be the same object (cached)
        assert posterior1 is posterior2


class TestEstimatorOutputPosteriorObs:
    """Tests for EstimatorOutput posterior_obs properties."""

    def test_posterior_obs_property(self):
        """Test posterior_obs property."""
        problem = MockInverseProblem(n_x=5, n_z=10)
        posterior_obs = problem.posterior_obs
        
        assert isinstance(posterior_obs, pd.Series)
        assert len(posterior_obs) == 10
        assert posterior_obs.name == "posterior_obs"

    def test_posterior_obs_uses_estimator_y_hat(self):
        """Test that posterior_obs uses estimator's y_hat."""
        problem = MockInverseProblem(n_x=5, n_z=8)
        estimator_y_hat = problem.estimator.y_hat.copy()
        
        posterior_obs = problem.posterior_obs
        
        assert np.allclose(posterior_obs.values, estimator_y_hat)

    def test_posterior_obs_index_matches_obs_index(self):
        """Test that posterior_obs index matches obs index."""
        problem = MockInverseProblem(n_x=3, n_z=7)
        posterior_obs = problem.posterior_obs
        
        assert posterior_obs.index.equals(problem.obs_index)


class TestEstimatorOutputPosteriorError:
    """Tests for EstimatorOutput posterior_error properties."""

    def test_posterior_error_property(self):
        """Test posterior_error property."""
        problem = MockInverseProblem(n_x=5, n_z=10)
        posterior_error = problem.posterior_error
        
        assert isinstance(posterior_error, CovarianceMatrix)
        assert posterior_error.shape == (5, 5)

    def test_posterior_error_uses_estimator_S_hat(self):
        """Test that posterior_error uses estimator's S_hat."""
        problem = MockInverseProblem(n_x=4, n_z=8)
        estimator_S_hat = problem.estimator.S_hat.copy()
        
        posterior_error = problem.posterior_error
        
        assert np.allclose(posterior_error.values, estimator_S_hat)

    def test_posterior_error_index_matches_state_index(self):
        """Test posterior error index matching."""
        problem = MockInverseProblem(n_x=3, n_z=6)
        posterior_error = problem.posterior_error
        
        assert posterior_error.index.equals(problem.state_index)
        assert posterior_error.columns.equals(problem.state_index)

    def test_posterior_error_is_symmetric(self):
        """Test that posterior error is symmetric."""
        problem = MockInverseProblem()
        posterior_error = problem.posterior_error
        
        assert np.allclose(posterior_error.values, posterior_error.values.T)


class TestEstimatorOutputPriorObs:
    """Tests for EstimatorOutput prior_obs properties."""

    def test_prior_obs_property(self):
        """Test prior_obs property."""
        problem = MockInverseProblem(n_x=5, n_z=10)
        prior_obs = problem.prior_obs
        
        assert isinstance(prior_obs, pd.Series)
        assert len(prior_obs) == 10
        assert prior_obs.name == "prior_obs"

    def test_prior_obs_uses_estimator_y_0(self):
        """Test that prior_obs uses estimator's y_0."""
        problem = MockInverseProblem(n_x=5, n_z=8)
        estimator_y_0 = problem.estimator.y_0.copy()
        
        prior_obs = problem.prior_obs
        
        assert np.allclose(prior_obs.values, estimator_y_0)

    def test_prior_obs_index_matches_obs_index(self):
        """Test that prior_obs index matches obs index."""
        problem = MockInverseProblem(n_x=4, n_z=9)
        prior_obs = problem.prior_obs
        
        assert prior_obs.index.equals(problem.obs_index)


class TestEstimatorOutputURed:
    """Tests for EstimatorOutput U_red properties."""

    def test_u_red_property(self):
        """Test U_red property."""
        problem = MockInverseProblem(n_x=5, n_z=10)
        u_red = problem.U_red
        
        assert isinstance(u_red, pd.Series)
        assert len(u_red) == 5
        assert u_red.name == "U_red"

    def test_u_red_uses_estimator_U_red(self):
        """Test that U_red uses estimator's U_red."""
        problem = MockInverseProblem(n_x=6, n_z=12)
        estimator_u_red = problem.estimator.U_red.copy()
        
        u_red = problem.U_red
        
        assert np.allclose(u_red.values, estimator_u_red)

    def test_u_red_index_matches_state_index(self):
        """Test U_red index matching."""
        problem = MockInverseProblem(n_x=7, n_z=11)
        u_red = problem.U_red
        
        assert u_red.index.equals(problem.state_index)


class TestXRInterface:
    """Tests for XR interface."""

    def test_xr_initialization(self):
        """Test XR interface initialization."""
        problem = MockInverseProblem()
        xr = XR(problem)
        
        assert xr._inversion is problem

    def test_xr_stores_inversion_reference(self):
        """Test that XR stores reference to inversion object."""
        problem = MockInverseProblem(n_x=5, n_z=10)
        xr = XR(problem)
        
        assert xr._inversion is problem
        assert xr._inversion.n_x == 5


class TestEstimatorOutputMultipleDimensions:
    """Tests for EstimatorOutput with various dimensions."""

    def test_large_dimensions(self):
        """Test with large state and observation dimensions."""
        problem = MockInverseProblem(n_x=100, n_z=200)
        
        posterior = problem.posterior
        posterior_error = problem.posterior_error
        posterior_obs = problem.posterior_obs
        
        assert len(posterior) == 100
        assert posterior_error.shape == (100, 100)
        assert len(posterior_obs) == 200

    def test_small_dimensions(self):
        """Test with small dimensions."""
        problem = MockInverseProblem(n_x=1, n_z=2)
        
        posterior = problem.posterior
        posterior_error = problem.posterior_error
        
        assert len(posterior) == 1
        assert posterior_error.shape == (1, 1)

    def test_equal_dimensions(self):
        """Test when state and observation dimensions are equal."""
        problem = MockInverseProblem(n_x=10, n_z=10)
        
        posterior = problem.posterior
        posterior_obs = problem.posterior_obs
        
        assert len(posterior) == 10
        assert len(posterior_obs) == 10


class TestEstimatorOutputConsistency:
    """Tests for consistency of EstimatorOutput properties."""

    def test_all_properties_consistent_shapes(self):
        """Test that all properties have consistent shapes."""
        problem = MockInverseProblem(n_x=7, n_z=13)
        
        posterior = problem.posterior
        posterior_error = problem.posterior_error
        posterior_obs = problem.posterior_obs
        prior_obs = problem.prior_obs
        u_red = problem.U_red
        
        assert len(posterior) == 7
        assert posterior_error.shape[0] == 7
        assert len(posterior_obs) == 13
        assert len(prior_obs) == 13
        assert len(u_red) == 7

    def test_all_properties_have_correct_dtype(self):
        """Test that all properties return correct types."""
        problem = MockInverseProblem()
        
        assert isinstance(problem.posterior, pd.Series)
        assert isinstance(problem.posterior_error, CovarianceMatrix)
        assert isinstance(problem.posterior_obs, pd.Series)
        assert isinstance(problem.prior_obs, pd.Series)
        assert isinstance(problem.U_red, pd.Series)

    def test_posterior_error_matches_estimator(self):
        """Test that posterior_error matches estimator S_hat."""
        problem = MockInverseProblem(n_x=5, n_z=10)
        
        posterior_error = problem.posterior_error
        estimator_S_hat = problem.estimator.S_hat
        
        assert np.allclose(posterior_error.values, estimator_S_hat)

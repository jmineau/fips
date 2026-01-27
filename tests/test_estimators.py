"""Test suite for fips.estimators module."""

import pytest
import numpy as np
import pandas as pd
from abc import ABC

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

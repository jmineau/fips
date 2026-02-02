"""Test suite for fips.interfaces module."""

import numpy as np
import pandas as pd
import pytest

from fips.covariance import CovarianceMatrix
from fips.interfaces import XR, EstimatorOutput


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
        super().__init__()
        import pandas as pd

        from fips.structures import Block, Vector

        self.n_x = n_x
        self.n_z = n_z
        self._estimator = MockEstimator(n_x, n_z)

        # Create Vector objects with proper block-level MultiIndex
        state_series = pd.Series(
            [1.0] * n_x,
            index=pd.Index([f"state_{i}" for i in range(n_x)], name="state_dim"),
            name="state_block",
        )
        obs_series = pd.Series(
            [1.0] * n_z,
            index=pd.Index([f"obs_{i}" for i in range(n_z)], name="obs_dim"),
            name="obs_block",
        )

        self._state_vector = Vector(name="state", blocks=[Block(state_series)])
        self._obs_vector = Vector(name="obs", blocks=[Block(obs_series)])

    @property
    def state_index(self):
        return self._state_vector.index

    @property
    def obs_index(self):
        return self._obs_vector.index

    @property
    def estimator(self):
        return self._estimator


class TestEstimatorOutputPosterior:
    """Tests for EstimatorOutput posterior properties."""

    def test_posterior_property(self):
        """Test posterior property returns correct Vector."""
        from fips.structures import Vector

        problem = MockInverseProblem(n_x=5, n_z=10)
        posterior = problem.posterior

        assert isinstance(posterior, Vector)
        assert posterior.n == 5
        assert posterior.name == "posterior"

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

        # Should have identical values (may be different objects due to wrapping)
        import numpy as np

        assert np.allclose(posterior1.values, posterior2.values)


class TestEstimatorOutputPosteriorObs:
    """Tests for EstimatorOutput posterior_obs properties."""

    def test_posterior_obs_property(self):
        """Test posterior_obs property."""
        from fips.structures import Vector

        problem = MockInverseProblem(n_x=5, n_z=10)
        posterior_obs = problem.posterior_obs

        assert isinstance(posterior_obs, Vector)
        assert posterior_obs.n == 10
        assert posterior_obs.name == "obs"

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
        from fips.structures import Vector

        problem = MockInverseProblem(n_x=5, n_z=10)
        prior_obs = problem.prior_obs

        assert isinstance(prior_obs, Vector)
        assert prior_obs.n == 10
        assert prior_obs.name == "obs"

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
        from fips.structures import Vector

        problem = MockInverseProblem(n_x=5, n_z=10)
        u_red = problem.U_red

        assert isinstance(u_red, Vector)
        assert u_red.n == 5
        assert u_red.name == "state"

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

        assert posterior.n == 100
        assert posterior_error.shape == (100, 100)
        assert posterior_obs.n == 200

    def test_small_dimensions(self):
        """Test with small dimensions."""
        problem = MockInverseProblem(n_x=1, n_z=2)

        posterior = problem.posterior
        posterior_error = problem.posterior_error

        assert posterior.n == 1
        assert posterior_error.shape == (1, 1)

    def test_equal_dimensions(self):
        """Test when state and observation dimensions are equal."""
        problem = MockInverseProblem(n_x=10, n_z=10)

        posterior = problem.posterior
        posterior_obs = problem.posterior_obs

        assert posterior.n == 10
        assert posterior_obs.n == 10


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

        assert posterior.n == 7
        assert posterior_error.shape[0] == 7
        assert posterior_obs.n == 13
        assert prior_obs.n == 13
        assert u_red.n == 7

    def test_all_properties_have_correct_dtype(self):
        """Test that all properties return correct types."""
        from fips.covariance import CovarianceMatrix
        from fips.structures import Vector

        problem = MockInverseProblem()

        assert isinstance(problem.posterior, Vector)
        assert isinstance(problem.posterior_error, CovarianceMatrix)
        assert isinstance(problem.posterior_obs, Vector)
        assert isinstance(problem.prior_obs, Vector)
        assert isinstance(problem.U_red, Vector)

    def test_posterior_error_matches_estimator(self):
        """Test that posterior_error matches estimator S_hat."""
        problem = MockInverseProblem(n_x=5, n_z=10)

        posterior_error = problem.posterior_error
        estimator_S_hat = problem.estimator.S_hat

        assert np.allclose(posterior_error.values, estimator_S_hat)


class TestXarrayConversions:
    """Tests for xarray conversion utilities."""

    def test_series_to_xarray(self):
        """Test converting Series to xarray DataArray."""
        import xarray as xr

        from fips.interfaces import series_to_xarray

        series = pd.Series([1, 2, 3], index=["a", "b", "c"], name="test")

        da = series_to_xarray(series)

        assert isinstance(da, xr.DataArray)
        assert da.shape == (3,)

    def test_series_to_xarray_with_name(self):
        """Test converting Series to xarray with custom name."""
        from fips.interfaces import series_to_xarray

        series = pd.Series([1, 2, 3], index=["a", "b", "c"])

        da = series_to_xarray(series, name="custom_name")

        assert da.name == "custom_name"

    def test_dataframe_to_xarray(self):
        """Test converting DataFrame to xarray DataArray."""
        import xarray as xr

        from fips.interfaces import dataframe_to_xarray

        df = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]], index=["a", "b", "c"], columns=["x", "y"]
        )

        da = dataframe_to_xarray(df)

        assert isinstance(da, xr.DataArray)

    def test_dataframe_to_xarray_with_multiindex_columns(self):
        """Test converting DataFrame with MultiIndex columns to xarray."""
        from fips.interfaces import dataframe_to_xarray

        arrays = [["x", "x", "y", "y"], [1, 2, 1, 2]]
        columns = pd.MultiIndex.from_arrays(arrays, names=["letter", "number"])
        df = pd.DataFrame(np.random.randn(3, 4), columns=columns)

        da = dataframe_to_xarray(df)

        assert da is not None

    def test_convert_to_xarray_series(self):
        """Test convert_to_xarray with Series."""
        import xarray as xr

        from fips.interfaces import convert_to_xarray

        series = pd.Series([1, 2, 3], index=["a", "b", "c"])

        result = convert_to_xarray(series, "test")

        assert isinstance(result, xr.DataArray)

    def test_convert_to_xarray_series_with_blocks(self):
        """Test convert_to_xarray with multi-block Series."""
        import xarray as xr

        from fips.interfaces import convert_to_xarray

        # Create a Series with block-level MultiIndex
        idx = pd.MultiIndex.from_product(
            [["b1", "b2"], ["x", "y"]], names=["block", "dim"]
        )
        series = pd.Series([1, 2, 3, 4], index=idx)

        result = convert_to_xarray(series, "test")

        # Should return Dataset when blocks are present
        assert isinstance(result, xr.Dataset)

    def test_convert_to_xarray_dataframe(self):
        """Test convert_to_xarray with DataFrame."""
        import xarray as xr

        from fips.interfaces import convert_to_xarray

        df = pd.DataFrame([[1, 2], [3, 4]], index=["a", "b"], columns=["x", "y"])

        result = convert_to_xarray(df, "test")

        assert isinstance(result, xr.DataArray)

    def test_convert_to_xarray_invalid_type(self):
        """Test that convert_to_xarray raises error for invalid input."""
        from fips.interfaces import convert_to_xarray

        with pytest.raises(TypeError, match="Cannot convert object"):
            convert_to_xarray([1, 2, 3], "test")


class TestPDandXRWrappers:
    """Tests for PD and XR wrapper classes."""

    def test_pd_wrapper_importable(self):
        """Test that PD wrapper class can be imported."""
        from fips.interfaces import PD

        assert PD is not None

    def test_xr_wrapper_importable(self):
        """Test that XR wrapper class can be imported."""
        from fips.interfaces import XR

        assert XR is not None

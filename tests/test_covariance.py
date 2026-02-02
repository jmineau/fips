"""Tests for covariance module."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fips.covariance import CovarianceMatrix, SpaceTimeCovariance, variances_as_series


class TestVariancesAsSeries:
    """Tests for variances_as_series helper function."""

    def test_series_input(self):
        """Test with pandas Series input."""
        s = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
        result = variances_as_series(s)
        assert isinstance(result, pd.Series)
        pd.testing.assert_series_equal(result, s)

    def test_series_with_index_warning(self):
        """Test that providing index with Series input gives warning."""
        s = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
        idx = pd.Index(["x", "y", "z"])
        with pytest.warns(UserWarning, match="index.*ignored"):
            result = variances_as_series(s, index=idx)
        # Should use Series' own index
        assert list(result.index) == ["a", "b", "c"]

    def test_xarray_input(self):
        """Test with xarray DataArray input."""
        da = xr.DataArray([1.0, 2.0, 3.0], coords={"x": ["a", "b", "c"]}, dims=["x"])
        result = variances_as_series(da)
        assert isinstance(result, pd.Series)
        assert len(result) == 3

    def test_scalar_with_index(self):
        """Test scalar value with index."""
        idx = pd.Index(["a", "b", "c", "d"])
        result = variances_as_series(5.0, index=idx)
        assert isinstance(result, pd.Series)
        assert len(result) == 4
        assert np.all(result.values == 5.0)

    def test_scalar_without_index_raises(self):
        """Test that scalar without index raises ValueError."""
        with pytest.raises(ValueError, match="index must be provided"):
            variances_as_series(5.0)

    def test_array_with_index(self):
        """Test array with index."""
        idx = pd.Index(["a", "b", "c"])
        arr = np.array([1.0, 2.0, 3.0])
        result = variances_as_series(arr, index=idx)
        assert isinstance(result, pd.Series)
        assert list(result.index) == ["a", "b", "c"]
        assert list(result.values) == [1.0, 2.0, 3.0]

    def test_array_without_index_raises(self):
        """Test that array without index raises ValueError."""
        arr = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="index must be provided"):
            variances_as_series(arr)

    def test_array_length_mismatch_raises(self):
        """Test that mismatched array and index lengths raise ValueError."""
        idx = pd.Index(["a", "b"])
        arr = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Length of variances must match"):
            variances_as_series(arr, index=idx)

    def test_list_with_index(self):
        """Test list with index."""
        idx = pd.Index(["a", "b", "c"])
        lst = [1.0, 2.0, 3.0]
        result = variances_as_series(lst, index=idx)
        assert isinstance(result, pd.Series)
        assert list(result.values) == [1.0, 2.0, 3.0]

    def test_invalid_type_raises(self):
        """Test that invalid types raise ValueError."""
        with pytest.raises(ValueError, match="must be a scalar, sequence"):
            variances_as_series({"a": 1, "b": 2})


class TestCovarianceMatrix:
    """Tests for CovarianceMatrix class."""

    def test_from_variances_with_series(self):
        """Test creating CovarianceMatrix from Series."""
        variances = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
        cov = CovarianceMatrix.from_variances(variances)
        assert cov.data.shape == (3, 3)
        assert np.allclose(np.diag(cov.data.values), [1.0, 2.0, 3.0])
        assert list(cov.data.index) == ["a", "b", "c"]

    def test_from_variances_with_array_and_index(self):
        """Test creating CovarianceMatrix from array with index."""
        variances = np.array([1.0, 2.0, 3.0])
        idx = pd.Index(["x", "y", "z"])
        cov = CovarianceMatrix.from_variances(variances, index=idx)
        assert cov.data.shape == (3, 3)
        assert np.allclose(np.diag(cov.data.values), [1.0, 2.0, 3.0])

    def test_from_variances_with_scalar(self):
        """Test creating CovarianceMatrix from scalar."""
        idx = pd.Index(["a", "b", "c", "d"])
        cov = CovarianceMatrix.from_variances(5.0, index=idx)
        assert cov.data.shape == (4, 4)
        assert np.allclose(np.diag(cov.data.values), 5.0)

    def test_diagonal_matrix(self):
        """Test that covariance matrix is diagonal."""
        variances = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
        cov = CovarianceMatrix.from_variances(variances)
        # Check that off-diagonal elements are zero
        assert np.allclose(cov.data.values - np.diag(np.diag(cov.data.values)), 0.0)

    def test_get_variances(self):
        """Test get_variances method."""
        variances = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
        cov = CovarianceMatrix.from_variances(variances)
        result = cov.get_variances()
        assert isinstance(result, pd.Series)
        assert np.allclose(result.values, [1.0, 2.0, 3.0])

    def test_get_variances_with_block(self):
        """Test get_variances with block parameter."""
        idx = pd.MultiIndex.from_arrays(
            [["a", "a", "b", "b"], [1, 2, 1, 2]], names=["block", "number"]
        )
        variances = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
        cov = CovarianceMatrix.from_variances(variances)
        result = cov.get_variances(block="a")
        assert len(result) == 2
        assert np.allclose(result.values, [1.0, 2.0])


class TestSpaceTimeCovariance:
    """Tests for SpaceTimeCovariance class."""

    def test_from_variances_diagonal_only(self):
        """Test creating SpaceTimeCovariance without correlations."""
        times = pd.date_range("2020-01-01", periods=3, freq="1h")
        lats = [40.0, 41.0]
        lons = [-111.0, -112.0]
        idx = pd.MultiIndex.from_product(
            [times, lats, lons], names=["time", "lat", "lon"]
        )
        variances = pd.Series(np.ones(len(idx)), index=idx)

        cov = SpaceTimeCovariance.from_variances(variances)
        assert cov.data.shape == (12, 12)  # 3 times × 2 lats × 2 lons
        # Should be diagonal
        assert np.allclose(cov.data.values - np.diag(np.diag(cov.data.values)), 0.0)

    def test_from_variances_with_spatial_corr(self):
        """Test creating SpaceTimeCovariance with spatial correlation."""
        times = pd.date_range("2020-01-01", periods=2, freq="1h")
        lats = [40.0, 41.0]
        lons = [-111.0, -112.0]
        idx = pd.MultiIndex.from_product(
            [times, lats, lons], names=["time", "lat", "lon"]
        )
        variances = pd.Series(np.ones(len(idx)), index=idx)

        spatial_corr = {"length_scale": 100.0}
        cov = SpaceTimeCovariance.from_variances(variances, spatial_corr=spatial_corr)
        assert cov.data.shape == (8, 8)
        # Should NOT be diagonal due to spatial correlation
        off_diag = cov.data.values - np.diag(np.diag(cov.data.values))
        assert not np.allclose(off_diag, 0.0)

    def test_from_variances_with_temporal_corr(self):
        """Test creating SpaceTimeCovariance with temporal correlation."""
        times = pd.date_range("2020-01-01", periods=3, freq="1h")
        lats = [40.0, 41.0]
        lons = [-111.0, -112.0]
        idx = pd.MultiIndex.from_product(
            [times, lats, lons], names=["time", "lat", "lon"]
        )
        variances = pd.Series(np.ones(len(idx)), index=idx)

        temporal_corr = {"length_scale": "1h"}
        cov = SpaceTimeCovariance.from_variances(variances, temporal_corr=temporal_corr)
        assert cov.data.shape == (12, 12)
        # Should NOT be diagonal due to temporal correlation
        off_diag = cov.data.values - np.diag(np.diag(cov.data.values))
        assert not np.allclose(off_diag, 0.0)

    def test_from_variances_with_both_correlations(self):
        """Test creating SpaceTimeCovariance with both spatial and temporal correlations."""
        times = pd.date_range("2020-01-01", periods=2, freq="1h")
        lats = [40.0, 41.0]
        lons = [-111.0, -112.0]
        idx = pd.MultiIndex.from_product(
            [times, lats, lons], names=["time", "lat", "lon"]
        )
        variances = pd.Series(np.ones(len(idx)), index=idx)

        spatial_corr = {"length_scale": 100.0}
        temporal_corr = {"length_scale": "1h"}
        cov = SpaceTimeCovariance.from_variances(
            variances, spatial_corr=spatial_corr, temporal_corr=temporal_corr
        )
        assert cov.data.shape == (8, 8)
        # Should be symmetric
        assert np.allclose(cov.data.values, cov.data.values.T)

    def test_from_variances_with_scalar_variances(self):
        """Test creating SpaceTimeCovariance from scalar variances."""
        times = pd.date_range("2020-01-01", periods=2, freq="1h")
        lats = [40.0, 41.0]
        lons = [-111.0, -112.0]
        idx = pd.MultiIndex.from_product(
            [times, lats, lons], names=["time", "lat", "lon"]
        )

        cov = SpaceTimeCovariance.from_variances(5.0, index=idx)
        assert cov.data.shape == (8, 8)
        assert np.allclose(np.diag(cov.data.values), 5.0)

    def test_custom_dimension_names(self):
        """Test creating SpaceTimeCovariance with custom dimension names."""
        times = pd.date_range("2020-01-01", periods=2, freq="1h")
        x_vals = [0.0, 1.0]
        y_vals = [0.0, 1.0]
        idx = pd.MultiIndex.from_product(
            [times, y_vals, x_vals], names=["datetime", "y", "x"]
        )
        variances = pd.Series(np.ones(len(idx)), index=idx)

        cov = SpaceTimeCovariance.from_variances(
            variances, time_dim="datetime", x_dim="x", y_dim="y"
        )
        assert cov.data.shape == (8, 8)

    def test_invalid_spatial_corr_type_raises(self):
        """Test that invalid spatial_corr type raises ValueError."""
        times = pd.date_range("2020-01-01", periods=2, freq="1h")
        lats = [40.0, 41.0]
        lons = [-111.0, -112.0]
        idx = pd.MultiIndex.from_product(
            [times, lats, lons], names=["time", "lat", "lon"]
        )
        variances = pd.Series(np.ones(len(idx)), index=idx)

        with pytest.raises(ValueError, match="spatial_corr must be"):
            SpaceTimeCovariance.from_variances(variances, spatial_corr="invalid")

    def test_invalid_temporal_corr_type_raises(self):
        """Test that invalid temporal_corr type raises ValueError."""
        times = pd.date_range("2020-01-01", periods=2, freq="1h")
        lats = [40.0, 41.0]
        lons = [-111.0, -112.0]
        idx = pd.MultiIndex.from_product(
            [times, lats, lons], names=["time", "lat", "lon"]
        )
        variances = pd.Series(np.ones(len(idx)), index=idx)

        with pytest.raises(ValueError, match="temporal_corr must be"):
            SpaceTimeCovariance.from_variances(variances, temporal_corr="invalid")

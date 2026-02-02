"""Tests for spacetime module."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from fips.utils.spacetime import (
    haversine_matrix,
    integrate_over_time_bins,
    spatial_decay_kernel,
    temporal_decay_kernel,
    time_decay_matrix,
    time_difference_matrix,
)


class TestHaversineMatrix:
    """Tests for haversine_matrix function."""

    def test_single_point(self):
        """Test distance matrix for a single point."""
        lats = [40.7]
        lons = [-111.9]
        result = haversine_matrix(lats, lons)
        assert result.shape == (1, 1)
        assert result[0, 0] == 0.0

    def test_two_points_known_distance(self):
        """Test distance between two known points."""
        # Salt Lake City and Provo (roughly 60 km apart)
        lats = [40.7608, 40.2338]
        lons = [-111.8910, -111.6585]
        result = haversine_matrix(lats, lons)
        assert result.shape == (2, 2)
        assert result[0, 0] == 0.0
        assert result[1, 1] == 0.0
        # Check symmetric
        assert np.isclose(result[0, 1], result[1, 0])
        # Distance should be roughly 60 km
        assert 50 < result[0, 1] < 70

    def test_diagonal_is_zero(self):
        """Test that diagonal elements are zero."""
        lats = [40.0, 41.0, 42.0]
        lons = [-111.0, -112.0, -113.0]
        result = haversine_matrix(lats, lons)
        assert np.allclose(np.diag(result), 0.0)

    def test_symmetric_matrix(self):
        """Test that result is symmetric."""
        lats = [40.0, 41.0, 42.0, 43.0]
        lons = [-111.0, -112.0, -113.0, -114.0]
        result = haversine_matrix(lats, lons)
        assert np.allclose(result, result.T)

    def test_radians_input(self):
        """Test with input in radians."""
        lats = [np.radians(40.0), np.radians(41.0)]
        lons = [np.radians(-111.0), np.radians(-112.0)]
        result = haversine_matrix(lats, lons, deg=False)
        assert result.shape == (2, 2)
        assert result[0, 0] == 0.0

    def test_custom_earth_radius(self):
        """Test with custom Earth radius."""
        lats = [0.0, 1.0]
        lons = [0.0, 0.0]
        result1 = haversine_matrix(lats, lons, earth_radius=6371.0)
        result2 = haversine_matrix(lats, lons, earth_radius=3000.0)
        # Distances should scale with radius (avoid dividing by zero on diagonal)
        ratio = result1[0, 1] / result2[0, 1]
        assert np.isclose(ratio, 6371.0 / 3000.0)

    def test_invalid_dimensions(self):
        """Test that 2D inputs raise ValueError."""
        lats = [[40.0, 41.0], [42.0, 43.0]]
        lons = [[-111.0, -112.0], [-113.0, -114.0]]
        with pytest.raises(ValueError, match="must be 1D"):
            haversine_matrix(lats, lons)

    def test_antipodal_points(self):
        """Test distance between antipodal points."""
        # Points on opposite sides of Earth should be ~20000 km apart
        lats = [0.0, 0.0]
        lons = [0.0, 180.0]
        result = haversine_matrix(lats, lons)
        # Half the Earth's circumference
        expected = np.pi * 6371.0
        assert np.isclose(result[0, 1], expected, rtol=0.01)


class TestSpatialDecayKernel:
    """Tests for spatial_decay_kernel function."""

    def test_basic_computation(self):
        """Test basic spatial decay kernel computation."""
        lats = [40.0, 41.0]
        lons = [-111.0, -112.0]
        length_scale = 100.0  # km
        result = spatial_decay_kernel(lats, lons, length_scale)
        # meshgrid creates 2x2 grid = 4 points
        assert result.shape == (4, 4)
        # Diagonal should be 1 (zero distance)
        assert np.allclose(np.diag(result), 1.0)
        # Off-diagonal should be < 1 (positive distance)
        assert 0 < result[0, 1] < 1

    def test_pandas_index_input(self):
        """Test with pandas Index inputs."""
        lats = pd.Index([40.0, 41.0, 42.0])
        lons = pd.Index([-111.0, -112.0, -113.0])
        length_scale = 100.0
        result = spatial_decay_kernel(lats, lons, length_scale)
        # meshgrid creates 3x3 grid = 9 points
        assert result.shape == (9, 9)


class TestIntegrateOverTimeBins:
    """Tests for integrate_over_time_bins function."""

    def test_series_integration(self):
        """Test integrating a Series over time bins."""
        times = pd.date_range("2020-01-01 00:00", periods=10, freq="1h")
        # Create a location dimension so we have a MultiIndex
        locations = ["A"] * 10
        idx = pd.MultiIndex.from_arrays([times, locations], names=["time", "location"])
        data = pd.Series(np.ones(10), index=idx, name="values")

        # Create 3-hour bins
        bins = pd.interval_range(
            start=pd.Timestamp("2020-01-01 00:00"),
            end=pd.Timestamp("2020-01-01 09:00"),
            freq="3h",
        )

        result = integrate_over_time_bins(data, bins, time_dim="time")
        assert isinstance(result, pd.Series)
        # Should have 3 bins
        assert len(result) == 3
        # Each bin should sum to approximately 3 (3 hours × 1)
        assert np.allclose(result.values, 3.0)

    def test_dataframe_integration(self):
        """Test integrating a DataFrame over time bins."""
        times = pd.date_range("2020-01-01 00:00", periods=10, freq="1h")
        # Create a location dimension so we have a MultiIndex
        locations = ["A"] * 10
        idx = pd.MultiIndex.from_arrays([times, locations], names=["time", "location"])
        data = pd.DataFrame({"col1": np.ones(10), "col2": np.ones(10) * 2}, index=idx)

        bins = pd.interval_range(
            start=pd.Timestamp("2020-01-01 00:00"),
            end=pd.Timestamp("2020-01-01 09:00"),
            freq="3h",
        )

        result = integrate_over_time_bins(data, bins, time_dim="time")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert np.allclose(result["col1"].values, 3.0)
        assert np.allclose(result["col2"].values, 6.0)

    def test_multiindex_integration(self):
        """Test integrating data with MultiIndex over time bins."""
        times = pd.date_range("2020-01-01 00:00", periods=6, freq="1h")
        locations = ["A", "B"] * 3
        idx = pd.MultiIndex.from_arrays([times, locations], names=["time", "location"])
        data = pd.Series(np.ones(6), index=idx, name="values")

        bins = pd.interval_range(
            start=pd.Timestamp("2020-01-01 00:00"),
            end=pd.Timestamp("2020-01-01 05:00"),
            freq="2h",
        )

        result = integrate_over_time_bins(data, bins, time_dim="time")
        assert isinstance(result, pd.Series)
        # Should have 3 time bins × 2 locations = 6 values, but some might be empty
        assert result.index.names == ["time", "location"]

    def test_missing_time_dim_raises(self):
        """Test that missing time dimension raises ValueError."""
        data = pd.Series([1, 2, 3], index=["a", "b", "c"])
        bins = pd.interval_range(start=0, end=3, freq=1)
        with pytest.raises(ValueError, match="not found in data index"):
            integrate_over_time_bins(data, bins, time_dim="time")


class TestTimeDifferenceMatrix:
    """Tests for time_difference_matrix function."""

    def test_basic_computation(self):
        """Test basic time difference computation."""
        times = [
            dt.datetime(2020, 1, 1, 0, 0),
            dt.datetime(2020, 1, 1, 1, 0),
            dt.datetime(2020, 1, 1, 2, 0),
        ]
        result = time_difference_matrix(times, absolute=True)
        assert result.shape == (3, 3)
        # Diagonal should be zero
        assert np.all(np.diag(result) == pd.Timedelta(0))

    def test_absolute_differences(self):
        """Test absolute time differences."""
        times = [
            dt.datetime(2020, 1, 1, 0, 0),
            dt.datetime(2020, 1, 1, 2, 0),
        ]
        result = time_difference_matrix(times, absolute=True)
        # Should be symmetric with absolute=True
        assert result[0, 1] == result[1, 0]
        assert result[0, 1] == pd.Timedelta(hours=2)

    def test_signed_differences(self):
        """Test signed time differences."""
        times = [
            dt.datetime(2020, 1, 1, 0, 0),
            dt.datetime(2020, 1, 1, 2, 0),
        ]
        result = time_difference_matrix(times, absolute=False)
        # Should be anti-symmetric with absolute=False
        assert result[0, 1] == -result[1, 0]
        assert result[0, 1] == -pd.Timedelta(hours=2)
        assert result[1, 0] == pd.Timedelta(hours=2)


class TestTimeDecayMatrix:
    """Tests for time_decay_matrix function."""

    def test_basic_decay(self):
        """Test basic time decay computation."""
        times = [
            dt.datetime(2020, 1, 1, 0, 0),
            dt.datetime(2020, 1, 1, 1, 0),
            dt.datetime(2020, 1, 1, 2, 0),
        ]
        decay = "1h"
        result = time_decay_matrix(times, decay)
        assert result.shape == (3, 3)
        # Diagonal should be 1 (exp(0) = 1)
        assert np.allclose(np.diag(result), 1.0)
        # Adjacent times should be exp(-1) ≈ 0.368
        assert np.isclose(result[0, 1], np.exp(-1), rtol=0.01)

    def test_timedelta_input(self):
        """Test with pd.Timedelta input."""
        times = [
            dt.datetime(2020, 1, 1, 0, 0),
            dt.datetime(2020, 1, 1, 6, 0),
        ]
        decay = pd.Timedelta(hours=3)
        result = time_decay_matrix(times, decay)
        # 6 hours / 3 hours decay = 2, so exp(-2) ≈ 0.135
        assert np.isclose(result[0, 1], np.exp(-2), rtol=0.01)


class TestTemporalDecayKernel:
    """Tests for temporal_decay_kernel function."""

    def test_exponential_method(self):
        """Test exponential decay method."""
        times = pd.date_range("2020-01-01", periods=5, freq="1h")
        result = temporal_decay_kernel(times, method="exp", length_scale="1h")
        assert result.shape == (5, 5)
        assert np.allclose(np.diag(result), 1.0)
        # Adjacent times should decay exponentially
        assert 0 < result[0, 1] < 1

    def test_diel_method(self):
        """Test diel (diurnal) decay method."""
        times = pd.date_range("2020-01-01 10:00", periods=24, freq="1h")
        result = temporal_decay_kernel(times, method="diel", length_scale="12h")
        # Same hour of day should have correlation, others should be zero
        # Time index 0 (10:00) and index 24 (10:00 next day) would match,
        # but we only have 24 hours, so check hour matching
        assert result.shape == (24, 24)

    def test_clim_method(self):
        """Test climatological (monthly) method."""
        times = pd.date_range("2020-01-01", periods=13, freq="MS")  # 13 months
        result = temporal_decay_kernel(times, method="clim")
        assert result.shape == (13, 13)
        # Same month in different years should be correlated
        # January 2020 (index 0) and January 2021 (index 12) should be 0.9
        assert np.isclose(result[0, 12], 0.9)

    def test_method_weights(self):
        """Test combining methods with weights."""
        times = pd.date_range("2020-01-01", periods=10, freq="1h")
        method = {"exp": 0.7, "diel": 0.3}
        result = temporal_decay_kernel(times, method=method, length_scale="2h")
        assert result.shape == (10, 10)

    def test_weights_must_sum_to_one(self):
        """Test that weights must sum to 1.0."""
        times = pd.date_range("2020-01-01", periods=5, freq="1h")
        method = {"exp": 0.5, "diel": 0.3}  # Sum = 0.8 != 1.0
        with pytest.raises(ValueError, match="must sum to 1.0"):
            temporal_decay_kernel(times, method=method, length_scale="1h")

    def test_invalid_method_raises(self):
        """Test that invalid method raises ValueError."""
        times = pd.date_range("2020-01-01", periods=5, freq="1h")
        with pytest.raises(ValueError, match="Unknown method"):
            temporal_decay_kernel(times, method="invalid", length_scale="1h")

    def test_default_method(self):
        """Test default method (exp with weight 1.0)."""
        times = pd.date_range("2020-01-01", periods=5, freq="1h")
        result = temporal_decay_kernel(times, method=None, length_scale="1h")
        assert result.shape == (5, 5)

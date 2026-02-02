"""Tests for correlation module."""

import datetime as dt

import numpy as np
import pandas as pd

from fips.utils.spacetime import (
    build_latlon_corr_matrix,
    build_temporal_corr_matrix,
    exponential_decay,
)


class TestExponentialDecay:
    """Tests for exponential_decay function."""

    def test_zero_distance(self):
        """Test decay at zero distance is 1."""
        result = exponential_decay(0, length_scale=100)
        assert result == 1.0

    def test_one_length_scale(self):
        """Test decay at one length scale is 1/e."""
        length_scale = 50.0
        result = exponential_decay(length_scale, length_scale)
        assert np.isclose(result, np.exp(-1))

    def test_array_input(self):
        """Test with array of distances."""
        distances = np.array([0, 50, 100, 150])
        length_scale = 50.0
        result = exponential_decay(distances, length_scale)
        expected = np.exp(-distances / length_scale)
        assert np.allclose(result, expected)

    def test_decay_properties(self):
        """Test that decay is between 0 and 1."""
        distances = np.linspace(0, 1000, 100)
        length_scale = 100.0
        result = exponential_decay(distances, length_scale)
        assert np.all(result >= 0)
        assert np.all(result <= 1)
        # Decay should be monotonically decreasing
        assert np.all(np.diff(result) <= 0)


class TestBuildLatlonCorrMatrix:
    """Tests for build_latlon_corr_matrix function."""

    def test_single_location(self):
        """Test correlation matrix for single location."""
        lats = [40.0]
        lons = [-111.0]
        length_scale = 100.0
        result = build_latlon_corr_matrix(lats, lons, length_scale)
        assert result.shape == (1, 1)
        assert result[0, 0] == 1.0

    def test_two_locations(self):
        """Test correlation matrix for two locations."""
        lats = [40.0, 41.0]
        lons = [-111.0, -112.0]
        length_scale = 100.0
        result = build_latlon_corr_matrix(lats, lons, length_scale)
        # meshgrid creates 2x2 grid = 4 points
        assert result.shape == (4, 4)
        # Diagonal should be 1
        assert np.allclose(np.diag(result), 1.0)
        # Off-diagonal should be between 0 and 1
        assert 0 < result[0, 1] < 1
        # Should be symmetric
        assert np.isclose(result[0, 1], result[1, 0])

    def test_grid_locations(self):
        """Test correlation matrix for grid of locations."""
        lats = [40.0, 41.0, 42.0]
        lons = [-111.0, -112.0]
        length_scale = 200.0
        result = build_latlon_corr_matrix(lats, lons, length_scale)
        # Should be 3x2 grid = 6 locations, but meshgrid creates 3 lats × 2 lons
        # Actually, meshgrid will create (3, 2) arrays that get raveled to 6 points
        assert result.shape == (6, 6)
        # Diagonal should be 1
        assert np.allclose(np.diag(result), 1.0)

    def test_custom_earth_radius(self):
        """Test with custom Earth radius."""
        lats = [0.0, 1.0]
        lons = [0.0, 0.0]
        length_scale = 100.0
        result1 = build_latlon_corr_matrix(
            lats, lons, length_scale, earth_radius=6371.0
        )
        result2 = build_latlon_corr_matrix(
            lats, lons, length_scale, earth_radius=3000.0
        )
        # Different Earth radius should give different correlations
        assert not np.allclose(result1, result2)

    def test_longer_length_scale_higher_correlation(self):
        """Test that longer length scale gives higher correlation."""
        lats = [40.0, 41.0]
        lons = [-111.0, -112.0]
        result_short = build_latlon_corr_matrix(lats, lons, length_scale=50.0)
        result_long = build_latlon_corr_matrix(lats, lons, length_scale=500.0)
        # Longer length scale should give higher off-diagonal correlations
        assert result_long[0, 1] > result_short[0, 1]

    def test_symmetric_matrix(self):
        """Test that result is symmetric."""
        lats = [40.0, 41.0, 42.0]
        lons = [-111.0, -112.0, -113.0]
        length_scale = 100.0
        result = build_latlon_corr_matrix(lats, lons, length_scale)
        assert np.allclose(result, result.T)

    def test_correlation_bounds(self):
        """Test that all correlations are between 0 and 1."""
        lats = np.linspace(35, 45, 5)
        lons = np.linspace(-115, -105, 5)
        length_scale = 100.0
        result = build_latlon_corr_matrix(lats, lons, length_scale)
        assert np.all(result >= 0)
        assert np.all(result <= 1)


class TestBuildTemporalCorrMatrix:
    """Tests for build_temporal_corr_matrix function."""

    def test_single_time(self):
        """Test correlation matrix for single time."""
        times = [dt.datetime(2020, 1, 1, 0, 0)]
        length_scale = "1h"
        result = build_temporal_corr_matrix(times, length_scale)
        assert result.shape == (1, 1)
        assert result[0, 0] == 1.0

    def test_two_times(self):
        """Test correlation matrix for two times."""
        times = [
            dt.datetime(2020, 1, 1, 0, 0),
            dt.datetime(2020, 1, 1, 1, 0),
        ]
        length_scale = "1h"
        result = build_temporal_corr_matrix(times, length_scale)
        assert result.shape == (2, 2)
        # Diagonal should be 1
        assert np.allclose(np.diag(result), 1.0)
        # 1 hour apart with 1 hour length scale should be exp(-1)
        assert np.isclose(result[0, 1], np.exp(-1))

    def test_multiple_times(self):
        """Test correlation matrix for multiple times."""
        times = [
            dt.datetime(2020, 1, 1, 0, 0),
            dt.datetime(2020, 1, 1, 2, 0),
            dt.datetime(2020, 1, 1, 4, 0),
        ]
        length_scale = "2h"
        result = build_temporal_corr_matrix(times, length_scale)
        assert result.shape == (3, 3)
        # Diagonal should be 1
        assert np.allclose(np.diag(result), 1.0)
        # Adjacent times (2h apart, 2h length scale) should be exp(-1)
        assert np.isclose(result[0, 1], np.exp(-1))
        assert np.isclose(result[1, 2], np.exp(-1))
        # Times 0 and 2 are 4h apart with 2h length scale, so exp(-2)
        assert np.isclose(result[0, 2], np.exp(-2))

    def test_symmetric_matrix(self):
        """Test that result is symmetric."""
        times = pd.date_range("2020-01-01", periods=5, freq="1h")
        length_scale = "2h"
        result = build_temporal_corr_matrix(times, length_scale)
        assert np.allclose(result, result.T)

    def test_correlation_bounds(self):
        """Test that all correlations are between 0 and 1."""
        times = pd.date_range("2020-01-01", periods=10, freq="1h")
        length_scale = "3h"
        result = build_temporal_corr_matrix(times, length_scale)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_longer_length_scale_higher_correlation(self):
        """Test that longer length scale gives higher correlation."""
        times = [
            dt.datetime(2020, 1, 1, 0, 0),
            dt.datetime(2020, 1, 1, 6, 0),
        ]
        result_short = build_temporal_corr_matrix(times, length_scale="1h")
        result_long = build_temporal_corr_matrix(times, length_scale="10h")
        # Longer length scale should give higher correlation
        assert result_long[0, 1] > result_short[0, 1]

    def test_timedelta_string_formats(self):
        """Test various Timedelta string formats."""
        times = [
            dt.datetime(2020, 1, 1, 0, 0),
            dt.datetime(2020, 1, 1, 1, 0),
        ]
        result1 = build_temporal_corr_matrix(times, length_scale="1h")
        result2 = build_temporal_corr_matrix(times, length_scale="60min")
        result3 = build_temporal_corr_matrix(times, length_scale="3600s")
        # All should give the same result
        assert np.allclose(result1, result2)
        assert np.allclose(result1, result3)

    def test_decay_with_days(self):
        """Test correlation with day-scale length scales."""
        times = [
            dt.datetime(2020, 1, 1),
            dt.datetime(2020, 1, 2),
            dt.datetime(2020, 1, 3),
        ]
        length_scale = "1D"
        result = build_temporal_corr_matrix(times, length_scale)
        # Adjacent days with 1-day length scale should be exp(-1)
        assert np.isclose(result[0, 1], np.exp(-1))
        # 2 days apart should be exp(-2)
        assert np.isclose(result[0, 2], np.exp(-2))

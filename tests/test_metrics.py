"""Tests for fips metrics module."""

import numpy as np
import pandas as pd
import pytest

from fips.metrics import haversine_matrix, time_diff_matrix


def test_haversine_matrix_shape():
    """Test that haversine_matrix returns correct shape."""
    lats = [40.0, 41.0, 42.0]
    lons = [-110.0, -111.0, -112.0]

    dist = haversine_matrix(lats, lons)
    assert dist.shape == (3, 3)
    assert np.allclose(np.diag(dist), 0.0)


def test_haversine_matrix_values():
    """Test haversine calculation against known roughly correct values."""
    # Distance between (0,0) and (0,1) deg at equator is ~111km
    lats = [0.0, 0.0]
    lons = [0.0, 1.0]

    dist = haversine_matrix(lats, lons)
    assert np.isclose(dist[0, 1], 111.19, atol=1.0)  # approx 111 km


def test_haversine_matrix_radians():
    """Test haversine with radians input."""
    # (0,0) to (0, 0.01745) radians is roughly 1 degree
    lats = np.radians([0.0, 0.0])
    lons = np.radians([0.0, 1.0])

    # Pass deg=False
    dist = haversine_matrix(lats, lons, deg=False)
    assert np.isclose(dist[0, 1], 111.19, atol=1.0)


def test_haversine_input_validation():
    """Test that it raises error for mismatched dimensions."""
    # It requires 1D arrays, so passing 2D should fail if checked, or different lengths?
    # The code checks dim=1
    with pytest.raises(ValueError):
        haversine_matrix(np.array([[1, 2]]), np.array([3, 4]))


def test_time_diff_matrix():
    """Test time_diff_matrix calculation."""
    times = pd.to_datetime(["2023-01-01 10:00", "2023-01-01 12:00"])

    # Calculate absolute difference
    diffs = time_diff_matrix(times, absolute=True)

    # Expected: 2 hours in nanoseconds
    # (2 * 3600 * 1e9)
    expected_ns = pd.Timedelta("2h")

    assert diffs.shape == (2, 2)
    assert diffs[0, 0] == 0
    assert diffs[1, 1] == 0
    assert diffs[0, 1] == expected_ns
    assert diffs[1, 0] == expected_ns


def test_time_diff_matrix_signed():
    """Test time_diff_matrix allows signed differences."""
    times = pd.to_datetime(["2023-01-01 10:00", "2023-01-01 12:00"])

    # Calculate signed difference
    diffs = time_diff_matrix(times, absolute=False)

    expected_ns = pd.Timedelta("2h")

    # t0 - t1 = 10:00 - 12:00 = -2h
    assert diffs[0, 1] == -expected_ns
    # t1 - t0 = 12:00 - 10:00 = +2h
    assert diffs[1, 0] == expected_ns

"""Tests for fips.covariance.distances: the distances between coordinates."""

import numpy as np
import pandas as pd
import pytest

from fips.covariance.distances import chordal_matrix, haversine_matrix, time_diff_matrix


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


def test_haversine_antipodal_points_are_half_the_circumference():
    """Exactly antipodal points are pi times the radius apart, never NaN (#108)."""
    rng = np.random.default_rng(0)
    lat = rng.uniform(-80, 80, 300)
    lon = rng.uniform(-180, 180, 300)
    lats = np.concatenate([lat, -lat])
    lons = np.concatenate([lon, np.where(lon > 0, lon - 180, lon + 180)])

    dist = haversine_matrix(lats, lons)
    assert not np.isnan(dist).any()
    np.testing.assert_allclose(np.diag(dist[:300, 300:]), np.pi * 6371.0, rtol=1e-6)


def test_haversine_input_validation():
    """Test that it raises error for mismatched dimensions."""
    # It requires 1D arrays, so passing 2D should fail if checked, or different lengths?
    # The code checks dim=1
    with pytest.raises(ValueError):
        haversine_matrix(np.array([[1, 2]]), np.array([3, 4]))


def test_time_diff_matrix():
    """Test time_diff_matrix gives the absolute differences."""
    times = pd.to_datetime(["2023-01-01 10:00", "2023-01-01 12:00"])

    diffs = time_diff_matrix(times)

    # Expected: 2 hours in nanoseconds
    # (2 * 3600 * 1e9)
    expected_ns = pd.Timedelta("2h")

    assert diffs.shape == (2, 2)
    assert diffs[0, 0] == 0
    assert diffs[1, 1] == 0
    assert diffs[0, 1] == expected_ns
    assert diffs[1, 0] == expected_ns


def test_chordal_matrix():
    """The chord through the Earth: shorter than the great circle, equal near zero."""
    lats, lons = [0.0, 0.0, 40.0, 40.001], [0.0, 90.0, -112.0, -112.0]
    chord, arc = chordal_matrix(lats, lons), haversine_matrix(lats, lons)
    assert chord.shape == (4, 4) and np.all(np.diag(chord) == 0.0)
    np.testing.assert_allclose(chord, chord.T)
    assert chord[0, 1] == pytest.approx(6371.0 * np.sqrt(2.0))
    assert chord[2, 3] == pytest.approx(arc[2, 3], rel=1e-9)
    assert (chord <= arc + 1e-9).all()
    with pytest.raises(ValueError, match="1D"):
        chordal_matrix([[0.0]], [[0.0]])

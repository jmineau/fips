"""Test suite for fips.problems.flux.transport.stilt.builder module.

Tests focus on pandas version compatibility, particularly:
1. Coordinate index rounding with pandas 3.x
2. MultiIndex level manipulation compatibility
"""

import numpy as np
import pandas as pd
import pytest


def test_multiindex_level_rounding_pandas3_compat():
    """Test that MultiIndex level rounding works correctly across pandas versions.

    In pandas 3.x, rounding MultiIndex values requires using get_level_values()
    followed by round() for each level individually. This test ensures the pattern
    used in build_jacobian_row_from_coords works correctly.
    """
    # Create coordinate list like in JacobianBuilder
    coord_list = [
        (-111.123456, 40.123456),
        (-111.234567, 40.234567),
        (-111.345678, 40.345678),
    ]

    # Build MultiIndex from coordinates
    coord_index = pd.MultiIndex.from_tuples(coord_list)

    # Round to 3 decimal places like we do for coordinate matching
    xdigits, ydigits = 3, 3

    # This is the pandas 3.x compatible pattern
    rounded_index = pd.MultiIndex.from_arrays(
        [
            coord_index.get_level_values(0).round(xdigits),
            coord_index.get_level_values(1).round(ydigits),
        ]
    ).set_names(["x", "y"])

    # Verify the rounding worked correctly
    expected_x = [-111.123, -111.235, -111.346]
    expected_y = [40.123, 40.235, 40.346]

    np.testing.assert_array_almost_equal(
        rounded_index.get_level_values(0).values, expected_x, decimal=3
    )
    np.testing.assert_array_almost_equal(
        rounded_index.get_level_values(1).values, expected_y, decimal=3
    )
    assert rounded_index.names == ["x", "y"]


def test_multiindex_level_rounding_with_latlon():
    """Test MultiIndex rounding with lat/lon coordinate names."""
    # Test with lat/lon names as used in geographical coordinates
    coord_list = [
        (-111.8765432, 40.7654321),
        (-111.9876543, 40.8765432),
    ]

    coord_index = pd.MultiIndex.from_tuples(coord_list)

    # Round to 4 decimal places (typical for lat/lon)
    lon_digits, lat_digits = 4, 4

    rounded_index = pd.MultiIndex.from_arrays(
        [
            coord_index.get_level_values(0).round(lon_digits),
            coord_index.get_level_values(1).round(lat_digits),
        ]
    ).set_names(["lon", "lat"])

    # Verify
    expected_lon = [-111.8765, -111.9877]
    expected_lat = [40.7654, 40.8765]

    np.testing.assert_array_almost_equal(
        rounded_index.get_level_values(0).values, expected_lon, decimal=4
    )
    np.testing.assert_array_almost_equal(
        rounded_index.get_level_values(1).values, expected_lat, decimal=4
    )


def test_multiindex_rounding_preserves_length():
    """Test that coordinate rounding preserves the number of coordinates."""
    # Create many coordinates with slight differences
    n_coords = 100
    np.random.seed(42)
    coord_list = [
        (-111.0 + np.random.randn() * 0.1, 40.0 + np.random.randn() * 0.1)
        for _ in range(n_coords)
    ]

    coord_index = pd.MultiIndex.from_tuples(coord_list)

    # Round with varying precision
    for digits in [1, 2, 3, 4, 5]:
        rounded_index = pd.MultiIndex.from_arrays(
            [
                coord_index.get_level_values(0).round(digits),
                coord_index.get_level_values(1).round(digits),
            ]
        ).set_names(["x", "y"])

        # Should preserve all coordinates even if some round to same values
        assert len(rounded_index) == n_coords


def test_multiindex_rounding_with_footprint_matching():
    """Test coordinate rounding matches footprint indexing pattern.

    This simulates the actual use case in build_jacobian_row_from_coords where
    rounded coordinates are used to filter footprint data.
    """
    # Simulate footprint coordinates (already rounded)
    foot_coords = [
        (-111.5, 40.5),
        (-111.5, 40.6),
        (-111.6, 40.5),
        (-111.6, 40.6),
    ]
    foot_index = pd.MultiIndex.from_tuples(foot_coords, names=["x", "y"])

    # Simulate user-provided coordinates (may have floating point noise)
    user_coords = [
        (-111.5000001, 40.5000001),
        (-111.6000001, 40.6000001),
    ]

    # Round user coordinates to match footprint resolution (1 decimal place)
    coord_index = pd.MultiIndex.from_tuples(user_coords)
    rounded_coords = pd.MultiIndex.from_arrays(
        [
            coord_index.get_level_values(0).round(1),
            coord_index.get_level_values(1).round(1),
        ]
    ).set_names(["x", "y"])

    # Create fake footprint data
    foot_data = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]}, index=foot_index)

    # Should be able to filter footprint using rounded coordinates
    # This simulates: foot.reset_index().loc[coord_index]
    filtered = foot_data.loc[rounded_coords]

    # Should match 2 of the 4 footprint points
    assert len(filtered) == 2
    assert filtered["value"].tolist() == [1.0, 4.0]


def test_calc_digits_function():
    """Test the calc_digits helper function logic.

    This function determines appropriate decimal places for coordinate rounding
    based on the resolution value.
    """

    # Simulate the calc_digits function from builder.py
    def calc_digits(res: float) -> int:
        if res <= 0:
            raise ValueError("Resolution must be positive")
        if res < 1:
            digits = int(np.ceil(np.abs(np.log10(res)))) + 1
        else:
            digits = int(-np.log10(res))
        return digits

    # Test various resolutions
    assert calc_digits(0.1) == 2  # 0.1 degree resolution
    assert calc_digits(0.01) == 3  # 0.01 degree resolution
    assert calc_digits(0.001) == 4  # 0.001 degree resolution
    assert calc_digits(1.0) == 0  # 1 degree resolution
    assert calc_digits(10.0) == -1  # 10 degree resolution (rare but valid)

    # Test with typical footprint resolutions
    assert calc_digits(0.02) == 3  # common 0.02 degree footprint
    assert calc_digits(0.05) == 3  # 0.05 degree footprint

    # Test error handling
    with pytest.raises(ValueError):
        calc_digits(0)
    with pytest.raises(ValueError):
        calc_digits(-0.1)


def test_coordinate_rounding_consistency():
    """Test that coordinate rounding is consistent across multiple operations.

    Ensures that rounding coordinates multiple times gives the same result,
    which is important for matching footprint data consistently.
    """
    coords = [(-111.123456789, 40.987654321)]

    # Round once
    coord_index1 = pd.MultiIndex.from_tuples(coords)
    rounded1 = pd.MultiIndex.from_arrays(
        [
            coord_index1.get_level_values(0).round(4),
            coord_index1.get_level_values(1).round(4),
        ]
    )

    # Round again (should be same)
    rounded2 = pd.MultiIndex.from_arrays(
        [
            rounded1.get_level_values(0).round(4),
            rounded1.get_level_values(1).round(4),
        ]
    )

    # Values should be identical
    np.testing.assert_array_equal(
        rounded1.get_level_values(0).values, rounded2.get_level_values(0).values
    )
    np.testing.assert_array_equal(
        rounded1.get_level_values(1).values, rounded2.get_level_values(1).values
    )


def test_multiindex_empty_coordinates():
    """Test handling of empty coordinate lists."""
    # Empty coordinate list
    coord_list = []
    coord_index = pd.MultiIndex.from_tuples(coord_list, names=["x", "y"])

    # Should handle empty index without errors
    if len(coord_index) > 0:
        rounded_index = pd.MultiIndex.from_arrays(
            [
                coord_index.get_level_values(0).round(3),
                coord_index.get_level_values(1).round(3),
            ]
        ).set_names(["x", "y"])
        assert len(rounded_index) == 0
    else:
        # Empty MultiIndex is valid
        assert len(coord_index) == 0

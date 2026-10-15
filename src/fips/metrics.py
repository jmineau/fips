"""
Distance and similarity metrics.

This module provides functions for calculating distances and similarities
between data points, such as Haversine distance for geographic coordinates
and time differences for temporal data, and the chordal distance through the
Earth that the covariance models use with Gaussian and Matérn kernels.
"""

import numpy as np
import pandas as pd


def haversine_matrix(lats, lons, earth_radius=None, deg=True):
    """
    Calculate the pairwise Haversine distance matrix between a set of coordinates.

    Parameters
    ----------
    lats : array-like
        1D array-like of latitude coordinates in degrees (if deg=True) else in radians.
    lons : array-like
        1D array-like of longitude coordinates in degrees (if deg=True) else in radians.
    earth_radius : float, optional
        Radius of the Earth in kilometers, by default 6371.0 km.
    deg : bool, optional
        If True, input coordinates are in degrees and will be converted to radians.
        If False, input coordinates are assumed to be in radians, by default True.

    Returns
    -------
    np.ndarray
        A 2D NumPy array (matrix) where the element at (i, j) is the
        Haversine distance between the i-th and j-th coordinate.
        The diagonal of the matrix will be zero.
    """
    if earth_radius is None:
        earth_radius = 6371.0  # Earth's radius in kilometers

    # Convert to numpy
    lats = np.asarray(lats)
    lons = np.asarray(lons)

    if not (lats.ndim == 1 and lons.ndim == 1):
        raise ValueError("lats and lons must be 1D sequences")

    if deg:
        # Convert degrees to radians
        lats = np.radians(lats)
        lons = np.radians(lons)

    # Reshape for broadcasting to column vectors
    lats = lats[:, np.newaxis]
    lons = lons[:, np.newaxis]

    # Calculate pairwise differences in latitude and longitude
    # Broadcasting (n, 1) with (1, n) results in an (n, n) matrix
    dlat = lats - lats.T
    dlon = lons - lons.T

    # Apply the Haversine formula
    a = np.sin(dlat / 2) ** 2 + np.cos(lats) * np.cos(lats.T) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance_matrix = earth_radius * c

    # The diagonal is already ~0 due to the calculation, but this ensures it's exactly 0.
    np.fill_diagonal(
        distance_matrix, 0
    )  # This line is often not needed but ensures precision.

    return distance_matrix


def chordal_matrix(lats, lons, earth_radius: float = 6371.0) -> np.ndarray:
    """
    Calculate the pairwise chordal distance matrix between points on the Earth.

    The chordal distance is the length of the straight line through the Earth
    between two points, ``2 R sin(c / 2)`` for a great-circle angle ``c``. A
    Gaussian or Matérn correlation of it is positive-definite on the sphere,
    which one of the great-circle distance is not (Gneiting 2013).

    Parameters
    ----------
    lats : array-like
        1D array-like of latitudes in degrees.
    lons : array-like
        1D array-like of longitudes in degrees.
    earth_radius : float, default 6371.0
        Radius of the Earth, in kilometers.

    Returns
    -------
    np.ndarray
        The matrix of distances in kilometers, zero on the diagonal.

    Raises
    ------
    ValueError
        If `lats` or `lons` is not 1D.

    Examples
    --------
    A quarter of the equator is 10,008 km along the surface and 9,010 km
    through the Earth:

    >>> from fips.metrics import chordal_matrix, haversine_matrix
    >>> haversine_matrix([0.0, 0.0], [0.0, 90.0]).round()
    array([[    0., 10008.],
           [10008.,     0.]])
    >>> chordal_matrix([0.0, 0.0], [0.0, 90.0]).round()
    array([[   0., 9010.],
           [9010.,    0.]])
    """
    lats = np.radians(np.asarray(lats, dtype=float))
    lons = np.radians(np.asarray(lons, dtype=float))
    if not (lats.ndim == 1 and lons.ndim == 1):
        raise ValueError("lats and lons must be 1D sequences")
    lats, lons = lats[:, np.newaxis], lons[:, np.newaxis]
    # sin(c / 2) is the square root of the haversine of the angle
    a = (
        np.sin((lats - lats.T) / 2) ** 2
        + np.cos(lats) * np.cos(lats.T) * np.sin((lons - lons.T) / 2) ** 2
    )
    distance = 2.0 * earth_radius * np.sqrt(np.clip(a, 0.0, 1.0))
    np.fill_diagonal(distance, 0.0)
    return distance


def time_diff_matrix(times, absolute: bool = True) -> np.ndarray:
    """
    Calculate the time differences between each pair of times.

    Parameters
    ----------
    times : list[dt.datetime]
        The list of times to calculate the differences between.
    absolute : bool, optional
        If True, return the absolute differences. Default is True.

    Returns
    -------
    np.ndarray
        The matrix of time differences.
    """
    # Normalize inputs (Series, DataFrame, ndarray) to a flat 1D sequence of datetimes.
    times = np.asarray(times).reshape(-1)
    # wrap in pandas DatetimeIndex as np.subtract.outer doesn't like pd.Series
    times = pd.DatetimeIndex(times)
    diffs = np.subtract.outer(times, times)
    if absolute:
        diffs = np.abs(diffs)
    return diffs

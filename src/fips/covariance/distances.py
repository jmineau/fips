"""
Distances between points: on the Earth, and in time.

The functions give the haversine (great-circle) distance and the chordal
distance through the Earth between points given by latitude and longitude in
degrees, in kilometers on a sphere of radius 6371 km, and the time difference
between datetimes. The covariance kernels measure with them; the chordal
distance goes with the Gaussian and Matérn kernels.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd

_EARTH_RADIUS = 6371.0
"""The Earth's radius, in km."""


def _half_angle_term(lats: npt.ArrayLike, lons: npt.ArrayLike) -> np.ndarray:
    """
    Return ``sin(c / 2) ** 2`` for the great-circle angle ``c`` between every two points.

    This is the haversine of the angle, which both distances are a function
    of. The coordinates are in degrees; the diagonal is exactly zero.
    """
    lats = np.radians(np.asarray(lats))
    lons = np.radians(np.asarray(lons))
    if not (lats.ndim == 1 and lons.ndim == 1):
        raise ValueError("lats and lons must be 1D sequences")
    # (n, 1) against (1, n) makes an (n, n) matrix
    lats, lons = lats[:, np.newaxis], lons[:, np.newaxis]
    a = (
        np.sin((lats - lats.T) / 2) ** 2
        + np.cos(lats) * np.cos(lats.T) * np.sin((lons - lons.T) / 2) ** 2
    )
    np.fill_diagonal(a, 0.0)
    return a


def haversine_matrix(lats: npt.ArrayLike, lons: npt.ArrayLike) -> np.ndarray:
    """
    Calculate the pairwise great-circle (haversine) distance matrix between points.

    The Earth is a sphere of radius 6371 km.

    Parameters
    ----------
    lats : array-like
        1D array-like of latitudes in degrees.
    lons : array-like
        1D array-like of longitudes in degrees.

    Returns
    -------
    np.ndarray
        The matrix of distances in kilometers, zero on the diagonal.

    Raises
    ------
    ValueError
        If `lats` or `lons` is not 1D.
    """
    a = _half_angle_term(lats, lons)
    return 2.0 * _EARTH_RADIUS * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def chordal_matrix(lats: npt.ArrayLike, lons: npt.ArrayLike) -> np.ndarray:
    """
    Calculate the pairwise chordal distance matrix between points on the Earth.

    The chordal distance is the length of the straight line through the Earth
    (a sphere of radius ``R`` = 6371 km) between two points, ``2 R sin(c / 2)``
    for a great-circle angle ``c``. A Gaussian or Matérn correlation of it is
    positive-definite on the sphere, which one of the great-circle distance is
    not (Gneiting 2013).

    Parameters
    ----------
    lats : array-like
        1D array-like of latitudes in degrees.
    lons : array-like
        1D array-like of longitudes in degrees.

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

    >>> from fips.covariance.distances import chordal_matrix, haversine_matrix
    >>> haversine_matrix([0.0, 0.0], [0.0, 90.0]).round()
    array([[    0., 10008.],
           [10008.,     0.]])
    >>> chordal_matrix([0.0, 0.0], [0.0, 90.0]).round()
    array([[   0., 9010.],
           [9010.,    0.]])
    """
    a = _half_angle_term(lats, lons)
    return 2.0 * _EARTH_RADIUS * np.sqrt(np.clip(a, 0.0, 1.0))


def time_diff_matrix(times: npt.ArrayLike) -> np.ndarray:
    """
    Calculate the absolute time difference between each pair of times.

    Parameters
    ----------
    times : array-like
        The datetimes to calculate the differences between (a list, array,
        Series or DatetimeIndex).

    Returns
    -------
    np.ndarray
        The matrix of time differences, as ``timedelta64``.
    """
    # Normalize inputs (Series, DataFrame, ndarray) to a flat 1D sequence of datetimes.
    times = np.asarray(times).reshape(-1)
    # wrap in pandas DatetimeIndex as np.subtract.outer doesn't like pd.Series
    times = pd.DatetimeIndex(times)
    return np.abs(np.subtract.outer(times, times))

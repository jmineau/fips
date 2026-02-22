import numpy as np
import pandas as pd


def haversine_matrix(lats, lons, earth_radius=None, deg=True):
    """
    Calculates the pairwise Haversine distance matrix between a set of coordinates.

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
    times = pd.DatetimeIndex(
        times
    )  # wrap in pandas DatetimeIndex as np.subtract.outer doesn't like pd.Series
    diffs = np.subtract.outer(times, times)
    if absolute:
        diffs = np.abs(diffs)
    return diffs

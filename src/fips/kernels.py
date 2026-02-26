"""
Covariance kernel functions.

This module provides kernel functions for generating covariance matrices,
such as exponential decay, constant correlation, and other spatial/temporal
correlation structures.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from fips.metrics import haversine_matrix, time_diff_matrix


def _exponential_decay(d, scale):
    return np.exp(-d / scale)


def _time_decay(times, scale, decay_func=_exponential_decay):
    diffs = time_diff_matrix(times)
    # diffs is already timedelta64[ns] from time_diff_matrix
    return decay_func(diffs, pd.Timedelta(scale))


def ConstantCorrelation():
    """Return a constant 1.0 correlation callable."""
    return lambda df: np.ones((len(df), len(df)))


def RaggedTimeDecay(
    time_dim: str, scale: str | pd.Timedelta, decay_func: Callable = _exponential_decay
):
    """
    Create a ragged temporal decay kernel.

    Defaults to exponential decay, but can accept any math function.

    Parameters
    ----------
    time_dim : str
        Name of the time dimension in the input DataFrame.
    scale : str or pd.Timedelta
        Scale parameter for the decay function. If a string is provided, it will be converted to a pd.Timedelta.
    decay_func : callable, optional
        A function that takes a distance matrix and a scale parameter and returns a decay matrix.
        Defaults to the exponential decay function.

    Returns
    -------
    function
        A kernel function that can be applied to a DataFrame to compute the temporal decay matrix based on the specified time dimension and scale.
    """

    def kernel(group_df: pd.DataFrame):
        times = group_df[time_dim]
        return _time_decay(times, scale, decay_func)

    return kernel


def GridTimeDecay(scale: str, decay_func: Callable = _exponential_decay):
    """
    Create a grid temporal decay kernel.

    Parameters
    ----------
    scale : str or pd.Timedelta
        Scale parameter for the decay function. If a string is provided, it will be converted to a pd.Timedelta.
    decay_func : callable, optional
        A function that takes a distance matrix and a scale parameter and returns a decay matrix.
        Defaults to the exponential decay function.

    Returns
    -------
    function
        A kernel function that can be applied to a DataFrame to compute the temporal decay matrix based on the specified scale, treating all time points as part of a single grid (i.e., not grouped by any time dimension).
    """
    return lambda times: _time_decay(times, scale, decay_func)


def GridSpatialDecay(
    lat_dim: str, lon_dim: str, scale: float, decay_func: Callable = _exponential_decay
):
    """
    Create a grid spatial decay kernel (Haversine).

    Parameters
    ----------
    lat_dim : str
        Name of the latitude dimension in the input DataFrame.
    lon_dim : str
        Name of the longitude dimension in the input DataFrame.
    scale : float
        Scale parameter for the decay function, in the same units as the distance matrix (e.g., kilometers if using Haversine distance).
    decay_func : callable, optional
        A function that takes a distance matrix and a scale parameter and returns a decay matrix.
        Defaults to the exponential decay function.

    Returns
    -------
    function
        A kernel function that can be applied to a DataFrame to compute the spatial decay matrix based on the specified latitude and longitude dimensions and scale, using Haversine distance for spatial separation.
    """

    def kernel(unique_space_coords: pd.DataFrame):
        lats = unique_space_coords[lat_dim].to_numpy()
        lons = unique_space_coords[lon_dim].to_numpy()
        distances = haversine_matrix(lats, lons)
        return decay_func(distances, scale)

    return kernel

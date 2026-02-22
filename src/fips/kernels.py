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
    """Factory that returns a constant 1.0 correlation callable."""
    return lambda df: np.ones((len(df), len(df)))


def RaggedTimeDecay(
    time_dim: str, scale: str | pd.Timedelta, decay_func: Callable = _exponential_decay
):
    """
    Factory for ragged temporal decay.
    Defaults to exponential decay, but can accept any math function.
    """

    def kernel(group_df: pd.DataFrame):
        times = group_df[time_dim]
        return _time_decay(times, scale, decay_func)

    return kernel


def GridTimeDecay(scale: str, decay_func: Callable = _exponential_decay):
    """Factory for grid temporal decay."""
    return lambda times: _time_decay(times, scale, decay_func)


def GridSpatialDecay(
    lat_dim: str, lon_dim: str, scale: float, decay_func: Callable = _exponential_decay
):
    """Factory for grid spatial decay (Haversine)."""

    def kernel(unique_space_coords: pd.DataFrame):
        lats = unique_space_coords[lat_dim].to_numpy()
        lons = unique_space_coords[lon_dim].to_numpy()
        distances = haversine_matrix(lats, lons)
        return decay_func(distances, scale)

    return kernel

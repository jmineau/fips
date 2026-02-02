import numpy as np
import numpy.typing as npt
import pandas as pd

from fips.spacetime import haversine_matrix, time_difference_matrix


def exponential_decay(distances, length_scale):
    """
    Compute exponential decay based on distances and length scale.

    Parameters
    ----------
    distances
        Array of distances.
    length_scale
        Length scale for the decay.

    Returns
    -------
    Exponential decay values [0, 1].
    """
    return np.exp(-distances / length_scale)


def build_latlon_corr_matrix(
    lats, lons, length_scale, earth_radius=None
) -> npt.NDArray:
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    distances = haversine_matrix(
        lats=lat_grid.ravel(), lons=lon_grid.ravel(), earth_radius=earth_radius
    )
    return exponential_decay(distances=distances, length_scale=length_scale)


def build_temporal_corr_matrix(times, length_scale) -> npt.NDArray:
    time_diffs = time_difference_matrix(times)
    return exponential_decay(
        distances=pd.DataFrame(time_diffs), length_scale=pd.Timedelta(length_scale)
    ).to_numpy()

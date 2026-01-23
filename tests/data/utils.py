"""Utility functions for test data generation."""

import numpy as np
import pandas as pd


def create_time_index(n_times, start="2020-01-01", freq="D"):
    """Create a pandas DatetimeIndex for time dimension.

    Parameters
    ----------
    n_times : int
        Number of time steps.
    start : str, optional
        Start date as string (default: "2020-01-01").
    freq : str, optional
        Pandas frequency string (default: "D" for daily).

    Returns
    -------
    pd.DatetimeIndex
        Time index.
    """
    return pd.date_range(start=start, periods=n_times, freq=freq)


def create_spatial_index(n_lat, n_lon, lat_range=(30, 45), lon_range=(-120, -105)):
    """Create latitude and longitude indices on a regular grid.

    Parameters
    ----------
    n_lat : int
        Number of latitude points.
    n_lon : int
        Number of longitude points.
    lat_range : tuple, optional
        (min_lat, max_lat) for latitude bounds.
    lon_range : tuple, optional
        (min_lon, max_lon) for longitude bounds.

    Returns
    -------
    tuple of (pd.Index, pd.Index)
        Latitude and longitude indices.
    """
    lats = pd.Index(np.linspace(*lat_range, n_lat), name="lat")
    lons = pd.Index(np.linspace(*lon_range, n_lon), name="lon")
    return lats, lons


def create_state_multiindex(
    n_times,
    n_lat,
    n_lon,
    time_start="2020-01-01",
    time_freq="D",
    lat_range=(30, 45),
    lon_range=(-120, -105),
):
    """Create state (flux) MultiIndex with time, lat, lon dimensions.

    Parameters
    ----------
    n_times, n_lat, n_lon : int
        Dimensions of the state grid.
    time_start : str, optional
        Start date for time index.
    time_freq : str, optional
        Frequency for time index.
    lat_range, lon_range : tuple, optional
        Ranges for spatial dimensions.

    Returns
    -------
    pd.MultiIndex
        MultiIndex with (time, lat, lon) levels.
    """
    times = create_time_index(n_times, start=time_start, freq=time_freq)
    lats, lons = create_spatial_index(
        n_lat, n_lon, lat_range=lat_range, lon_range=lon_range
    )
    return pd.MultiIndex.from_product([times, lats, lons], names=["time", "lat", "lon"])


def create_obs_multiindex(
    n_obs_locations,
    n_obs_times,
    time_start="2020-01-01",
    time_freq="D",
    lat_range=(30, 45),
    lon_range=(-120, -105),
):
    """Create observations MultiIndex with obs_location and obs_time dimensions.

    Parameters
    ----------
    n_obs_locations : int
        Number of observation locations (stations).
    n_obs_times : int
        Number of observation times.
    time_start : str, optional
        Start date for time index.
    time_freq : str, optional
        Frequency for time index.
    lat_range, lon_range : tuple, optional
        Ranges for observation location coordinates.

    Returns
    -------
    pd.MultiIndex
        MultiIndex with (obs_location, obs_time) levels.
    """
    obs_locs = pd.RangeIndex(n_obs_locations, name="obs_location")
    obs_times = create_time_index(n_obs_times, start=time_start, freq=time_freq)
    return pd.MultiIndex.from_product(
        [obs_locs, obs_times], names=["obs_location", "obs_time"]
    )


def create_correlated_noise(shape, correlation="none", seed=None):
    """Generate correlated noise with specified correlation structure.

    Parameters
    ----------
    shape : tuple
        Shape of output array.
    correlation : {"none", "spatial", "temporal"}, optional
        Type of correlation to apply (default: "none").
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Noise array with specified correlation structure.
    """
    if seed is not None:
        np.random.seed(seed)

    if correlation == "none":
        return np.random.randn(*shape)

    elif correlation == "spatial":
        # Create spatially correlated noise using smoothing
        noise = np.random.randn(*shape)
        if len(shape) >= 2:
            # Apply 1D smoothing along first spatial dimension
            kernel_size = max(3, shape[0] // 4)
            kernel = np.ones(kernel_size) / kernel_size
            noise = np.apply_along_axis(
                lambda x: np.convolve(x, kernel, mode="same"), 0, noise
            )
        return noise

    elif correlation == "temporal":
        # Create temporally correlated noise using AR(1) process
        noise = np.random.randn(*shape)
        if len(shape) >= 1:
            rho = 0.7  # Autocorrelation coefficient
            # Apply AR(1) along first axis (time)
            for i in range(1, shape[0]):
                noise[i] = rho * noise[i - 1] + np.sqrt(1 - rho**2) * np.random.randn(
                    *shape[1:]
                )
        return noise

    else:
        raise ValueError(f"Unknown correlation type: {correlation}")


def normalize_jacobian(jacobian, scale=1.0):
    """Normalize Jacobian matrix (forward operator) to have reasonable magnitude.

    Parameters
    ----------
    jacobian : np.ndarray
        Jacobian matrix of shape (n_obs, n_state).
    scale : float, optional
        Scale factor for normalization (default: 1.0).

    Returns
    -------
    np.ndarray
        Normalized Jacobian.
    """
    # Normalize columns to have unit norm, then scale
    column_norms = np.linalg.norm(jacobian, axis=0, keepdims=True)
    column_norms[column_norms == 0] = 1  # Avoid division by zero
    return scale * jacobian / column_norms

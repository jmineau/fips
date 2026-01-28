"""
Correlation kernels for covariance matrices.

This module provides reusable functions to compute spatial, temporal, and spatio-temporal
correlation kernels for use with CovarianceMatrix.set_block() and
CovarianceMatrix.set_interaction() methods across different inverse problem types.

Kernels accept flexible dimension names as parameters, allowing them to work with
any naming convention for time, location, latitude, longitude, etc.

Example:
    >>> from fips.kernels import exponential_decay_kernel, spatial_decay_kernel
    >>> # Extract times from a custom index dimension
    >>> times = obs_index.get_level_values("measurement_time")
    >>> temporal_kernel = exponential_decay_kernel(times, length_scale=5.0)
    >>> # Use custom dimension names for spatial indices
    >>> lats = state_index.get_level_values("latitude")
    >>> lons = state_index.get_level_values("longitude")
    >>> spatial_kernel = spatial_decay_kernel(lats, lons, length_scale=100.0)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def exponential_decay_kernel(
    times: np.ndarray | pd.Index,
    length_scale: float,
    same_day_only: bool = False,
) -> np.ndarray:
    """
    Exponential decay kernel for temporal correlations.

    Parameters
    ----------
    times : np.ndarray or pd.DatetimeIndex
        Array of datetime objects or timestamps.
    length_scale : float
        E-folding time scale (in days) for exponential decay.
    same_day_only : bool, optional
        If True, only compute correlations for same-day observations,
        by default False.

    Returns
    -------
    np.ndarray
        (n_times, n_times) correlation matrix with exponential decay.
    """
    if isinstance(times, pd.Index):
        times = times.to_numpy()

    # Convert to pandas Timestamp for reliable comparison
    times = pd.to_datetime(times)

    # Convert back to numpy for array operations
    if isinstance(times, pd.DatetimeIndex):
        times = times.to_numpy()

    # Compute pairwise time differences in days
    time_diffs = np.abs((times[:, None] - times[None, :]) / np.timedelta64(1, "D"))

    # Compute exponential decay
    correlation = np.exp(-time_diffs / length_scale)

    if same_day_only:
        # Zero out correlations across different calendar days
        times_dates = np.array([t.date() for t in times])
        same_day_mask = times_dates[:, None] == times_dates[None, :]
        correlation = correlation * same_day_mask

    return correlation


def spatial_decay_kernel(
    lats: np.ndarray | pd.Index,
    lons: np.ndarray | pd.Index,
    length_scale: float,
) -> np.ndarray:
    """
    Spatial decay kernel using Haversine distance and exponential correlation.

    Parameters
    ----------
    lats : np.ndarray or pd.Index
        Latitude coordinates (degrees).
    lons : np.ndarray or pd.Index
        Longitude coordinates (degrees).
    length_scale : float
        E-folding length scale (in km) for exponential decay.

    Returns
    -------
    np.ndarray
        (n_locations, n_locations) correlation matrix with spatial decay.
    """
    from fips.problems.flux.utils import haversine_matrix

    if isinstance(lats, pd.Index):
        lats = lats.to_numpy()
    if isinstance(lons, pd.Index):
        lons = lons.to_numpy()

    # Compute pairwise Haversine distances
    distance_matrix = haversine_matrix(lats, lons)

    # Exponential decay
    correlation = np.exp(-distance_matrix / (2 * length_scale**2))

    return correlation


def spatio_temporal_kernel(
    coords: pd.Index,
    spatial_length_scale: float,
    temporal_length_scale: float,
    same_day_only: bool = False,
    time_dim: str = "time",
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> np.ndarray:
    """
    Combined spatio-temporal correlation kernel using Kronecker product.

    Parameters
    ----------
    coords : pd.MultiIndex
        MultiIndex with time, lat, lon dimensions (names customizable).
    spatial_length_scale : float
        E-folding length scale (km) for spatial correlations.
    temporal_length_scale : float
        E-folding length scale (days) for temporal correlations.
    same_day_only : bool, optional
        If True, only correlate observations from the same day, by default False.
    time_dim : str, optional
        Name of the time dimension in MultiIndex, by default "time".
    lat_dim : str, optional
        Name of the latitude dimension in MultiIndex, by default "lat".
    lon_dim : str, optional
        Name of the longitude dimension in MultiIndex, by default "lon".

    Returns
    -------
    np.ndarray
        (n_coords, n_coords) spatio-temporal correlation matrix.
    """
    if not isinstance(coords, pd.MultiIndex):
        raise TypeError("coords must be a pd.MultiIndex with time, lat, lon dimensions")

    # Extract level values
    times = coords.get_level_values(time_dim).to_numpy()
    lats = coords.get_level_values(lat_dim).to_numpy()
    lons = coords.get_level_values(lon_dim).to_numpy()

    # Get unique values for each dimension
    unique_times = pd.to_datetime(np.unique(times))
    unique_lats = np.unique(lats)
    unique_lons = np.unique(lons)

    # Build individual kernels
    temporal_corr = exponential_decay_kernel(
        unique_times, temporal_length_scale, same_day_only=same_day_only
    )
    spatial_corr = spatial_decay_kernel(unique_lats, unique_lons, spatial_length_scale)

    # Combine via Kronecker product: temporal ⊗ spatial
    # This assumes data is ordered with time as fastest-varying
    combined_corr = np.kron(temporal_corr, spatial_corr)

    return combined_corr


def same_day_groupwise_kernel(
    obs_index: pd.MultiIndex,
    time_dim: str = "obs_time",
) -> np.ndarray:
    """
    Kernel that gives full correlation for same-day observations, zero otherwise.

    Used for same-day error correlations (e.g., aggregation error).

    Parameters
    ----------
    obs_index : pd.MultiIndex
        Observation index with time dimension.
    time_dim : str, optional
        Name of the time dimension in obs_index, by default "obs_time".

    Returns
    -------
    np.ndarray
        (n_obs, n_obs) correlation matrix with 1.0 for same-day pairs, 0.0 otherwise.
    """
    times = obs_index.get_level_values(time_dim).to_numpy()
    times_dates = np.array([pd.Timestamp(t).date() for t in times])

    # Same day = 1, different day = 0
    correlation = (times_dates[:, None] == times_dates[None, :]).astype(float)

    return correlation


def location_specific_kernel(
    obs_index: pd.MultiIndex,
    location_map: dict,
    location_dim: str = "obs_location",
) -> np.ndarray:
    """
    Kernel that applies location-specific variances (diagonal only).

    Used for instrument errors that vary by location.

    Parameters
    ----------
    obs_index : pd.MultiIndex
        Observation index with location dimension.
    location_map : dict
        Mapping from location name to variance value.
    location_dim : str, optional
        Name of the location dimension in obs_index, by default "obs_location".

    Returns
    -------
    np.ndarray
        (n_obs, n_obs) diagonal matrix with location-specific variances.
    """
    locations = obs_index.get_level_values(location_dim).to_numpy()

    # Map locations to variances
    variances = np.array([location_map.get(loc, 0.0) for loc in locations])

    # Return diagonal matrix
    return np.diag(variances)


def exponential_with_masking_kernel(
    times: np.ndarray | pd.Index,
    locations: np.ndarray | pd.Index | None,
    temporal_length_scale: float,
    interday: bool = True,
) -> np.ndarray:
    """
    Exponential decay kernel with optional interday masking.

    Used for transport and background errors with time-scale dependence.

    Parameters
    ----------
    times : np.ndarray or pd.Index
        Array of datetime objects.
    locations : np.ndarray, pd.Index, or None
        Array of location identifiers (not currently used, for future extensions).
    temporal_length_scale : float
        E-folding time scale (days).
    interday : bool, optional
        If False, zero out correlations across different calendar days,
        by default True.

    Returns
    -------
    np.ndarray
        (n_obs, n_obs) correlation matrix.
    """
    if isinstance(times, pd.Index):
        times = times.to_numpy()

    times = pd.to_datetime(times)

    # Compute time differences in days
    time_diffs = np.abs((times[:, None] - times[None, :]).total_seconds() / 86400.0)

    # Exponential decay
    correlation = np.exp(-time_diffs / temporal_length_scale)

    if not interday:
        # Zero out correlations across different days
        times_dates = np.array([t.date() for t in times])
        same_day_mask = times_dates[:, None] == times_dates[None, :]
        correlation = correlation * same_day_mask

    return correlation

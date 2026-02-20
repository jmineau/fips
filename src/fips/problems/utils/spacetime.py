"""Spatial and temporal utilities for atmospheric inverse problems.

This module provides utilities for:
- Computing spatial distances (Haversine)
- Integrating data over time bins
- Computing time differences
"""

import math

import numpy as np
import numpy.typing as npt
import pandas as pd
from typing_extensions import Self

from fips.base import Structure1D
from fips.covariance import CovarianceMatrix


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


# ==============================================================================
# SPATIAL UTILITIES
# ==============================================================================


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


def spatial_decay_kernel(
    lats: np.ndarray | pd.Index,
    lons: np.ndarray | pd.Index,
    length_scale: float,
) -> np.ndarray:
    """
    Compute a spatial correlation kernel based on Haversine distances with exponential decay.

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
    # Get the lat/lon coordinates of the grid cells
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Calculate pairwise haversine distances between grid cells
    distances = haversine_matrix(lats=lat_grid.ravel(), lons=lon_grid.ravel())
    corr = np.exp((-1 * distances) / length_scale)

    return corr


def build_latlon_corr_matrix(
    lats, lons, length_scale, earth_radius=None
) -> npt.NDArray:
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    distances = haversine_matrix(
        lats=lat_grid.ravel(), lons=lon_grid.ravel(), earth_radius=earth_radius
    )
    return exponential_decay(distances=distances, length_scale=length_scale)


# ==============================================================================
# TEMPORAL UTILITIES
# ==============================================================================


def integrate_over_time_bins(
    data: pd.DataFrame | pd.Series, time_bins: pd.IntervalIndex, time_dim: str = "time"
) -> pd.DataFrame | pd.Series:
    """
    Integrate data over time bins.

    Parameters
    ----------
    data : pd.DataFrame | pd.Series
        Data to integrate.
    time_bins : pd.IntervalIndex
        Time bins for integration.
    time_dim : str, optional
        Time dimension name, by default 'time'

    Returns
    -------
    pd.DataFrame | pd.Series
        Integrated footprint. The bin labels are set to the left edge of the bin.
    """
    is_series = isinstance(data, pd.Series)

    dims = data.index.names
    if time_dim not in dims:
        raise ValueError(f"time_dim '{time_dim}' not found in data index levels {dims}")
    other_levels = [lvl for lvl in dims if lvl != time_dim]

    data = data.reset_index()

    # Use pd.cut to bin the data by time into time bins
    data[time_dim] = pd.cut(
        data[time_dim], bins=time_bins, include_lowest=True, right=False
    )

    # Set Intervals to the left edge of the bin (start of time interval)
    data[time_dim] = data[time_dim].apply(lambda x: x.left)

    # Group the date by the time bins & any other existing levels
    grouped = data.groupby([time_dim] + other_levels, observed=True)

    # Sum over the groups
    integrated = grouped.sum()

    # Order the index levels
    integrated = integrated.reorder_levels(dims)

    if is_series:
        # Return a Series if the input was a Series
        return integrated.iloc[:, 0]
    return integrated


def time_difference_matrix(times, absolute: bool = True) -> np.ndarray:
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


def time_decay_matrix(times, decay: str | pd.Timedelta) -> np.ndarray:
    """
    Calculate the time decay matrix for the specified times and decay.

    Parameters
    ----------
    times : list[dt.datetime]
        The list of times to calculate the decay matrix for.
    decay : str | pd.Timedelta
        The decay to use for the exponential decay.

    Returns
    -------
    np.ndarray
        The matrix of time decay values.
    """
    # Calculate the time differences
    diffs = time_difference_matrix(times, absolute=True)

    # Wrap in pandas DataFrame to use pd.Timedelta functionality
    diffs = pd.DataFrame(diffs)

    # Get decay as a pd.Timedelta
    decay = pd.Timedelta(decay)

    # Calculate the decay matrix using an exponential decay
    decay_matrix = np.exp(-diffs / decay).to_numpy()
    return decay_matrix


def temporal_decay_kernel(
    times: pd.Index | pd.DatetimeIndex,
    method: str | dict | None = None,
    length_scale: pd.Timedelta | str | None = None,
) -> np.ndarray:
    """
    Build the temporal error correlation matrix for the inversion.
    Has dimensions of flux_times x flux_times.

    Parameters
    ----------
    times : pd.Index or pd.DatetimeIndex
        Datetime-like index of flux times. Values will be coerced to a
        pandas.DatetimeIndex internally so attributes like `.hour` and
        `.month` are available.
    method : dict, optional
        Method for calculating the temporal error correlation matrix.
        The key defines the method and the value is the weight for the method.
        Options include:
            - 'exp': exponentially decaying with an e-folding length of self.t_bins freq
            - 'diel': like 'exp', except correlations are only non-zero for the same time of day
            - 'clim': each month is highly correlated with the same month in other years

    Returns
    -------
    np.ndarray
        Temporal error correlation matrix for the inversion.
    """
    # Coerce to DatetimeIndex so we can use .hour/.month safely
    times = pd.DatetimeIndex(times)

    # Handle method input
    if method is None:
        method = {"exp": 1.0}
    elif isinstance(method, str):
        method = {method: 1.0}
    elif isinstance(method, dict):
        if not math.isclose(float(sum(method.values())), 1.0):
            raise ValueError("Weights for temporal error methods must sum to 1.0")
    else:
        raise ValueError("method must be a dict")

    # Initialize the temporal correlation matrix
    N = len(times)  # number of flux times
    corr = np.zeros((N, N))

    # Calculate and combine correlation matrices based on the method weights
    for method_name, weight in method.items():
        if method_name in ["exp", "diel"]:  # Closer times are more correlated
            if not isinstance(length_scale, (str, pd.Timedelta)):
                raise ValueError(
                    'length_scale must be a str or pd.Timedelta for "exp" and "diel" methods'
                )
            method_corr = time_decay_matrix(times, decay=length_scale)

            if method_name == "diel":
                # Set the correlation values for the same hour of day
                # use a NumPy array for safe ndarray-style indexing
                hours = times.hour.values
                same_time_mask = (hours[:, None] - hours[None, :]) == 0
                method_corr[~same_time_mask] = 0

        elif (
            method_name == "clim"
        ):  # Each month is highly correlated with the same month in other years
            # Initialize the correlation matrix as identity matrix
            method_corr = np.eye(N)  # the diagonal

            # Set the correlation values for the same month in other years
            corr_val = 0.9
            months = times.month.values

            # Create a mask for the same month in different years
            same_month_mask = (months[:, None] - months[None, :]) % 12 == 0

            # Apply the correlation value using the mask
            method_corr[same_month_mask] = corr_val
        else:
            raise ValueError(f"Unknown method: {method_name}")

        # Combine the correlation matrices based on the method weights
        corr += weight * method_corr

    return corr


def build_temporal_corr_matrix(times, length_scale) -> npt.NDArray:
    time_diffs = time_difference_matrix(times)
    return exponential_decay(
        distances=pd.DataFrame(time_diffs), length_scale=pd.Timedelta(length_scale)
    ).to_numpy()


class SpaceTimeCovariance(CovarianceMatrix):
    """
    Represents a spatiotemporal Covariance Matrix.
    Combines spatial and temporal correlations with variances to build a full covariance matrix.
    """

    @classmethod
    def from_variances(
        cls,
        variances: pd.Series | np.ndarray | float,
        index: pd.Index | None = None,
        spatial_corr: dict | None = None,
        temporal_corr: dict | None = None,
        time_dim: str = "time",
        x_dim: str = "lon",
        y_dim: str = "lat",
        **kwargs,
    ) -> Self:
        """
        Create a SpaceTimeCovariance instance from variances and correlation matrices.

        Parameters
        ----------
        variances : float | np.ndarray
            Prior error variances for the inversion.
        index : pd.MultiIndex
            MultiIndex for flux state.
        spatial_corr : dict, optional
            Spatial correlation parameters for the inversion, by default None.
            If None, the spatial error correlation matrix is an identity matrix.
            See `build_spatial_corr` for more options.
        temporal_corr : dict, optional
            Temporal correlation parameters for the inversion, by default None.
            If None, the temporal error correlation matrix is an identity matrix.
            See `build_temporal_corr` for more options.
        time_dim : str
            Name of the time dimension in the index, by default "time".
        x_dim : str
            Name of the x (longitude) dimension in the index, by default "lon".
        y_dim : str
            Name of the y (latitude) dimension in the index, by default "lat".

        Returns
        -------
        SpaceTimeCovariance
            A new SpaceTimeCovariance instance
        """
        variances = Structure1D(variances, index=index).to_series()

        index = variances.index
        variances = variances.to_numpy()

        if spatial_corr is None and temporal_corr is None:
            # If no correlations specified, return diagonal covariance matrix
            return cls(variances, index=index)

        # Get unique times, lats, and lons
        times = index.get_level_values(time_dim).unique()
        lats = index.get_level_values(y_dim).unique().to_numpy()
        lons = index.get_level_values(x_dim).unique().to_numpy()

        # Build spatial correlation matrix
        if spatial_corr is None:
            spatial_corr_matrix = np.eye(len(lats) * len(lons))
        elif isinstance(spatial_corr, dict):
            spatial_corr_matrix = build_latlon_corr_matrix(
                lats=lats, lons=lons, **spatial_corr
            )
        else:
            raise ValueError("spatial_corr must be a dict or None")

        # Build temporal correlation matrix
        if temporal_corr is None:
            temporal_corr_matrix = np.eye(len(times))
        elif isinstance(temporal_corr, dict):
            temporal_corr_matrix = build_temporal_corr_matrix(
                times=times, **temporal_corr
            )
        else:
            raise ValueError("temporal_corr must be a dict or None")

        # Combine spatial and temporal correlations using Kronecker product
        sigma = np.diag(np.sqrt(variances))  # Scale by variances
        kron = np.kron(temporal_corr_matrix, spatial_corr_matrix)  # order matters
        S_0 = sigma @ kron @ sigma
        return cls(S_0, index=index)

    def apply_temporal_decay(
        self,
        length_scale: pd.Timedelta | str,
        interday: bool = True,
        time_dim: str = "obs_time",
    ) -> Self:
        """
        Apply temporal decay to the covariance matrix.

        Parameters
        ----------
        length_scale : pd.Timedelta or str
            Length scale for the temporal decay (e.g., '32h' for 32 hours)
        interday : bool
            Whether to apply decay across days. If False, decay is only applied
            within the same day.
        time_dim : str
            Name of the time dimension

        Returns
        -------
        SpaceTimeCovariance
            A new instance with time decay applied
        """

        # Define the decay function
        def calculate_temporal_decay(
            group: pd.DataFrame, time_dim, length_scale
        ) -> np.ndarray:
            times = group[time_dim]  # Extract times for this group
            return build_temporal_corr_matrix(times=times, length_scale=length_scale)

        # Find all dimensions other than the decay dimension
        groupers = [col for col in self.index.names if col != time_dim]

        # If not spanning days, add the date as a grouper
        if not interday:
            dates = self.index.get_level_values(time_dim).date
            groupers.append(dates)

        # Apply the decay
        return self.apply_groupwise(
            groupers=groupers,
            func=calculate_temporal_decay,
            time_dim=time_dim,
            length_scale=length_scale,
            diagonal=False,
        )

    def apply_spatial_decay(
        self,
        length_scale: float,
        location_latlon_mapper: dict[str, tuple[float, float]],
        earth_radius: float | None = None,
        spatial_dim: str = "obs_location",
    ) -> Self:
        """
        Apply spatial decay to the covariance matrix.

        Parameters
        ----------
        length_scale : float
            Length scale for the spatial decay (in km)
        location_latlon_mapper : dict
            Dictionary mapping location names to (latitude, longitude) tuples
        earth_radius : float
            Earth radius in km
        spatial_dim : str
            Name of the spatial dimension

        Returns
        -------
        SpaceTimeCovariance
            A new instance with spatial decay applied
        """

        # Define the decay function
        def calculate_spatial_decay(
            group: pd.DataFrame,
            spatial_dim: str,
            length_scale: float,
            earth_radius: float,
        ) -> np.ndarray:
            # Get lat/lon for each location in the group
            lats = [
                location_latlon_mapper[loc][0] for loc in group[spatial_dim]
            ]  # TODO make this more flexible
            lons = [location_latlon_mapper[loc][1] for loc in group[spatial_dim]]
            return build_latlon_corr_matrix(
                lats=lats,
                lons=lons,
                length_scale=length_scale,
                earth_radius=earth_radius,
            )

        # Find all dimensions other than the decay dimension
        groupers = [col for col in self.index.names if col != spatial_dim]

        # Apply the decay
        return self.apply_groupwise(
            groupers=groupers,
            func=calculate_spatial_decay,
            spatial_dim=spatial_dim,
            length_scale=length_scale,
            earth_radius=earth_radius,
            diagonal=False,
        )

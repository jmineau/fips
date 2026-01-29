"""Utility functions for inversion module.

This module provides reusable utility functions for:
- Spatial distance calculations and time binning (data processing)
- Parallelization and task execution with timeouts
- Data validation and conversion between pandas and xarray formats
- Index manipulation and filtering operations
- Pickle loading from file paths
"""

import multiprocessing
import pickle
import signal
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, Literal, TypeVar, overload

import numpy as np
import pandas as pd
import xarray as xr

T = TypeVar("T")

# ==============================================================================
# PICKLE LOADING
# ==============================================================================


def load_or_pass(obj: str | Path | T) -> T:
    """
    Load an object from a pickle file if obj is a file path, otherwise return as-is.

    Parameters
    ----------
    obj : str | Path | T
        Either a file path (str or Path) to a pickled object, or an object to pass through.

    Returns
    -------
    T
        The unpickled object or the input object.

    Raises
    ------
    FileNotFoundError
        If the path doesn't exist.
    pickle.UnpicklingError
        If the file cannot be unpickled.
    """
    if isinstance(obj, (str, Path)):
        path = Path(obj)
        if not path.exists():
            raise FileNotFoundError(f"Pickle file not found: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
    return obj


# ==============================================================================
# EXECUTION & PARALLELIZATION
# ==============================================================================


def exec_with_timeout(func, timeout, kwargs, item):
    """Helper to execute a function with a timeout using signals."""

    def handler(signum, frame):
        raise TimeoutError(f"Task timed out after {timeout} seconds")

    # Register the signal function handler
    signal.signal(signal.SIGALRM, handler)
    # Define a timeout for the function (supports floats)
    signal.setitimer(signal.ITIMER_REAL, timeout)

    try:
        return func(item, **kwargs)
    finally:
        # Disable the alarm
        signal.setitimer(signal.ITIMER_REAL, 0)


def parallelize(
    func: Callable,
    num_processes: int | Literal["max"] = 1,
    timeout: float | None = None,
) -> Callable:
    """
    Parallelize a function across an iterable.

    Parameters
    ----------
    func : function
        The function to parallelize.
    num_processes : int or 'max', optional
        The number of processes to use. Uses the minimum of the number of
        items in the iterable and the number of CPUs requested. If 'max',
        uses all available CPUs. Default is 1.
    timeout : float, optional
        The maximum time (in seconds) allowed for each item to be processed.
        If a task exceeds this time, a TimeoutError is raised.
        Default is None (no timeout).

    Returns
    -------
    parallelized : function
        A function that will execute the input function in parallel across
        an iterable.
    """

    def parallelized(iterable, **kwargs) -> list[Any]:
        """
        Execute the input function in parallel across an iterable.

        Parameters
        ----------
        iterable : iterable
            The iterable to parallelize the function across.
        **kwargs : dict
            Additional keyword arguments to pass to the function.

        Returns
        -------
        results : list
            The results of the function applied to each item in the iterable.
        """
        if len(iterable) == 0:
            raise ValueError("Iterable is empty; nothing to parallelize.")

        # Determine the number of processes to use
        cpu_count = multiprocessing.cpu_count()
        if num_processes == "max" or num_processes > cpu_count:
            processes = cpu_count
        else:
            processes = num_processes

        if processes > len(iterable):
            processes = len(iterable)

        # If only one process is requested, execute the function sequentially
        if processes == 1:
            if timeout is not None:
                results = [
                    exec_with_timeout(func, timeout, kwargs, i) for i in iterable
                ]
            else:
                results = [func(i, **kwargs) for i in iterable]
            return results

        # Create a multiprocessing Pool
        pool = multiprocessing.Pool(processes=processes)

        try:
            # Use the pool to map the function across the iterable
            if timeout is not None:
                # Use partial to bind func, timeout, and kwargs.
                # The iterable item is passed as the last argument by pool.map
                worker = partial(exec_with_timeout, func, timeout, kwargs)
                results = pool.map(func=worker, iterable=iterable)
            else:
                results = pool.map(func=partial(func, **kwargs), iterable=iterable)
        except Exception:
            pool.terminate()
            raise
        else:
            # Close the pool to free resources
            pool.close()
        finally:
            pool.join()

        return results

    return parallelized


# ==============================================================================
# XARRAY CONVERSION
# ==============================================================================


def series_to_xarray(series: pd.Series, name=None) -> xr.DataArray:
    """
    Convert a Pandas Series to an Xarray DataArray.

    Parameters
    ----------
    series : pd.Series
        Pandas Series to convert.
    name : str, optional
        Name for the resulting DataArray.

    Returns
    -------
    xr.DataArray
        Xarray DataArray representation of the series.
    """
    series = series.copy()
    if name is not None:
        series.name = name
    return series.to_xarray()


def dataframe_to_xarray(df: pd.DataFrame, name=None) -> xr.DataArray:
    """
    Convert a Pandas DataFrame to an Xarray DataArray.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame to convert.
    name : str, optional
        Name for the resulting DataArray.

    Returns
    -------
    xr.DataArray
        Xarray DataArray representation of the DataFrame.
    """
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # Stack all levels of the columns MultiIndex into the index
        n_levels = len(df.columns.levels)
        s = df.stack(list(range(n_levels)), future_stack=True)
    else:
        s = df.stack(future_stack=True)
    if isinstance(s, pd.DataFrame):
        raise ValueError("DataFrame could not be stacked into a Series.")
    return series_to_xarray(series=s, name=name)


# ==============================================================================
# INTERVAL FILTERING
# ==============================================================================


def enough_obs_per_interval(
    index: pd.Index,
    intervals: pd.IntervalIndex,
    threshold: int,
    level: str | None = None,
) -> list[bool]:
    """
    Determine which observations have enough data points per time interval.

    Parameters
    ----------
    index : pd.Index
        Index containing observations.
    intervals : pd.IntervalIndex
        Intervals to group observations into.
    threshold : int
        Minimum number of observations required per interval.
    level : str, optional
        Level name to use if index is a MultiIndex. If None, uses the entire index.

    Returns
    -------
    list[bool]
        Boolean mask indicating which observations meet the threshold.
    """
    obs = index if level is None else index.get_level_values(level)
    groups = pd.Index(pd.cut(obs, bins=intervals))
    counts = obs.to_series().groupby(groups).transform("count")
    return (counts >= threshold).tolist()


@overload
def filter_intervals(
    data: pd.Series,
    intervals: pd.IntervalIndex,
    threshold: int,
    level: str | None = None,
) -> pd.Series: ...


@overload
def filter_intervals(
    data: pd.DataFrame,
    intervals: pd.IntervalIndex,
    threshold: int,
    level: str | None = None,
) -> pd.DataFrame: ...


def filter_intervals(
    data: pd.Series | pd.DataFrame,
    intervals: pd.IntervalIndex,
    threshold: int,
    level: str | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Filter data to only include observations with enough data points per time interval.

    Parameters
    ----------
    data : pd.Series | pd.DataFrame
        Data to filter.
    intervals : pd.IntervalIndex
        Intervals to group observations into.
    threshold : int
        Minimum number of observations required per interval.
    level : str, optional
        Level name to use if index is a MultiIndex. If None, uses the entire index.

    Returns
    -------
    pd.Series | pd.DataFrame
        Filtered data.
    """
    mask = enough_obs_per_interval(
        index=data.index, intervals=intervals, threshold=threshold, level=level
    )
    return data[mask]


# ==============================================================================
# SPATIAL & TEMPORAL UTILITIES
# ==============================================================================


def haversine_matrix(lats, lons, earth_radius=6371.0, deg=True):
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

"""
Utility functions for inversion module.
"""

import multiprocessing
import signal
from collections.abc import Callable
from functools import partial
from typing import Any, Literal, overload

import pandas as pd
import xarray as xr
from pandas.api.types import is_float_dtype


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


@overload
def validate_single_column_df(name: str, obj: pd.DataFrame) -> pd.Series: ...


@overload
def validate_single_column_df(name: str, obj: pd.Series) -> pd.Series: ...


@overload
def validate_single_column_df(name: str, obj: Any) -> Any: ...


def validate_single_column_df(name: str, obj: Any) -> Any | pd.Series:
    if isinstance(obj, pd.DataFrame):
        ncols = obj.shape[1]
        if ncols > 1:
            raise ValueError(f"{name} DataFrame must have a single column.")
        return obj.iloc[:, 0]
    return obj


def round_index(
    index: pd.Index | pd.MultiIndex, decimals: int
) -> pd.Index | pd.MultiIndex:
    """
    Rounds the values in a pandas Index or MultiIndex if the level's
    data type is a numpy floating type.

    Parameters
    ----------
    index : pd.Index | pd.MultiIndex
        Input index to round.
    decimals : int
        Number of decimal places to round to.

    Returns
    -------
    pd.Index | pd.MultiIndex
        Rounded index.
    """
    if not isinstance(index, (pd.Index, pd.MultiIndex)):
        raise TypeError("Input must be a pandas Index or MultiIndex.")

    if isinstance(index, pd.MultiIndex):
        # Handle MultiIndex
        new_levels = []
        changed = False
        for i in range(index.nlevels):
            level = index.get_level_values(i)
            if is_float_dtype(level):
                # Round the level if it's a float type
                new_levels.append(level.round(decimals))
                changed = True
            else:
                new_levels.append(level)

        if changed:
            # Reconstruct the MultiIndex with the new, rounded levels
            return pd.MultiIndex.from_arrays(new_levels, names=index.names)
            # return new_levels
        else:
            # Return original index if no levels were changed
            return index

    elif is_float_dtype(index.dtype):
        # Handle single Index
        return index.round(decimals)
    else:
        # Return original index if it's not a float type
        return index


def series_to_xarray(series: pd.Series, name=None) -> xr.DataArray:
    """
    Convert a Pandas Series to an Xarray DataArray.

    Parameters
    ----------
    series : pd.Series
        Pandas Series to convert.
    attr : str
        Attribute name.

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
    name : str
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
    pd.Series
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

import numpy as np
import pandas as pd


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
    decay_matrix = np.exp(-diffs / decay).values  # values gets the numpy array
    return decay_matrix

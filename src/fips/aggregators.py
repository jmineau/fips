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

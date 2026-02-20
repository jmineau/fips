from typing import overload

import pandas as pd

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
    counts = obs.to_series().groupby(groups, observed=True).transform("count")
    return (counts >= threshold).tolist()


@overload
def select_intervals_with_min_obs(
    data: pd.Series,
    intervals: pd.IntervalIndex,
    threshold: int,
    level: str | None = None,
) -> pd.Series: ...


@overload
def select_intervals_with_min_obs(
    data: pd.DataFrame,
    intervals: pd.IntervalIndex,
    threshold: int,
    level: str | None = None,
) -> pd.DataFrame: ...


def select_intervals_with_min_obs(
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

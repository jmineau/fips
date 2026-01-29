"""Index validation and manipulation utilities.

This module provides utilities for checking index overlap, promoting indices,
and sanitizing index types for consistent handling across data structures.
"""

from typing import overload
import warnings

import pandas as pd


def check_overlap(target_idx: pd.Index, available_idx: pd.Index, name: str):
    """Warns if the intersection between target vector index and available matrix index is poor."""
    intersection = target_idx.intersection(available_idx)
    if len(intersection) == 0:
        warnings.warn(
            f"No overlap found between {name} Vector and Matrix indices. "
            f"The matrix will be filled with zeros, leading to a disconnected problem.",
            UserWarning,
            stacklevel=2,
        )
    elif len(intersection) < len(target_idx):
        missing = len(target_idx) - len(intersection)
        warnings.warn(
            f"Partial overlap for {name}: {missing} / {len(target_idx)} vector elements "
            f"are missing from the provided matrix and will be zero-filled.",
            UserWarning,
            stacklevel=2,
        )


def promote_index(index: pd.Index, promotion, promotion_level):
    df = index.to_frame(index=False)
    df.insert(0, promotion_level, promotion)
    return pd.MultiIndex.from_frame(df)


def sanitize_index(index: pd.Index, decimals: int | None = None) -> pd.Index:
    """
    Helper to convert index to float (if possible) and round to specified decimals.
    Handles both flat Index and MultiIndex.
    """

    # Handle MultiIndex recursively
    if isinstance(index, pd.MultiIndex):
        new_levels = [sanitize_index(level, decimals) for level in index.levels]
        return index.set_levels(new_levels, level=range(len(new_levels)))

    # Try converting to numeric (handles strings like "1.00")
    if not pd.api.types.is_numeric_dtype(
        index
    ) and not pd.api.types.is_datetime64_dtype(index):
        try:
            numeric_index = pd.to_numeric(index, errors="raise")
            index = pd.Index(numeric_index, name=index.name)
        except (ValueError, TypeError):
            # If conversion fails (e.g. text labels), keep as is
            pass

    if decimals is not None:
        # If it is now float (or was already), round it
        if pd.api.types.is_float_dtype(index):
            return index.round(decimals)

    return index


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


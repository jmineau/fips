"""Index validation and manipulation utilities.

This module provides utilities for checking index overlap, promoting indices,
and sanitizing index types for consistent handling across data structures.
"""

import warnings
from typing import overload

import numpy as np
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


def ensure_block_axis(frame: pd.DataFrame, axis: str, block: str) -> pd.DataFrame:
    if axis == "index":
        idx = frame.index
        if "block" not in idx.names:
            frame.index = promote_index(idx, promotion=block, promotion_level="block")
        elif idx.get_level_values("block").unique()[0] != block:
            frame.index = update_level_values(idx, "block", block)
    else:
        cols = frame.columns
        if "block" not in cols.names:
            frame.columns = promote_index(
                cols, promotion=block, promotion_level="block"
            )
        elif cols.get_level_values("block").unique()[0] != block:
            frame.columns = update_level_values(cols, "block", block)
    return frame


def ensure_block(
    frame: pd.DataFrame, block: str, axes=("index", "columns")
) -> pd.DataFrame:
    for axis in axes:
        frame = ensure_block_axis(frame, axis, block)
    return frame


def get_index(obj, axis: str | int = 0) -> pd.Index:
    """Get index or columns from DataFrame based on axis."""
    if axis in (0, "index"):
        return obj.index
    elif axis in (1, "columns"):
        return obj.columns
    else:
        raise ValueError("axis must be 0/'index' or 1/'columns'")


def set_index(obj: pd.DataFrame | pd.Series, index: pd.Index, axis: str | int = 0):
    """Set index or columns on DataFrame/Series based on axis."""
    if axis in (0, "index"):
        obj.index = index
    elif axis in (1, "columns"):
        if not isinstance(obj, pd.DataFrame):
            raise TypeError("Cannot set columns on a Series.")
        obj.columns = index
    else:
        raise ValueError("axis must be 0/'index' or 1/'columns'")
    return obj


def xs(
    obj: pd.DataFrame | pd.Series, key, axis=0, level=None, drop_level=True
) -> pd.DataFrame | pd.Series:
    if axis == "both":
        xs_0 = xs(obj, key, axis=0, level=level, drop_level=drop_level)
        return xs(xs_0, key, axis=1, level=level, drop_level=drop_level)

    # Call the underlying xs method
    s_or_df = obj.xs(key, axis=axis, level=level, drop_level=drop_level).copy()

    # Drop index levels that are all nan
    idx = get_index(s_or_df, axis=axis)
    idx = idx.droplevel(
        [level for level in idx.names if idx.get_level_values(level).isna().all()]
    )

    # Reassign cleaned index/columns
    set_index(s_or_df, idx, axis=axis)

    return s_or_df


def update_level_values(index: pd.Index, level: str, values) -> pd.Index:
    """Update specific level values in a MultiIndex."""
    if not isinstance(index, pd.MultiIndex):
        raise TypeError("index must be a pd.MultiIndex to update level values.")
    idx = index.to_frame(index=False)
    idx[level] = values
    return pd.MultiIndex.from_frame(idx)


def outer_align_levels(
    dfs: list[pd.DataFrame], axis=0, fill_value=np.nan
) -> list[pd.DataFrame]:
    """
    Aligns MultiIndexes by performing an OUTER JOIN on level names,
    strictly preserving the order of appearance (First-Seen Priority).
    """
    # Recursive composition for aligning both axes
    if axis == "both":
        dfs = outer_align_levels(dfs, axis=0, fill_value=fill_value)
        dfs = outer_align_levels(dfs, axis=1, fill_value=fill_value)
        return dfs

    # Build the Master Schema (Order of Appearance)
    # The first DF establishes the top of the hierarchy.
    # Later DFs only append NEW levels to the end.
    schema_names = []
    for df in dfs:
        idx = get_index(df, axis=axis)
        for name in idx.names:
            if name not in schema_names:
                schema_names.append(name)

    # Align DataFrames to Schema
    aligned_dfs = []
    for df in dfs:
        df_out = df.copy()
        idx = get_index(df_out, axis=axis)

        # Extract current levels into a DataFrame
        idx_df = idx.to_frame(index=False)

        # Fill missing levels with NaN
        for name in schema_names:
            if name not in idx_df.columns:
                idx_df[name] = fill_value

        # Reorder columns to match Master Schema exactly.
        idx_df = idx_df[schema_names]

        # Rebuild MultiIndex
        new_idx = pd.MultiIndex.from_frame(idx_df)
        set_index(df_out, new_idx, axis=axis)

        aligned_dfs.append(df_out)
    return aligned_dfs


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
        df = index.to_frame(index=False)
        for col in df.columns:
            sanitized = sanitize_index(pd.Index(df[col], name=col), decimals=decimals)
            df[col] = sanitized.values
        return pd.MultiIndex.from_frame(df, names=index.names)

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

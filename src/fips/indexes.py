"""
Index validation and manipulation utilities.

This module provides utilities for checking index overlap, promoting indices,
and sanitizing index types for consistent handling across data structures.
"""

from functools import wraps
from typing import Literal

import numpy as np
import pandas as pd


def apply_to_index(func):
    """
    Apply a single-index function to each level of a MultiIndex.

    Parameters
    ----------
    func : function
        A function that takes a single pd.Index and returns a modified pd.Index.
        This function will be applied to each level of a MultiIndex, or directly to a single Index.

    Returns
    -------
    function
        A wrapper function that applies the given function to each level of a MultiIndex or to a single Index.
    """

    @wraps(func)
    def wrapper(index: pd.Index, *args, **kwargs) -> pd.Index:
        if isinstance(index, pd.MultiIndex):
            # Extract fully materialized arrays, apply the function, and rebuild
            new_arrays = [
                func(index.get_level_values(i), *args, **kwargs)
                for i in range(index.nlevels)
            ]
            return pd.MultiIndex.from_arrays(new_arrays, names=index.names)

        # Fallback for single Index
        return func(index, *args, **kwargs)

    return wrapper


def assign_block(index: pd.Index, block: str) -> pd.Index:
    """
    Assign or overwrite a 'block' level in the index.

    Parameters
    ----------
    index : pd.Index
        The original index to which the block level will be assigned.
    block : str
        The block name to assign to the index.

    Returns
    -------
    pd.Index
        A new index with the 'block' level assigned to the specified block name.
    """
    arrays = [index.get_level_values(i) for i in range(index.nlevels)]
    names = list(index.names)

    if "block" in names:
        block_idx = names.index("block")
        arrays[block_idx] = np.full(len(index), block)
    else:
        arrays.insert(0, np.full(len(index), block))
        names.insert(0, "block")

    if len(arrays) > 1:
        return pd.MultiIndex.from_arrays(arrays, names=names)
    return pd.Index(arrays[0], name=names[0])


def outer_align_levels(
    dfs: list[pd.DataFrame], axis: int | Literal["both"] = 0, fill_value=np.nan
) -> list[pd.DataFrame]:
    """
    Align MultiIndexes by performing an OUTER JOIN on level names.

    Strictly preserves the order of appearance (First-Seen Priority).

    Parameters
    ----------
    dfs : list of pd.DataFrame
        The DataFrames to align.
    axis : int or 'both', default 0
        The axis along which to align the DataFrames. 0 or 'index' for row alignment, 1 or 'columns' for column alignment, 'both' for both axes.
    fill_value : scalar, default np.nan
        The value to use for missing entries after alignment. By default, missing entries are filled with NaN.

    Returns
    -------
    list of pd.DataFrame
        A list of DataFrames with aligned MultiIndexes along the specified axis.
    """
    working_dfs = dfs.copy()

    for ax in resolve_axes(axis):
        # Build the Master Schema (Order of Appearance)
        # The first DF establishes the top of the hierarchy.
        # Later DFs only append NEW levels to the end.
        schema_names = []
        for df in working_dfs:
            for name in df.axes[ax].names:
                if name not in schema_names:
                    schema_names.append(name)

        # Align DataFrames to Schema
        for i, df in enumerate(working_dfs):
            idx = df.axes[ax]
            arrays = []

            for name in schema_names:
                if name in idx.names:
                    arrays.append(idx.get_level_values(name))
                else:
                    arrays.append(np.full(len(idx), fill_value))

            new_idx = pd.MultiIndex.from_arrays(arrays, names=schema_names)
            new_df = df.set_axis(new_idx, axis=ax)
            working_dfs[i] = new_df  # replace with aligned DataFrame

    return working_dfs


def overlaps(
    target_idx: pd.Index, available_idx: pd.Index
) -> bool | Literal["partial"]:
    """
    Check if target index overlaps with available index.

    Returns True if fully covered, 'partial' if partially covered, and False if no overlap.

    Parameters
    ----------
    target_idx : pd.Index
        The index we want to check for coverage.
    available_idx : pd.Index
        The index that represents available data.

    Returns
    -------
    bool or 'partial'
        True if target_idx is fully covered by available_idx, 'partial' if partially covered, and False if no overlap.
    """
    intersection = target_idx.intersection(available_idx)
    if len(intersection) == 0:
        return False
    elif len(intersection) < len(target_idx):
        return "partial"
    return True


def resolve_axes(axis: int | str | Literal["both"]) -> tuple[int, ...]:
    """Standardize axis input into a tuple of integer axes."""
    if axis == "both":
        return (0, 1)
    if axis in (0, "index"):
        return (0,)
    if axis in (1, "columns"):
        return (1,)
    raise ValueError("axis must be 0, 1, 'index', 'columns', or 'both'")


@apply_to_index
def round_index(index: pd.Index, decimals: int) -> pd.Index:
    """
    Round float indices to specified decimals.

    Parameters
    ----------
    index : pd.Index
        The index to round.
    decimals : int
        The number of decimal places to round to.

    Returns
    -------
    pd.Index
        A new index with float values rounded to the specified number of decimals.
        Non-float indices are returned unchanged.
    """
    if pd.api.types.is_float_dtype(index):
        return index.round(decimals)
    return index


@apply_to_index
def to_numeric(index: pd.Index) -> pd.Index:
    """
    Attempt to convert index to numeric types.

    Parameters
    ----------
    index : pd.Index
        The index to convert.

    Returns
    -------
    pd.Index
        A new index with values converted to numeric types where possible.
        Non-convertible values are returned unchanged.
    """
    # Prevent converting Datetime/Timedelta/Period indices to numeric
    if (
        pd.api.types.is_datetime64_any_dtype(index)
        or pd.api.types.is_timedelta64_dtype(index)
        or isinstance(index, pd.PeriodDtype)
    ):
        return index

    try:
        numeric_index = pd.to_numeric(index, errors="raise")
        return pd.Index(numeric_index, name=index.name)
    except (ValueError, TypeError):
        return index

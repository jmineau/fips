"""Index validation and manipulation utilities.

This module provides utilities for checking index overlap, promoting indices,
and sanitizing index types for consistent handling across data structures.
"""

from functools import wraps
from typing import Literal

import numpy as np
import pandas as pd


def apply_to_index(func):
    """Decorator to apply a single-index function to each level of a MultiIndex."""

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
    """Assigns or overwrites a 'block' level in the index."""
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
    dfs: list[pd.DataFrame], axis=0, fill_value=np.nan
) -> list[pd.DataFrame]:
    """
    Aligns MultiIndexes by performing an OUTER JOIN on level names,
    strictly preserving the order of appearance (First-Seen Priority).
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
    Returns True if fully covered, 'partial' if partially covered, and False if no overlap."""
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
    Round float indices to specified decimals. Non-float indices are returned unchanged.
    """
    if pd.api.types.is_float_dtype(index):
        return index.round(decimals)
    return index


@apply_to_index
def to_numeric(index: pd.Index) -> pd.Index:
    """
    Attempt to convert index to numeric types, if possible. If conversion fails, returns original index.
    """
    # Prevent converting Datetime/Timedelta/Period indices to numeric
    if pd.api.types.is_datetime64_any_dtype(index) or pd.api.types.is_timedelta64_dtype(
        index
    ) or isinstance(index, pd.PeriodDtype):
        return index

    try:
        numeric_index = pd.to_numeric(index, errors="raise")
        return pd.Index(numeric_index, name=index.name)
    except (ValueError, TypeError):
        return index

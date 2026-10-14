"""
Index validation and manipulation utilities.

This module provides utilities for checking index overlap and rounding labels
on request. fips never converts or rounds labels on its own; assembling blocks
and padding their levels is the job of :class:`~fips.Space`.
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
        """Apply ``func`` to every level of ``index`` and rebuild it."""
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
        # Round the underlying numpy array and reconstruct the index
        rounded_values = np.round(index.values, decimals)
        return pd.Index(rounded_values, name=index.name)
    return index

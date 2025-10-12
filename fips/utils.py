"""
Utility functions for inversion module.
"""

from functools import partial
import multiprocessing
from typing import Any, Callable, Literal

import pandas as pd
from pandas.api.types import is_float_dtype
import xarray as xr


def parallelize(func: Callable, num_processes: int | Literal['max'] = 1
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
        if num_processes == 'max':
            processes = cpu_count
        elif num_processes > cpu_count:
            processes = cpu_count
        else:
            processes = num_processes

        if processes > len(iterable):
            processes = len(iterable)

        # If only one process is requested, execute the function sequentially
        if processes == 1:
            results = [func(i, **kwargs) for i in iterable]
            return results

        # Create a multiprocessing Pool
        pool = multiprocessing.Pool(processes=processes)

        # Use the pool to map the function across the iterable
        results = pool.map(func=partial(func, **kwargs), iterable=iterable)

        # Close the pool to free resources
        pool.close()
        pool.join()

        return results

    return parallelized


def round_index(index: pd.Index | pd.MultiIndex, decimals: int
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


def dataframe_matrix_to_xarray(frame: pd.DataFrame) -> xr.DataArray:
    """
    Convert a pandas DataFrame to an xarray DataArray.

    If the DataFrame has a MultiIndex for columns, all levels of the MultiIndex
    are stacked into the index of the resulting DataArray.

    Parameters
    ----------
    frame : pd.DataFrame
        DataFrame to convert.

    Returns
    -------
    xr.DataArray
        Converted DataArray.
    """

    if isinstance(frame.columns, pd.MultiIndex):
        # Stack all levels of the columns MultiIndex into the index
        n_levels = len(frame.columns.levels)
        s = frame.stack(list(range(n_levels)), future_stack=True)
    else:
        s = frame.stack(future_stack=True)
    return s.to_xarray()

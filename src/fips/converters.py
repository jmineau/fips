"""
Utilities for converting between data structures.
"""

import pandas as pd
import xarray as xr


# ==============================================================================
# XARRAY CONVERSION
# ==============================================================================


def series_to_xarray(series: pd.Series, name=None) -> xr.DataArray:
    """
    Convert a Pandas Series to an Xarray DataArray.

    Parameters
    ----------
    series : pd.Series
        Pandas Series to convert.
    name : str, optional
        Name for the resulting DataArray.

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
    name : str, optional
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


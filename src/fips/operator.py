import numpy as np
import pandas as pd
import xarray as xr

from fips.utils import dataframe_matrix_to_xarray, round_index


def convolve(
    forward_operator: pd.DataFrame, state: pd.Series, coord_decimals: int = 6
) -> pd.Series:
    """
    Convolve a forward_operator with a state field to get modeled observations.

    Parameters
    ----------
    forward_operator : pd.DataFrame
        DataFrame with columns corresponding to the state index
        and rows corresponding to the observation index.
    state : pd.Series
        Series with rows corresponding to the state index.
    coord_decimals : int, optional
        Number of decimal places to round coordinates to when matching indices,
        by default 6.

    Returns
    -------
    pd.Series
        Series with the same index as the forward_operator,
        containing the modeled observations.
    """
    fo = forward_operator.copy()
    state = state.copy()

    # Round floating point coordinates to avoid precision issues
    fo.columns = round_index(fo.columns, decimals=coord_decimals)
    state.index = round_index(state.index, decimals=coord_decimals)

    # Ensure the state index matches the forward operator columns
    if isinstance(fo.columns, pd.MultiIndex):
        if not isinstance(state.index, pd.MultiIndex):
            raise ValueError(
                "If forward operator columns are a MultiIndex, state index must also be a MultiIndex."
            )
        state.index = state.index.reorder_levels(fo.columns.names)
    common = fo.columns.intersection(state.index)
    fo = fo.reindex(columns=common)
    state = state.reindex(index=common)

    if np.isnan(fo).any().any():
        raise ValueError("Forward operator contains NaN values after reindexing.")

    if np.isnan(state).any():
        raise ValueError("state contains NaN values after reindexing.")

    # Perform the matrix multiplication to get modeled observations
    modeled_obs = fo @ state
    modeled_obs.name = f"{state.name}_obs"

    return modeled_obs


class ForwardOperator:
    """
    Forward operator class for modeling observations.

    Parameters
    ----------
    data : pd.DataFrame
        Forward operator matrix.

    Attributes
    ----------
    data : pd.DataFrame
        Underlying forward operator matrix.
    obs_index : pd.Index

        Observation index (row index).
    state_index : pd.Index
        State index (column index).
    obs_dims : tuple
        Observation dimension names.
    state_dims : tuple
        State dimension names.

    Methods
    -------
    convolve(state: pd.Series, coord_decimals: int = 6) -> pd.Series
        Convolve the forward operator with a state vector.
    to_xarray() -> xr.DataArray
        Convert the forward operator to an xarray DataArray.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the ForwardOperator.

        Parameters
        ----------
        data : pd.DataFrame
            Forward operator matrix.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")
        self._data = data

    @property
    def data(self) -> pd.DataFrame:
        """
        Get the underlying data of the forward operator.

        Returns
        -------
        pd.DataFrame
            Forward operator matrix.
        """
        return self._data

    @property
    def obs_index(self) -> pd.Index:
        """
        Get the observation index (row index) of the forward operator.

        Returns
        -------
        pd.Index
            Observation index.
        """
        return self._data.index

    @property
    def state_index(self) -> pd.Index:
        """
        Get the state index (column index) of the forward operator.

        Returns
        -------
        pd.Index
            State index.
        """
        return self._data.columns

    @property
    def obs_dims(self) -> tuple:
        """
        Get the observation dimensions (names of the row index).

        Returns
        -------
        tuple
            Observation dimension names.
        """
        return tuple(self.obs_index.names)

    @property
    def state_dims(self) -> tuple:
        """
        Get the state dimensions (names of the column index).

        Returns
        -------
        tuple
            State dimension names.
        """
        return tuple(self.state_index.names)

    def convolve(self, state: pd.Series, coord_decimals: int = 6) -> pd.Series:
        """
        Convolve the forward operator with a state vector.

        Parameters
        ----------
        state : pd.Series
            State vector.
        coord_decimals : int, optional
            Number of decimal places to round coordinates to when matching indices,
            by default 6.

        Returns
        -------
        pd.Series
            Result of convolution.
        """
        return convolve(
            forward_operator=self._data, state=state, coord_decimals=coord_decimals
        )

    def to_xarray(self) -> xr.DataArray:
        """
        Convert the forward operator to an xarray DataArray.

        Returns
        -------
        xr.DataArray
            Xarray representation of the forward operator.
        """
        """Convert the forward operator to an xarray DataArray."""
        return dataframe_matrix_to_xarray(self._data)

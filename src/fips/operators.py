"""
Forward operators for inverse problems.

This module provides the `ForwardOperator` class, which represents the
mapping from state space to observation space, and related utilities for
convolving state vectors with the operator.
"""

import numpy as np
import pandas as pd

from fips.matrix import Matrix
from fips.vector import Vector, VectorLike


class ForwardOperator(Matrix):
    """
    Forward operator matrix mapping state vectors to observation space.

    A ForwardOperator wraps a pandas DataFrame and provides methods
    to convolve state vectors through the operator to produce modeled observations.

    The foward operator, or Jacobian matrix, is a key component of inverse problems.
    It defines how changes in the state vector affect the observations.
    The rows correspond to observations and the columns to state variables.

    Attributes
    ----------
    name : str
        Name of the forward operator. Optional.
    data : pd.DataFrame
        The underlying DataFrame containing the operator data.
    index : pd.MultiIndex
        Index for the rows of the ForwardOperator.
    obs_index : pd.MultiIndex
        Alias for index, representing the observation space index.
    columns : pd.MultiIndex
        Index for the columns of the ForwardOperator.
    state_index : pd.MultiIndex
        Alias for columns, representing the state space index.
    shape : tuple
        Shape of the ForwardOperator (number of rows, number of columns).
    values : np.ndarray
        The underlying data values as a NumPy array.
    is_sparse : bool
        Whether the data is stored in sparse format.

    Methods
    -------
    convolve(state, round_index=None, verify_overlap=True)
        Convolve a state vector through the forward operator.
    xs(key, axis=0, level=None, drop_level=True)
        Cross-select data based on index/column values.
    reindex(new_index, new_columns, fill_value=0.0)
        Reindex the matrix to new row and column indices, filling missing values with fill_value.
    round_index(decimals, axis='both')
        Round the index and/or columns to a specified number of decimal places for alignment.
    copy()
        Return a copy of the ForwardOperator.
    to_frame(add_block_level=False)
        Convert to a DataFrame, optionally adding block levels to the index and columns.
    to_dense()
        Return a copy of the matrix with dense internal storage.
    to_sparse(threshold=None)
        Return a copy of the matrix with sparse internal storage, zeroing values below the threshold.
    to_numpy()
        Get the underlying data as a NumPy array.
    """

    @property
    def state_index(self) -> pd.Index:
        """Return the state space index (columns)."""
        return self.columns

    @property
    def obs_index(self) -> pd.Index:
        """Return the observation space index (rows)."""
        return self.index

    def convolve(
        self,
        state: VectorLike,
        round_index: int | None = None,
        verify_overlap: bool = True,
    ) -> pd.Series:
        """Convolve a state vector through the forward operator."""
        state = Vector(state)

        if round_index:
            op = self.round_index(round_index, axis="both")
            state = state.round_index(round_index)
        else:
            op = self

        state = state.reindex(
            op.state_index, fill_value=0.0, verify_overlap=verify_overlap
        )

        x_vals = state.values
        if op.is_sparse:
            # Use scipy CSR for efficient sparse matrix-vector multiply
            y_values = op.data.sparse.to_coo().tocsr().dot(x_vals)
        else:
            y_values = op.data.values @ x_vals
        name = f"{state.name}_obs" if state.name else None
        return pd.Series(y_values, index=op.obs_index, name=name)


def convolve(
    state: Vector | pd.Series | np.ndarray,
    forward_operator: ForwardOperator | pd.DataFrame,
    round_index: int | None = None,
    verify_overlap: bool = True,
) -> pd.Series:
    """
    Convolve a state vector with a forward operator matrix.

    Parameters
    ----------
    state : Vector, pd.Series, or np.ndarray
        State vector to convolve. Can be a Vector, Series, or 1D array.
    forward_operator : ForwardOperator or pd.DataFrame
        Forward operator matrix to convolve with. Can be a ForwardOperator or a DataFrame.
    round_index : int, optional
        Number of decimal places to round indices for alignment. If None, no rounding is done.
    verify_overlap : bool, optional
        Whether to verify that the state index overlaps with the operator's state index. Defaults to True.

    Returns
    -------
    pd.Series
        The convolved observation vector.
    """
    forward_operator = ForwardOperator(forward_operator)

    return forward_operator.convolve(
        state=state, round_index=round_index, verify_overlap=verify_overlap
    )

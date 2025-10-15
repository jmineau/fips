import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from typing_extensions import Self


class SymmetricMatrix:
    """
    Symmetric matrix wrapper class for pandas DataFrames.

    Parameters
    ----------
    data : pd.DataFrame
        Symmetric matrix with identical row and column indices.

    Attributes
    ----------
    data : pd.DataFrame
        Symmetric matrix.
    index : pd.Index
        Index of the symmetric matrix.
    dims : tuple
        Dimension names of the symmetric matrix.
    values : np.ndarray
        Underlying data as a NumPy array.
    shape : tuple
        Dimensionality of the symmetric matrix.
    loc : SymmetricMatrix._Indexer
        Custom accessor for label-based selection and assignment.

    Methods
    -------
    from_numpy(array: np.ndarray, index: pd.Index) -> SymmetricMatrix
        Create a SymmetricMatrix from a NumPy array and an index.
    reindex(index: pd.Index, **kwargs) -> SymmetricMatrix
        Reindex the symmetric matrix, filling new entries with 0.
    reorder_levels(order) -> SymmetricMatrix
        Reorder the levels of a MultiIndex symmetric matrix.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the SymmetricMatrix with a square DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            Square symmetric matrix.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")
        if not data.index.equals(data.columns):
            raise ValueError(
                "Symmetric matrix must have identical row and column indices."
            )

        self._data = data

    @classmethod
    def from_numpy(cls, array: np.ndarray, index: pd.Index) -> Self:
        """
        Create a SymmetricMatrix from a NumPy array.

        Parameters
        ----------
        array : np.ndarray
            Symmetric matrix array.
        index : pd.Index
            Index for rows and columns.

        Returns
        -------
        SymmetricMatrix
            SymmetricMatrix instance.
        """
        return cls(pd.DataFrame(array, index=index, columns=index))

    @property
    def data(self) -> pd.DataFrame:
        """
        Returns the underlying data as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            Underlying symmetric matrix.
        """
        return self._data

    @property
    def dims(self) -> tuple:
        """
        Returns a tuple representing the dimension names of the matrix.

        Returns
        -------
        tuple
            Dimension names of the symmetric matrix.
        """
        return tuple(self.index.names)

    @property
    def index(self) -> pd.Index:
        """
        Returns the pandas Index of the matrix.

        Returns
        -------
        pd.Index
            Index of the symmetric matrix.
        """
        return self.data.index

    @index.setter
    def index(self, index: pd.Index) -> None:
        """
        Sets a new index for the symmetric matrix, ensuring it remains square.

        Parameters
        ----------
        index : pd.Index
            New index for the symmetric matrix.

        Raises
        ------
        TypeError
            If the index is not a pandas Index.
        ValueError
            If the index length does not match the number of rows/columns in the matrix.
        """
        if not isinstance(index, pd.Index):
            raise TypeError("Index must be a pandas Index.")
        if len(index) != self.data.shape[0]:
            raise ValueError(
                "Index length must match the number of rows/columns in the matrix."
            )
        self._data.index = index
        self._data.columns = index

    @property
    def values(self) -> np.ndarray:
        """
        Returns the underlying data as a NumPy array.

        Returns
        -------
        np.ndarray
            Underlying data array.
        """
        return self.data.to_numpy()

    @property
    def shape(self) -> tuple:
        """
        Returns a tuple representing the dimensionality of the matrix.

        Returns
        -------
        tuple
            Dimensionality of the symmetric matrix.
        """
        return self.data.shape

    def reindex(self, index: pd.Index, **kwargs) -> Self:
        """
        Reindex the symmetric matrix, filling new entries with 0.

        Parameters
        ----------
        index : pd.Index
            New index for the symmetric matrix.
        **kwargs : additional keyword arguments
            Passed to pandas' reindex method.

        Returns
        -------
        SymmetricMatrix
            Reindexed SymmetricMatrix instance.
        """
        reindexed_data = self.data.reindex(index=index, columns=index, **kwargs).fillna(
            0.0
        )
        return type(self)(data=reindexed_data)

    def reorder_levels(self, order) -> Self:
        """
        Reorder the levels of a MultiIndex symmetric matrix.

        Parameters
        ----------
        order : list
            New order for the levels.

        Returns
        -------
        SymmetricMatrix
            SymmetricMatrix instance with reordered levels.

        Raises
        ------
        TypeError
            If the index is not a MultiIndex.
        """
        if not isinstance(self.index, pd.MultiIndex):
            raise TypeError("Index must be a MultiIndex to reorder levels.")
        data = self.data.copy()
        data = data.reorder_levels(order, axis="index")
        data = data.reorder_levels(order, axis="columns")
        return type(self)(data=data)

    def sort_index(self, **kwargs) -> Self:
        """
        Sort the index of the symmetric matrix.

        Parameters
        ----------
        **kwargs : additional keyword arguments
            Passed to pandas' sort_index method.

        Returns
        -------
        SymmetricMatrix
            SymmetricMatrix instance with sorted index.
        """
        data = self.data.copy()
        data = data.sort_index(axis="index", **kwargs)
        data = data.sort_index(axis="columns", **kwargs)
        return type(self)(data=data)

    def operate(
        self,
        value: float | int,
        operation="replace",
        diagonal: bool | None = None,
        cross: dict | None = None,
        **kwargs,
    ) -> Self:
        """
        Operate on specific entries in the symmetric matrix.

        Parameters
        ----------
        value : float or int
            Value to assign or use in the operation.
        operation : {'replace', 'add', 'subtract', 'multiply', 'divide'}, optional
            Operation to perform with the value. Default is 'replace'.
        diagonal : bool, optional
            If True, only operate on diagonal entries. If False, only operate on off-diagonal entries.
            If None, operate on all selected entries. Default is None.
        cross : dict, optional
            Dictionary specifying cross indices for symmetric assignment.
            Keys are index level names, and values are the corresponding index values.
            If None, no cross assignment is performed. Default is None.
        **kwargs : additional keyword arguments
            Key-value pairs specifying index level names and their corresponding
            index values to select rows and columns for assignment. Use slice(None)
            to select all entries along a level.

        Returns
        -------
        SymmetricMatrix
            New SymmetricMatrix instance with the operation applied.
        """
        mat = self.data.copy()
        names = self.index.names

        cross = cross or {}

        # Build selection tuple-of-slices for the given level
        sel = tuple(kwargs.get(name, slice(None)) for name in names)

        # If cross assignment is specified, build cross selection
        if cross:
            cross_sel = tuple(cross.get(name, slice(None)) for name in names)
        else:
            cross_sel = sel

        # Update value based on operation
        if operation == "replace":
            pass  # value is already the value to assign
        elif operation == "add":
            value = mat.loc[sel, cross_sel] + value
        elif operation == "subtract":
            value = mat.loc[sel, cross_sel] - value
        elif operation == "multiply":
            value = mat.loc[sel, cross_sel] * value
        elif operation == "divide":
            value = mat.loc[sel, cross_sel] / value
        else:
            raise ValueError(f"Unsupported operation: {operation}")

        # Assign the value to the selected entries
        mat.loc[sel, cross_sel] = value

        # If cross assignment is specified, assign the transposed value
        if cross:
            # if scalar-> same value; otherwise transpose the array assigned
            if np.isscalar(value):
                mat.loc[cross_sel, sel] = value
            else:
                mat.loc[cross_sel, sel] = np.asarray(value).T

        return type(self)(mat)

    def __add__(self, other: Any) -> Self:
        """
        Add two SymmetricMatrix instances.

        Parameters
        ----------
        other : SymmetricMatrix
            Another SymmetricMatrix instance.

        Returns
        -------
        SymmetricMatrix
            Sum of the two SymmetricMatrix instances.
        """
        Klass = type(self)
        if isinstance(other, Klass):
            if not self.index.equals(other.index):
                raise ValueError("Indices must match for addition.")
            return Klass.from_numpy(
                self.data.values + other.data.values, index=self.index
            )
        elif np.isscalar(other):
            return Klass.from_numpy(self.data.values + other, index=self.index)
        else:
            raise TypeError(f"Cannot add {type(self)} and {type(other)}")


class CovarianceMatrix(SymmetricMatrix):
    """
    Covariance matrix class wrapping pandas DataFrames.

    Attributes
    ----------
    variance : pd.Series
        Series containing the variances (diagonal elements).
    """

    @property
    def variances(self) -> pd.Series:
        """
        Returns the diagonal of the covariance matrix (the variances).

        Returns
        -------
        pd.Series
            Series containing the variances.
        """
        return pd.Series(np.diag(self.data), index=self.index, name="variance")

    @classmethod
    def from_variances(
        cls,
        variances: pd.Series | np.ndarray | float,
        index: pd.Index | None = None,
        **kwargs,
    ) -> Self:
        # Normalize variances to a pandas Series
        variances = cls._variances_as_series(variances, index=index)

        # Create diagonal covariance matrix from variances
        index = variances.index
        values = np.diag(variances.to_numpy())

        return cls.from_numpy(array=values, index=index)

    @staticmethod
    def _variances_as_series(
        variances: pd.Series | xr.DataArray | np.ndarray | float,
        index: pd.Index | None = None,
    ) -> pd.Series:
        """
        Convert variances to a pandas Series with the given index.

        Parameters
        ----------
        variances : scalar | sequence | xr.DataArray | pd.Series
            Variances.
        index : pd.Index
            Index for the variances if variances is a scalar or sequence.
        Returns
        -------
        pd.Series
            Prior error variances as a pandas Series.
        """

        # If already a pandas Series, use it (warn if index argument is provided)
        if isinstance(variances, pd.Series):
            if index is not None:
                warnings.warn(
                    message="Provided 'index' is ignored when 'variances' is a pandas Series; "
                    "the Series' own index will be used.",
                    category=UserWarning,
                )
            return variances

        # Xarray -> pandas Series
        if isinstance(variances, xr.DataArray):
            return variances.to_series()

        # Scalar (any numeric scalar, numpy or python)
        if np.isscalar(variances):
            if index is None:
                raise ValueError("index must be provided if variances is a scalar")
            arr = np.full(len(index), variances)
            return pd.Series(arr, index=index)

        # Sequence-like (lists, tuples, numpy arrays, etc.), but exclude strings/bytes
        if isinstance(variances, (np.ndarray, Sequence)) and not isinstance(
            variances, (str, bytes)
        ):
            if index is None:
                raise ValueError("index must be provided if variances is a sequence")
            if len(variances) != len(index):
                raise ValueError("Length of variances must match length of index")
            return pd.Series(np.asarray(variances), index=index)

        raise ValueError(
            "variances must be a scalar, sequence, xr.DataArray, or pd.Series"
        )

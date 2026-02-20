"""
Data structures for inverse problems.

This module provides Block, Vector, and Matrix classes for organizing state and observation
data into structured hierarchies with automatic index management and serialization support.
"""

import logging
from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
from typing_extensions import Self

from fips.base import ArrayLike, MultiBlockMixin, SingleBlockMixin, Structure2D
from fips.indexes import assign_block, outer_align_levels

logger = logging.getLogger(__name__)

MatrixBlockLike: TypeAlias = "ArrayLike | MatrixBlock"
MatrixLike: TypeAlias = (
    "MatrixBlockLike | dict | Sequence[pd.DataFrame | MatrixBlock] | Matrix"
)


class MatrixBlock(SingleBlockMixin, Structure2D):
    """
    Single 2D data block with row and column block names.

    A MatrixBlock wraps a pandas DataFrame and can be initialized from an existing DataFrame or from raw values.

    MatrixBlocks are the fundamental 2D building units of `fips`,
    used to compose larger Matrix objects. MatrixBlocks represent the relationships
    between specific row and column blocks (e.g., state-to-observation mappings in forward operators,
    or covariance submatrices between specific state components). By organizing data into MatrixBlocks,
    users can create modular and interpretable representations of complex inverse problems, with clear
    semantics for how different components interact.
    """

    data: pd.DataFrame
    name: str

    def __init__(
        self,
        data: MatrixBlockLike,
        row_block: str,
        col_block: str,
        name: str | None = None,
        index: pd.Index | None = None,
        columns: pd.Index | None = None,
        dtype: Any = None,
        copy: bool = False,
    ):
        if isinstance(data, MatrixBlock):
            row_block = row_block or data.row_block
            col_block = col_block or data.col_block
            name = name or data.name
            data = data.data
            index = data.index
            columns = data.columns
            copy = True  # Always copy when passed a MatrixBlock

        self.row_block = row_block
        self.col_block = col_block
        name = name or f"{row_block}_{col_block}"

        super().__init__(
            data, name=name, index=index, columns=columns, dtype=dtype, copy=copy
        )

    def __repr__(self):
        header = (
            f"MatrixBlock(row_block='{self.row_block}', col_block='{self.col_block}')\n"
        )
        return header + repr(self.data)

    def to_frame(self, add_block_level=False) -> pd.DataFrame:
        df = self.data.copy()
        if add_block_level:
            df.index = assign_block(df.index, self.row_block)
            df.columns = assign_block(df.columns, self.col_block)
        return df

    def __getstate__(self):
        """Explicit pickle support: return state as dict."""
        return {
            "data": self.data,
            "name": self.name,
            "row_block": self.row_block,
            "col_block": self.col_block,
        }

    def __setstate__(self, state):
        """Explicit pickle support: restore state from dict."""
        self.data = state["data"]
        self.name = state["name"]
        self.row_block = state["row_block"]
        self.col_block = state["col_block"]


class Matrix(MultiBlockMixin, Structure2D):
    """
    Base class for all matrix-like objects in the inversion framework.
    Wraps a pandas DataFrame and ensures consistent index handling.

    Matrices represent 2D components of the inversion problem,
    such as forward operators and covariance matrices.
    """

    data: pd.DataFrame

    def __init__(
        self,
        data: MatrixLike,
        name: str | None = None,
        index: pd.Index | None = None,
        columns: pd.Index | None = None,
        dtype=None,
        copy=None,
    ):
        """

        Parameters
        ----------
        data : np.ndarray or pd.DataFrame or Matrix or scalar
            2D data representing the matrix.
        name : str, optional
            Name for the Matrix.
        index : pd.Index
            Index for the rows of the DataFrame.
        columns : pd.Index, optional
            Index for the columns of the DataFrame. If None, uses the same as `index`.
        dtype : data type, optional
            Data type to force.
        copy : bool, optional
            Whether to copy the data.

        Returns
        -------
        Matrix
            Instance of Matrix wrapping the DataFrame.
        """
        blocks = None

        # Accept Matrix - extract DataFrame
        if isinstance(data, Matrix):
            data = data.data
            index = data.index
            columns = data.columns
            copy = True  # Always copy when passed a Matrix

        elif isinstance(data, MatrixBlock):
            blocks = [data]

        elif isinstance(data, Sequence) and not isinstance(data, str):
            if any(isinstance(item, (pd.DataFrame, MatrixBlock)) for item in data):
                blocks = data

        elif isinstance(data, dict):
            # Allow dict mapping (row, col) -> data as a shorthand alternative
            blocks = [MatrixBlock(v, r, c) for (r, c), v in data.items()]

        if blocks:
            seen_blocks = set()
            dfs: list[pd.DataFrame] = []

            for block in blocks:
                block = MatrixBlock(block)

                if block.name in seen_blocks:
                    raise ValueError(f"Duplicate block name '{block.name}' found")
                seen_blocks.add(block.name)

                dfs.append(block.to_frame(add_block_level=True))

            # Align all blocks to the same index structure
            aligned_dfs = outer_align_levels(dfs, axis="both")

            # Concatenate aligned blocks
            data = pd.concat(aligned_dfs).fillna(0.0)

        # Accept scalar - create matrix with repeated value
        elif np.isscalar(data):
            n = len(index) if index is not None else 1
            m = len(columns) if columns is not None else n
            data = np.full((n, m), data)

        # Accept series - create diagonal matrix
        elif isinstance(data, pd.Series):
            name = name or data.name
            index = columns = data.index
            data = np.diag(data.to_numpy())

        # 1D array - treat as diagonal matrix
        elif isinstance(data, (np.ndarray, list)) and np.ndim(data) == 1:
            index = columns = index or pd.RangeIndex(len(data))
            data = np.diag(data)

        super().__init__(
            data, name=name, index=index, columns=columns, dtype=dtype, copy=copy
        )

    def __repr__(self):
        return f"<{self.__class__.__name__} : shape={self.shape}>"

    def __getitem__(self, block) -> pd.DataFrame:
        """Get the submatrix corresponding to the specified block (cross-block)."""
        return self.data.loc[block]

    def __add__(self, other: Any) -> Self:
        """
        Add two Matrix instances.

        Parameters
        ----------
        other : Matrix
            Another Matrix instance or a scalar to add.

        Returns
        -------
        Matrix
            Sum of the two Matrix instances.
        """
        Klass = type(self)
        if isinstance(other, Klass):
            if not self.index.equals(other.index) or not self.columns.equals(
                other.columns
            ):
                raise ValueError("Indices and columns must match for addition.")
            return Klass(
                self.data.values + other.data.values,
                index=self.index,
                columns=self.columns,
            )
        elif np.isscalar(other):
            return Klass(
                self.data.values + other, index=self.index, columns=self.columns
            )
        else:
            raise TypeError(f"Cannot add {type(self)} and {type(other)}")

    def scale(self, factor: float | int) -> Self:
        """
        Scale the matrix by a scalar factor.

        Parameters
        ----------
        factor : float or int
            Scalar factor to multiply the matrix by.

        Returns
        -------
        Matrix
            New Matrix instance scaled by the factor.
        """
        if not np.isscalar(factor):
            raise TypeError("Factor must be a scalar (float or int).")
        return type(self)(self.values * factor, index=self.index, columns=self.columns)

    def to_xarray(self):
        # TODO what is the best way to represent a 2D matrix in xarray?
        # Xarray does not like repeated levels (e.g. 'block' or in Covariance matrices where row/col indices are the same)
        raise NotImplementedError(
            "to_xarray method not implemented for base Matrix class."
        )


class SymmetricMatrix(Matrix):
    """
    Enforces symmetry and provides specialized matrix operations.
    """

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
            New Matrix instance with the operation applied.
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

        # If diagonal/off-diagonal selection is specified, mask accordingly
        if diagonal is not None:
            diag_mask = np.eye(len(mat), dtype=bool)
            if diagonal:
                # Keep only diagonal entries
                mat = mat.where(diag_mask, other=self.data)
            else:
                # Keep only off-diagonal entries
                mat = mat.where(~diag_mask, other=self.data)

        return type(self)(mat)

    def apply_groupwise(
        self,
        groupers: list[Any],
        func: Callable | float,
        diagonal: bool = True,
        scale_by_variance: bool = True,
        **kwargs,
    ) -> Self:
        """
        Generic method to apply decay (time or space) to the covariance matrix.

        Parameters
        ----------
        groupers : list
            List of pandas grouping objects (e.g., index level names) to group by.
        func : Callable | float
            Function that takes a group DataFrame and returns a numpy array.
        diagonal : bool
            If True, include diagonal elements in application. Default is True.
        **kwargs : dict
            Additional keyword arguments to pass to `func`.

        Returns
        -------
        SymmetricMatrix
            New SymmetricMatrix instance with func applied by group.
        """
        matrix = self.data.to_numpy(copy=True)
        diag_mask = np.eye(len(self.index), dtype=bool)
        off_diags = np.zeros_like(matrix)

        # Get the index coordinates as a DataFrame
        coords = self.index.to_frame(index=False)

        # Group by the identified dimensions
        groups = coords.groupby(groupers)

        # Calculate the submatrix for each group
        for _group_key, group in groups:
            group_indices = group.index.to_list()
            if callable(func):
                submatrix = func(group, **kwargs)
            else:
                # Assume scalar value
                submatrix = np.full((len(group_indices), len(group_indices)), func)
            off_diags[np.ix_(group_indices, group_indices)] = submatrix

        # Scale submatrix by variances
        if scale_by_variance:
            variances = np.diag(matrix)
            sigma = np.diag(np.sqrt(variances))
            off_diags = sigma @ off_diags @ sigma

        # Add the decayed off-diagonal terms to the original matrix
        mat = matrix.copy()
        if not diagonal:
            off_diags[diag_mask] = 0  # Zero out diagonal terms
        mat += off_diags

        return type(self)(mat, index=self.index)

"""
Data structures for inverse problems.

This module provides Block, Vector, and Matrix classes for organizing state and observation
data into structured hierarchies with automatic index management and serialization support.
"""

import logging
from collections.abc import Sequence
from functools import reduce
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

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame or MatrixBlock or scalar
        2D data representing the block. If MatrixBlock, creates a copy.
    row_block : str, optional
        Name of the row block (e.g., "state", "obs"). Required if data is not a MatrixBlock.
    col_block : str, optional
        Name of the column block (e.g., "state", "obs"). Required if data is not a MatrixBlock.
    name : str, optional
        Name for the block. If None, defaults to "{row_block}_{col_block}".
    index : pd.Index, optional
        Index for the rows of the DataFrame.
    columns : pd.Index, optional
        Index for the columns of the DataFrame. If None, uses the same as `index`.
    dtype : data type, optional
        Data type to force.
    copy : bool, optional
        Whether to copy the data.
    sparse : bool, default False
        If True, store the block in pandas sparse format.
        Sparsification is applied after initialization.
        Use threshold zeroing in your builder before passing data here for best results.
    """

    data: pd.DataFrame  # type: ignore[override]
    name: str  # type: ignore[override]
    row_block: str
    """Name of the row block (e.g., 'state', 'obs')."""
    col_block: str
    """Name of the column block (e.g., 'state', 'obs')."""

    def __init__(
        self,
        data: MatrixBlockLike,
        row_block: str | None = None,
        col_block: str | None = None,
        name: str | None = None,
        index: pd.Index | None = None,
        columns: pd.Index | None = None,
        dtype: Any = None,
        copy: bool = False,
        sparse: bool = False,
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
            data,
            name=name,
            index=index,
            columns=columns,
            dtype=dtype,
            copy=copy,
            sparse=sparse,
        )

    def _validate(self):
        super()._validate()

        if self.row_block is None or self.col_block is None:
            raise ValueError(
                "MatrixBlock must have both row_block and col_block defined."
            )

    def __repr__(self):
        """Return string representation."""
        header = (
            f"MatrixBlock(row_block='{self.row_block}', col_block='{self.col_block}')\n"
        )
        return header + repr(self.data)

    def to_frame(self, add_block_level=False) -> pd.DataFrame:
        """
        Convert to DataFrame with optional block level.

        Parameters
        ----------
        add_block_level : bool, default False
            Whether to add 'block' levels to the index and columns for the row and column blocks.

        Returns
        -------
        pd.DataFrame
            The DataFrame representation of the block, with optional block levels in the index and columns.
        """
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


class _MatrixBlockAccessor:
    """Accessor for retrieving MatrixBlock instances from a Matrix."""

    def __init__(self, matrix: "Matrix"):
        self._matrix = matrix

    def __getitem__(self, key: tuple[str, str]) -> MatrixBlock:
        row_block, col_block = key
        df = self._matrix.xs(row_block, level="block", axis=0)
        df = df.xs(col_block, level="block", axis=1)
        return MatrixBlock(df, row_block=row_block, col_block=col_block)


class Matrix(MultiBlockMixin, Structure2D):
    """
    Base class for all matrix-like objects in the inversion framework.

    Wraps a pandas DataFrame and ensures consistent index handling.
    Matrices represent 2D components of the inversion problem,
    such as forward operators and covariance matrices.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame or Matrix or scalar
        2D data representing the matrix.
    name : str, optional
        Name for the Matrix.
    index : pd.MultiIndex
        Index for the rows of the DataFrame.
    columns : pd.MultiIndex, optional
        Index for the columns of the DataFrame. If None, uses the same as `index`.
    dtype : data type, optional
        Data type to force.
    copy : bool, optional
        Whether to copy the data.
    sparse : bool, default False
        If True, store the assembled matrix in pandas sparse format.
        Sparsification is applied after block assembly; use threshold
        zeroing in your builder before passing data here.
    """

    data: pd.DataFrame  # type: ignore[override]
    """The underlying data, which must be numeric and non-NaN. Can be stored in sparse format."""

    def __init__(
        self,
        data: MatrixLike,
        name: str | None = None,
        index: pd.Index | None = None,
        columns: pd.Index | None = None,
        dtype=None,
        copy=None,
        sparse: bool = False,
    ):
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

            # Combine aligned blocks
            # combine_first performs an outer join, preserving all indices and filling missing values with NaN
            data = reduce(lambda left, right: left.combine_first(right), aligned_dfs)
            data = data.fillna(0.0)

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
            data,
            name=name,
            index=index,
            columns=columns,
            dtype=dtype,
            copy=copy,
            sparse=sparse,
        )

    def __repr__(self):
        """Return string representation."""
        return f"<{self.__class__.__name__} : shape={self.shape}>"

    def __getitem__(self, block: tuple[str, str]) -> pd.DataFrame:
        """Get the submatrix DataFrame for the given (row_block, col_block) tuple."""
        row_block, col_block = block
        df = self.xs(row_block, level="block", axis=0)
        return df.xs(col_block, level="block", axis=1)

    @property
    def blocks(self) -> _MatrixBlockAccessor:  # type: ignore[override]
        """Accessor for retrieving MatrixBlock instances from the Matrix."""
        return _MatrixBlockAccessor(self)

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
            # Densify both sides before adding; sparsity is not preserved through
            # arbitrary addition (fill positions become non-zero).
            a = self.data.sparse.to_dense() if self.is_sparse else self.data
            b = other.data.sparse.to_dense() if other.is_sparse else other.data
            return Klass(
                a.values + b.values,
                index=self.index,
                columns=self.columns,
            )
        elif np.isscalar(other):
            a = self.data.sparse.to_dense() if self.is_sparse else self.data
            return Klass(a.values + other, index=self.index, columns=self.columns)
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
        # Multiply via pandas so SparseDtype is preserved naturally
        scaled_data = self.data * factor
        result = type(self)(scaled_data, index=self.index, columns=self.columns)
        return result

    def to_xarray(self):
        """Convert to xarray (not implemented for base Matrix class)."""
        # TODO what is the best way to represent a 2D matrix in xarray?
        # Xarray does not like repeated levels (e.g. 'block' or in Covariance matrices where row/col indices are the same)
        raise NotImplementedError(
            "to_xarray method not implemented for base Matrix class."
        )

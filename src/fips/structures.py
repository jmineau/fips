"""
Data structures for inverse problems.

This module provides Block, Vector, and Matrix classes for organizing state and observation
data into structured hierarchies with automatic index management and serialization support.
"""

import logging
import warnings
from abc import ABC
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from typing_extensions import Self

from fips.converters import to_series
from fips.indices import check_overlap, promote_index, sanitize_index, xs
from fips.serialization import Pickleable, load_or_pass

logger = logging.getLogger(__name__)


class Structure(Pickleable, ABC):
    name: str | property
    data: pd.DataFrame | pd.Series

    @property
    def index(self) -> pd.Index:
        return self.data.index

    @property
    def values(self) -> npt.NDArray:
        return self.data.to_numpy()

    def xs(self, key, axis=0, level=None, drop_level=True):
        return xs(self.data, key=key, axis=axis, level=level, drop_level=drop_level)


class Structure1D(Structure, ABC):
    """Base class for 1D structures (Block, Vector)."""

    def to_series(self) -> pd.Series:
        """Get the underlying Series data."""
        return self.data


class Block(Structure1D):
    """Single data block with a named Series and consistent index.

    A Block wraps a pandas Series with automatic index name validation.
    Can be initialized from an existing Series or from raw values.
    """

    def __init__(
        self,
        data: pd.DataFrame | pd.Series | npt.ArrayLike,
        index: pd.Index | None = None,
        name: str | None = None,
    ):
        """Initialize a Block.

        Parameters
        ----------
        data : pd.Series or array-like
            Series data or raw values.
        index : pd.Index, optional
            Index for the Block. Required if data is array-like.
        name : str, optional
            Name for the Block. Required if data is array-like.
        """
        if isinstance(data, pd.DataFrame):
            data = to_series(data)

        # CASE 1: Input is already a Series
        if isinstance(data, pd.Series):
            self.data = data.copy()
            # Allow overriding the name if provided
            if name is not None:
                self.data.name = name

            # Warn if index argument is provided but ignored
            if index is not None:
                warnings.warn(
                    "Index argument is ignored when data is a pd.Series.",
                    UserWarning,
                    stacklevel=2,
                )

            # Validation: Series must have a name (from itself or argument)
            if self.data.name is None:
                raise ValueError(
                    "Series must have a name or 'name' arg must be provided."
                )

        # CASE 2: Input is raw values (requires index and name)
        else:
            if index is None or name is None:
                raise ValueError(
                    "If data is not a Series, 'index' and 'name' are required."
                )

            self.data = pd.Series(data, index=index, name=name)

        # Check for NaNs
        if self.data.isna().any():
            raise ValueError(f"Block '{self.name}' contains NaN values.")

        # Update dim names if missing
        old_names = list(self.data.index.names)
        new_names = [
            dim if dim is not None else f"{self.name}_{i}"
            for i, dim in enumerate(self.data.index.names)
        ]
        self.data.index.set_names(new_names, inplace=True)
        if new_names != old_names:
            logger.debug(
                f"Updated Block index names for {self.name}: {old_names} -> {new_names}"
            )

    def __getstate__(self):
        """Explicit pickle support: return state as dict."""
        return {"data": self.data}

    def __setstate__(self, state):
        """Explicit pickle support: restore state from dict."""
        self.data = state["data"]

    @property
    def name(self) -> str:
        return str(self.data.name)

    @property
    def dims(self) -> list[str]:
        return list(self.data.index.names)

    def __repr__(self):
        header = f"Block(name='{self.name}')\n"
        return header + repr(self.data)


class Vector(Structure1D):
    """State or observation vector composed of multiple Block objects.

    A Vector organizes one or more Blocks (prior, posterior, observations, etc.)
    into a single hierarchical structure with automatic index promotion.
    """

    def __init__(self, name: str, blocks: Sequence[Block | pd.Series]):
        """Initialize a Vector.

        Parameters
        ----------
        name : str
            Name for the Vector (e.g., 'prior', 'posterior', 'obs').
        blocks : Sequence[Block or pd.Series]
            Sequence of Block objects or Series (automatically converted to Blocks).
        """
        self.name = name
        self.blocks: dict[str | int, Block] = {}
        for block in blocks:
            block = block if isinstance(block, Block) else Block(block)
            if block.name in self.blocks:
                raise ValueError(f"Duplicate block name '{block.name}' found.")
            self.blocks[block.name] = block
        self._assemble()

    def __getstate__(self):
        """Explicit pickle support: return state as dict."""
        return {"name": self.name, "data": self.data}

    def __setstate__(self, state):
        """Explicit pickle support: restore state from dict."""
        self.name = state["name"]
        self.data = state["data"]
        # Reconstruct blocks from assembled data
        self.blocks = self._reconstruct_blocks_from_data(self.data)

    @staticmethod
    def _reconstruct_blocks_from_data(data: pd.Series) -> dict[str | int, Block]:
        """Reconstruct blocks dictionary from assembled Series.

        Parameters
        ----------
        data : pd.Series
            Series with MultiIndex containing 'block' level.

        Returns
        -------
        dict[str or int, Block]
            Dictionary mapping block names to Block objects.
        """
        blocks = {}
        if isinstance(data.index, pd.MultiIndex) and "block" in data.index.names:
            block_names = data.index.get_level_values("block").unique()
            for block_name in block_names:
                block_data = data.xs(block_name, level="block")
                blocks[block_name] = Block(block_data, name=block_name)
        return blocks

    @classmethod
    def from_series(
        cls, data: pd.Series, name: str | None = None, block: str | None = None
    ) -> "Vector":
        """Create a Vector from a Series with 'block' level in index.

        Parameters
        ----------
        data : pd.Series
            Series with MultiIndex containing 'block' level.
        name : str, optional
            Override the series name. If None, uses data.name.
        block : str, optional
            Block name when supplying a single block Series.
            Overrides the 'block' level in index if present.

        Returns
        -------
        Vector
            Vector constructed from the series blocks.
        """
        if isinstance(data, pd.DataFrame):
            data = to_series(data)

        if not isinstance(data, pd.Series):
            raise TypeError("data must be a pd.Series.")

        if name is not None:
            data.name = name
        if data.name is None:
            raise ValueError("Series must have a name or 'name' must be provided.")

        if "block" not in data.index.names:
            if block is None:
                block = f"{data.name}_block"
            data = pd.concat({block: data}, names=["block"] + list(data.index.names))
            logger.debug(
                f"Promoted Series to include block level '{block}' for Vector '{data.name}'"
            )
        elif block is not None:
            # Update block level to the provided block name
            idx = data.index.to_frame(index=False)
            idx["block"] = block
            data.index = pd.MultiIndex.from_frame(idx, names=data.index.names)
            logger.debug(f"Updated block level to '{block}' for Vector '{data.name}'")
        else:
            pass  # use existing 'block' level

        # Extract blocks from the series using shared helper
        blocks = list(cls._reconstruct_blocks_from_data(data).values())

        return cls(name=str(data.name), blocks=blocks)

    def _assemble(self):
        dfs: list[pd.DataFrame] = []
        dims = {}

        promotion_level = "block"

        def prep_block(block) -> pd.DataFrame:
            """Prepares the block for the vector assembly."""
            df = block.data.rename(self.name).to_frame()
            df = df.reset_index()
            df[promotion_level] = block.name
            return df

        for block in self.blocks.values():
            # Collect dimensions for the global index
            for dim in block.data.index.names:
                if dim:
                    dims[dim] = True

            # Get standardized dataframe
            dfs.append(prep_block(block))

        # Concatenate
        combined = pd.concat(dfs, ignore_index=True)

        # Define dynamic columns for MultiIndex
        dynamic_cols = [promotion_level] + list(dims.keys())

        # Handle Ragged Hierarchies (Fill integers with -1)
        # combined[list(dims.keys())] = combined[list(dims.keys())].fillna(-1)

        # Finalize
        self.data = combined.set_index(dynamic_cols)[self.name]

    def __repr__(self) -> str:
        header = f"Vector(name='{self.name}', n_blocks={len(self.blocks)})\n"
        return header + repr(self.data)

    def __getitem__(self, key) -> pd.Series:
        return self.blocks[key].data

    def __len__(self) -> int:
        return len(self.blocks)

    @property
    def n(self) -> int:
        return len(self.data)


class Matrix(Structure):
    """
    Base class for all matrix-like objects in the inversion framework.
    Wraps a pandas DataFrame and ensures consistent index handling.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        # Ensure indices are clean (standard numeric types)
        self.data.index = sanitize_index(self.data.index)
        self.data.columns = sanitize_index(self.data.columns)

    @classmethod
    def from_numpy(
        cls, array: np.ndarray, index: pd.Index, columns: pd.Index | None = None
    ) -> Self:
        """
        Create a Matrix instance from a numpy array with specified indices.

        Parameters
        ----------
        array : np.ndarray
            2D numpy array representing the matrix data.
        index : pd.Index
            Index for the rows of the DataFrame.
        columns : pd.Index, optional
            Index for the columns of the DataFrame. If None, uses the same as `index`.

        Returns
        -------
        Matrix
            Instance of Matrix wrapping the DataFrame.
        """
        if columns is None:
            columns = index
        df = pd.DataFrame(array, index=index, columns=columns)
        return cls(df)

    def __getstate__(self):
        """Explicit pickle support: return state as dict."""
        return {"data": self.data}

    def __setstate__(self, state):
        """Explicit pickle support: restore state from dict."""
        self.data = state["data"]

    @property
    def columns(self) -> pd.Index:
        return self.data.columns

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

    def __repr__(self):
        return f"<{self.__class__.__name__} : shape={self.shape}>"

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
        return type(self).from_numpy(
            self.values * factor, index=self.index, columns=self.columns
        )

    def copy(self, deep: bool = True) -> Self:
        """
        Create a copy of the SymmetricMatrix instance.

        Parameters
        ----------
        deep : bool, default True
            If True, perform a deep copy of the underlying data.

        Returns
        -------
        Matrix
            Copy of the Matrix instance.
        """
        return type(self)(self.data.copy(deep=deep))

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
            return Klass.from_numpy(
                self.data.values + other.data.values,
                index=self.index,
                columns=self.columns,
            )
        elif np.isscalar(other):
            return Klass.from_numpy(
                self.data.values + other, index=self.index, columns=self.columns
            )
        else:
            raise TypeError(f"Cannot add {type(self)} and {type(other)}")

    def to_frame(self) -> pd.DataFrame:
        """
        Get the underlying DataFrame data.

        Returns
        -------
        pd.DataFrame
            The DataFrame representation of the matrix.
        """
        return self.data


class SymmetricMatrix(Matrix):
    """
    Represents a symmetric matrix.
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
        Matrix
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

        return self.from_numpy(mat, index=self.index)


def prepare_vector(
    name: str,
    vector: str | Path | pd.Series | Block | Vector,
    block: str | None = None,
    float_precision: int | None = None,
) -> Vector:
    """Normalize and validate input into a Vector object.

    Parameters
    ----------
    name : str
        Name for the resulting Vector.
    vector : str, Path, Vector, pd.Series, or Block
        Vector data or path to pickled Vector.
    block : str, optional
        Block name when supplying a single block Series.
    float_precision : int, optional
        Round float indices to this many decimal places.

    Returns
    -------
    Vector
        Validated Vector object with sanitized indices.
    """
    # Load from pickle if path is provided
    vector = load_or_pass(vector)

    if isinstance(vector, (pd.Series, pd.DataFrame)):
        vector = Vector.from_series(data=to_series(vector), name=name, block=block)
    elif isinstance(vector, Block):
        vector = Vector(name=name, blocks=[vector])
    elif not isinstance(vector, Vector):
        raise TypeError(f"Cannot prepare vector from type: {type(vector)}")

    vector.data.index = sanitize_index(vector.data.index, float_precision)
    logger.debug(
        f"Prepared Vector '{vector.name}' with {len(vector.data.index)} elements"
    )

    return vector


def prepare_matrix(
    matrix: str | Path | pd.DataFrame | Matrix,
    matrix_class: type[Matrix],
    row_index: pd.Index,
    col_index: pd.Index,
    float_precision: int | None = None,
) -> Matrix:
    """Normalize and validate input into a Matrix object.

    Sanitizes, promotes if needed, and validates index overlap.

    Parameters
    ----------
    matrix : str, Path, pd.DataFrame, or Matrix
        Input matrix data (file path to pickled Matrix, DataFrame, or Matrix instance).
    matrix_class : type[Matrix]
        Class to wrap result in (ForwardOperator, CovarianceMatrix, etc.).
    row_index : pd.Index
        Target row index (assumed already promoted).
    col_index : pd.Index
        Target column index (assumed already promoted).
    float_precision : int, optional
        Decimals to round float indices to.

    Returns
    -------
    Matrix
        Instance of matrix_class wrapping the prepared DataFrame.
    """
    # Load from pickle if path is provided
    matrix = load_or_pass(matrix)

    # Unwrap if already a Matrix
    df = matrix.data.copy() if isinstance(matrix, Matrix) else matrix.copy()

    # Sanitize all indices
    df.index = sanitize_index(index=df.index, decimals=float_precision)
    df.columns = sanitize_index(index=df.columns, decimals=float_precision)
    row_index = sanitize_index(index=row_index, decimals=float_precision)
    col_index = sanitize_index(index=col_index, decimals=float_precision)

    # Handle block level promotion for both row and column indices
    for axis, (target_idx, name) in enumerate(
        [(row_index, "row"), (col_index, "column")]
    ):
        matrix_idx = df.index if axis == 0 else df.columns

        matrix_has_block = (
            isinstance(matrix_idx, pd.MultiIndex) and "block" in matrix_idx.names
        )
        target_has_block = (
            isinstance(target_idx, pd.MultiIndex) and "block" in target_idx.names
        )

        if not matrix_has_block and target_has_block:
            # Matrix doesn't have block level, but target does
            blocks = target_idx.get_level_values("block").unique()

            if len(blocks) > 1:
                raise ValueError(
                    f"Cannot automatically assign block to matrix: {name} index has multiple blocks {list(blocks)}. "
                    "Please provide a matrix with explicit block levels."
                )

            # Only one block, so promote the matrix to use this block
            promoted_idx = promote_index(
                index=matrix_idx, promotion=blocks[0], promotion_level="block"
            )
            if axis == 0:
                df.index = promoted_idx
            else:
                df.columns = promoted_idx
            logger.debug(f"Promoted {name} index with block '{blocks[0]}'")

        elif matrix_has_block and target_has_block:
            # Both have blocks - ensure they're compatible
            matrix_blocks = matrix_idx.get_level_values("block").unique()
            target_blocks = target_idx.get_level_values("block").unique()
            invalid_blocks = set(matrix_blocks) - set(target_blocks)

            if invalid_blocks:
                raise ValueError(
                    f"Matrix {name} blocks {invalid_blocks} not found in target {name} index blocks {list(target_blocks)}"
                )

    # Reorder MultiIndex levels to match target before checking overlap
    if isinstance(df.index, pd.MultiIndex) and isinstance(row_index, pd.MultiIndex):
        if (
            set(df.index.names) == set(row_index.names)
            and df.index.names != row_index.names
        ):
            df.index = df.index.reorder_levels(row_index.names)
            logger.debug(f"Reordered row index levels to {row_index.names}")

    if isinstance(df.columns, pd.MultiIndex) and isinstance(col_index, pd.MultiIndex):
        if (
            set(df.columns.names) == set(col_index.names)
            and df.columns.names != col_index.names
        ):
            df.columns = df.columns.reorder_levels(col_index.names)
            logger.debug(f"Reordered column index levels to {col_index.names}")

    # Check for overlapping indices
    check_overlap(target_idx=row_index, available_idx=df.index, name="Row")
    check_overlap(target_idx=col_index, available_idx=df.columns, name="Column")

    # Reindex to target indices, filling missing with zeros
    df = df.reindex(index=row_index, columns=col_index).fillna(0.0)
    logger.debug(
        f"Reindexed matrix to target indices (rows={len(row_index)}, cols={len(col_index)})"
    )

    # Wrap in appropriate matrix class and return
    return matrix_class(df)

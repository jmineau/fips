"""
Vector data structures for inverse problems.

This module provides the `Vector` and `Block` classes for representing
state vectors, observations, and other 1D data structures with automatic
index management and serialization support. Both are deprecated in favor of
:class:`~fips.Field`, which they are built with: a `Vector` assembles its
blocks as a field and selects a block through its :class:`~fips.Space`.
"""

import logging
from collections.abc import Sequence
from typing import Any, TypeAlias

import pandas as pd
import xarray as xr

from fips.base import (
    ArrayLike,
    MultiBlockMixin,
    SingleBlockMixin,
    Structure1D,
)
from fips.field import Field
from fips.space import Space

logger = logging.getLogger(__name__)

# Type Aliases
BlockLike: TypeAlias = "ArrayLike | Block"
VectorLike: TypeAlias = "BlockLike | Sequence[pd.Series | Block] | Vector"


class Block(SingleBlockMixin, Structure1D):
    """
    Single data block with a named Series and consistent index.

    A Block wraps a pandas Series and can be initialized
    from an existing Series or from raw values.

    Blocks are the fundamental building units of `fips`.
    Inverse problems can be customized by creating Blocks
    with specific indices and names to represent different state or
    observation components relevant to the application.

    .. deprecated:: 0.2.0b1
       `Block` will be removed. Build a :class:`~fips.Field` instead:
       ``Field.from_series(series, block="flux")``. A `Block` does not warn
       yet, because fips itself still builds blocks for an
       :class:`~fips.InverseProblem`.

    The labels are kept as given: they are not converted to numbers, and
    duplicate or missing labels raise a ``ValueError`` (the checks of
    :class:`~fips.Space`).

    Parameters
    ----------
    data : pd.Series, Block, or array-like
        Data for the block. If Block, creates a copy.
    name : str, optional
        Name for the block. If None, uses data.name.
    index : pd.Index, optional
        Index for the block. If None, uses data.index.
    dtype : dtype, optional
        Data type to force.
    copy : bool, default False
        Whether to copy the underlying data.
    """

    data: pd.Series  # type: ignore[override]
    name: str  # type: ignore[override]

    def __init__(
        self,
        data: BlockLike,
        name: str | None = None,
        index: pd.Index | None = None,
        dtype: Any = None,
        copy: bool = False,
    ):
        # Accept Block - extract Series
        if isinstance(data, Block):
            if name is None:
                name = data.name
            data = data.data
            index = data.index  # Use the index from the original Block
            copy = True  # Always copy when passed a Block

        super().__init__(data, name=name, index=index, dtype=dtype, copy=copy)

    def _validate(self):
        """Raise unless the underlying Series has a name."""
        super()._validate()

        if self.data.name is None:
            raise ValueError("Series must have a name or 'name' must be provided.")

        if "block" in self.data.index.names:
            raise ValueError("""Block data should not have a 'block' level in the index.
                             The Series name should be used for the block name instead.""")

        # Duplicate, missing and unnamed labels: the checks of a Space's block
        Space({self.name: self.index})

    def __repr__(self):
        """Return string representation."""
        header = f"Block(name='{self.name}')\n"
        return header + repr(self.data)

    def to_series(self, add_block_level=False) -> pd.Series:
        """
        Return the underlying Series data.

        Parameters
        ----------
        add_block_level : bool, default False
            Whether to add a 'block' level to the index with the block name.

        Returns
        -------
        pd.Series
            The underlying Series data, optionally with a 'block' level added to the index.
        """
        if add_block_level:
            # The block name moves from the Series name to the first index level.
            s = pd.concat({self.name: self.data}, names=["block"]).rename(None)
        else:
            s = self.data.copy()
        return s

    def to_xarray(self) -> xr.DataArray:
        """Convert the Block to an xarray DataArray."""
        return self.to_series().to_xarray()


class _BlockAccessor:
    """Accessor for retrieving Block instances from a Vector."""

    def __init__(self, vector: "Vector"):
        self._vector = vector

    def __getitem__(self, block_name: str) -> Block:
        return Block(self._vector[block_name], name=block_name)


class Vector(MultiBlockMixin, Structure1D):
    """
    State or observation vector composed of one or more Block objects.

    A Vector organizes one or more Blocks (prior, posterior, observations, etc.)
    into a single hierarchical structure.

    Vectors are used to represent the full state or observation space and are the
    1D matrix components in the inversion framework.

    .. deprecated:: 0.2.0b1
       `Vector` will be removed. Build a :class:`~fips.Field` instead:
       ``Field.from_blocks({"flux": flux, "bias": bias})``, or
       ``Field.from_series(series)`` for a Series with a ``block`` level. A
       `Vector` does not warn yet, because fips itself still builds vectors
       for an :class:`~fips.InverseProblem` and its posterior.

    Blocks are assembled as a :class:`~fips.Field`: the ``block`` level
    comes first, and a level one block has and another lacks is missing
    (NaN) on the other's rows (:attr:`Space.index <fips.Space.index>`). The
    labels are kept as given: they are not converted to numbers, and a block
    with duplicate labels, or with labels on a level for only some of its
    rows, raises a ``ValueError`` (:meth:`Space.from_index
    <fips.Space.from_index>`).

    Parameters
    ----------
    data : pd.Series, Vector, Block, Sequence[Block | pd.Series], or array-like
        Data for the vector.
    name : str, optional
        Name for the Vector.
    index : pd.Index, optional
        Index for the Vector. If None, uses data.index.
        Index must have a 'block' level if data is a Series.
    dtype : dtype, optional
        Data type to force.
    copy : bool, default False
        Whether to copy the underlying data.
    """

    data: pd.Series  # type: ignore[override]
    """The underlying data, which must be numeric and non-NaN. Can be stored in sparse format."""

    def __init__(
        self,
        data: VectorLike,
        name: str | None = None,
        index: pd.Index | None = None,
        dtype: Any = None,
        copy: bool = False,
    ):
        # There are two main paths to build a Vector:
        # 1) From a single Series or Block
        # 2) From a sequence of Blocks or Series (which are combined into one Series with a 'block' level)
        blocks = None

        # Accept Vector - extract Series
        if isinstance(data, Vector):
            data = data.data
            index = data.index  # Use the index from the original Vector
            copy = True  # Always copy when passed a Vector

        # Pre-built Series with block level - use directly
        elif isinstance(data, pd.Series) and "block" in data.index.names:
            index = data.index  # Use the index from the Series

        # Single Block
        elif isinstance(data, (Block, pd.Series)):
            blocks = [data]

        # Sequence - check if it's blocks or numeric data
        elif isinstance(data, Sequence) and not isinstance(data, str):
            # Check if sequence contains Block/Series objects
            if any(isinstance(item, (Block, pd.Series)) for item in data):
                blocks = data

        if blocks:  # Build from blocks, as a Field
            named: dict[str, pd.Series] = {}
            for b in blocks:
                block = Block(b)
                if block.name in named:
                    raise ValueError(f"Duplicate block name '{block.name}' found")
                named[block.name] = block.data
            data = Field.from_blocks(named, name=name).to_series()
        else:
            # If we get here, data should be array-like and we can let the parent class handle it
            if name is None:
                name = getattr(data, "name", None)

        super().__init__(data, name=name, index=index, dtype=dtype, copy=copy)

    def _validate(self):
        """Raise unless the labels make a Space: unique, and no level missing for only some of a block's rows."""
        super()._validate()
        Space.from_index(self.index)

    def __repr__(self) -> str:
        """Return string representation."""
        header = f"Vector(name='{self.name}', shape={self.shape})\n"
        return header + repr(self.data)

    def __getitem__(self, block) -> pd.Series:
        """
        Get a block by name, labeled as the block was given.

        The ``block`` level and the levels only other blocks have are
        dropped, and each level keeps the dtype the block was built with
        (:meth:`Space.select <fips.Space.select>`).

        Raises
        ------
        KeyError
            If the vector has no block of that name.
        """
        labels = Space.from_index(self.index).select(str(block))
        rows = self.index.get_level_values("block").astype(str) == str(block)
        return self.data[rows].set_axis(labels)

    @property
    def blocks(self) -> _BlockAccessor:  # type: ignore[override]
        """Accessor for retrieving Block instances from the Vector."""
        return _BlockAccessor(self)

    def to_xarray(self) -> xr.DataArray | xr.Dataset:
        """
        Convert the Vector to xarray.

        If the Vector contains multiple blocks, returns an xarray Dataset with each block as a separate DataArray.
        If the Vector contains a single block, returns an xarray DataArray.
        """
        ds = xr.Dataset()

        blocks = self.index.get_level_values("block").unique()
        for block in blocks:
            # Extract subset for this block & convert to DataArray
            ds[block] = self[block].to_xarray()

        return ds if len(blocks) > 1 else ds[blocks[0]]

import logging
from collections.abc import Sequence
from typing import Any, TypeAlias

import pandas as pd
import xarray as xr

from fips.base import ArrayLike, MultiBlockMixin, SingleBlockMixin, Structure1D

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
    """

    data: pd.Series
    name: str

    def __init__(
        self,
        data: BlockLike,
        name: str | None = None,
        index: pd.Index | None = None,
        dtype: Any = None,
        copy: bool = False,
    ):
        """
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
        # Accept Block - extract Series
        if isinstance(data, Block):
            if name is None:
                name = data.name
            data = data.data
            index = data.index  # Use the index from the original Block
            copy = True  # Always copy when passed a Block

        super().__init__(data, name=name, index=index, dtype=dtype, copy=copy)

    def _validate(self):
        super()._validate()

        if self.data.name is None:
            raise ValueError("Series must have a name or 'name' must be provided.")

        if "block" in self.data.index.names:
            raise ValueError("""Block data should not have a 'block' level in the index.
                             The Series name should be used for the block name instead.""")

    def __repr__(self):
        header = f"Block(name='{self.name}')\n"
        return header + repr(self.data)

    def to_series(self, add_block_level=False) -> pd.Series:
        """Return the underlying Series data."""
        if add_block_level:
            dims = self.index.names
            df = self.data.to_frame()
            df.columns.rename("block", inplace=True)
            s = df.stack()
            s.index = s.index.reorder_levels(["block"] + dims)
        else:
            s = self.data.copy()
        return s

    def to_xarray(self) -> xr.DataArray:
        """Convert the Block to an xarray DataArray."""
        return self.to_series().to_xarray()


class Vector(MultiBlockMixin, Structure1D):
    """State or observation vector composed of one or more Block objects.

    A Vector organizes one or more Blocks (prior, posterior, observations, etc.)
    into a single hierarchical structure.

    Vectors are used to represent the full state or observation space and are the
    1D matrix components in the inversion framework.
    """

    data: pd.Series

    def __init__(
        self,
        data: VectorLike,
        name: str | None = None,
        index: pd.Index | None = None,
        dtype: Any = None,
        copy: bool = False,
    ):
        """
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

        if blocks:  # Build from blocks
            seen_blocks = set()
            dfs: list[pd.DataFrame] = []
            dims = {}

            for b in blocks:
                # Convert to Block
                block = Block(b)
                if block.name in seen_blocks:
                    raise ValueError(f"Duplicate block name '{block.name}' found")
                seen_blocks.add(block.name)

                # Collect unique dimensions in order
                for dim in block.data.index.names:
                    dims[dim] = True

                # Format for concatenation
                dfs.append(
                    block.to_series(add_block_level=True).rename(name).reset_index()
                )

            combined = pd.concat(dfs, ignore_index=True)

            # Handle Ragged Hierarchies (Fill integers with -1)
            # combined[list(dims.keys())] = combined[list(dims.keys())].fillna(-1)

            levels = ["block"] + list(dims.keys())
            data = combined.set_index(levels)[name or 0].rename(name)
        else:
            # If we get here, data should be array-like and we can let the parent class handle it
            if name is None:
                name = getattr(data, "name", None)

        super().__init__(data, name=name, index=index, dtype=dtype, copy=copy)

    def __repr__(self) -> str:
        header = f"Vector(name='{self.name}', shape={self.shape})\n"
        return header + repr(self.data)

    def __getitem__(self, block) -> pd.Series:
        return self.xs(block, level="block")

    def to_xarray(self) -> xr.DataArray | xr.Dataset:
        """Convert the Vector to an xarray DataArray or Dataset."""
        ds = xr.Dataset()

        blocks = self.index.get_level_values("block").unique()
        for block in blocks:
            # Extract subset for this block & convert to DataArray
            ds[block] = self[block].to_xarray()

        return ds if len(blocks) > 1 else ds[blocks[0]]

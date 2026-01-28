import warnings
from collections.abc import Sequence

import numpy.typing as npt
import pandas as pd

from fips.indices import sanitize_index


class Block:
    def __init__(
        self,
        data: pd.Series | npt.ArrayLike,
        index: pd.Index | None = None,
        name: str | None = None,
    ):
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
        new_names = [
            dim if dim is not None else f"{self.name}_{i}"
            for i, dim in enumerate(self.data.index.names)
        ]
        self.data.index.set_names(new_names, inplace=True)

    @property
    def name(self) -> str:
        return str(self.data.name)

    @property
    def index(self) -> pd.Index:
        return self.data.index

    @property
    def dims(self) -> list[str]:
        return list(self.data.index.names)

    @property
    def values(self) -> npt.NDArray:
        return self.data.to_numpy()

    def __repr__(self):
        header = f"Block(name='{self.name}')\n"
        return header + repr(self.data)


class Vector:
    # State or Obs vector composed of multiple Blocks
    # can be prior, posterior, obs, etc.
    def __init__(self, name, blocks: Sequence[Block | pd.Series]):
        """
        Parameters
        ----------
        blocks : Sequence[Block | pd.Series]
            A sequence containing either Block objects, pd.Series, or both.
            pd.Series entries are automatically converted to Blocks.
        """
        self.name = name
        self.blocks: dict[str | int, Block] = {}
        for block in blocks:
            block = block if isinstance(block, Block) else Block(block)
            if block.name in self.blocks:
                raise ValueError(f"Duplicate block name '{block.name}' found.")
            self.blocks[block.name] = block
        self._assemble()

    @classmethod
    def from_series(cls, data: pd.Series, name: str | None = None) -> "Vector":
        """
        Create a Vector from a Series with 'block' level in index.

        Parameters
        ----------
        data : pd.Series
            Series with MultiIndex containing 'block' level.
        name : str, optional
            Override the series name. If None, uses data.name.

        Returns
        -------
        Vector
            Vector constructed from the series blocks.
        """
        if not isinstance(data, pd.Series):
            raise TypeError("data must be a pd.Series.")

        if "block" not in data.index.names:
            raise ValueError("Series index must include a 'block' level.")

        # Use series name if no override
        vector_name = name if name is not None else data.name
        if vector_name is None:
            raise ValueError("Series must have a name or 'name' must be provided.")

        # Extract blocks from the series
        blocks = []
        block_names = data.index.get_level_values("block").unique()

        for block_name in block_names:
            block_data = data.xs(block_name, level="block")
            blocks.append(Block(block_data, name=block_name))

        return cls(name=vector_name, blocks=blocks)

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
    def index(self) -> pd.Index:
        return self.data.index

    @property
    def dims(self) -> list[str]:
        return [str(level) for level in self.index.names if level != "block"]

    @property
    def values(self) -> npt.NDArray:
        return self.data.to_numpy()

    @property
    def n(self) -> int:
        return len(self.data)

    def xs(self, *args, drop_block=False, **kwargs) -> pd.Series:
        """
        Access a cross-section of the vector data.

        Parameters
        ----------
        *args, **kwargs : passed to pd.Series.xs
        drop_block : bool, default False
            If True, drops the 'block' level
        """
        # Call the underlying Series xs method
        s = self.data.xs(*args, **kwargs).copy()

        # Drop index levels that are all nan
        s.index = s.index.droplevel(
            [
                level
                for level in s.index.names
                if s.index.get_level_values(level).isna().all()
            ]
        )

        if drop_block:
            raise NotImplementedError("drop_block=True is not yet implemented.")

        return s


def prepare_vector(
    name: str, vector: Vector | pd.Series, float_precision: int | None
) -> Vector:
    """Helper to normalize input vectors into Vector objects and sanitize indices."""
    if isinstance(vector, pd.Series):
        v_clean = vector.copy()
        v_clean.index = sanitize_index(v_clean.index, float_precision)
        if "block" in v_clean.index.names:
            vector_obj = Vector.from_series(v_clean, name=name)
        else:
            if v_clean.name is None:
                v_clean.name = f"{name}_block"  # name of the series is the block name
            vector_obj = Vector(name=name, blocks=[Block(v_clean)])
    elif isinstance(vector, Vector):
        vector_obj = vector
        vector_obj.data.index = sanitize_index(vector_obj.data.index, float_precision)
    else:
        raise TypeError(f"Input must be a pd.Series or Vector. Got {type(vector)}")
    return vector_obj

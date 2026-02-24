import logging
import pickle
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeAlias

import numpy.typing as npt
import pandas as pd
from typing_extensions import Self

from fips.indexes import overlaps, resolve_axes, round_index, to_numeric

logger = logging.getLogger(__name__)

ArrayLike: TypeAlias = npt.ArrayLike | pd.Series | pd.DataFrame


def xselect(
    s_or_df: pd.DataFrame | pd.Series, key, axis=0, level=None, drop_level=True
) -> pd.DataFrame | pd.Series:
    for ax in resolve_axes(axis):
        # Call the underlying xs method
        s_or_df = s_or_df.xs(key, axis=ax, level=level, drop_level=drop_level).copy()

        # Drop index levels that are all nan
        idx = s_or_df.axes[ax]
        idx = idx.droplevel(
            [level for level in idx.names if idx.get_level_values(level).isna().all()]
        )

        # Reassign cleaned index/columns
        s_or_df = s_or_df.set_axis(idx, axis=ax)
    return s_or_df


class Pickleable:
    """Mixin to add to_file() and from_file() methods for pickle serialization.

    Provides automatic pickle file I/O with extension validation (.pkl or .pickle).
    """

    VALID_EXTENSIONS = {".pkl", ".pickle"}

    def to_file(self, path: str | Path) -> None:
        """Save object to a pickle file.

        Parameters
        ----------
        path : str or Path
            File path where object will be saved.
        """
        path = Path(path)
        logger.debug(f"Serializing {type(self).__name__} to {path}")
        if path.suffix not in self.VALID_EXTENSIONS:
            raise ValueError(
                f"File extension must be one of {self.VALID_EXTENSIONS}, got {path.suffix}"
            )
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_file(cls, path: str | Path):
        """Load object from a pickle file.

        Parameters
        ----------
        path : str or Path
            File path to load from.

        Returns
        -------
        object
            The unpickled object.
        """
        path = Path(path)
        logger.debug(f"Deserializing {cls.__name__} from {path}")
        if path.suffix not in cls.VALID_EXTENSIONS:
            raise ValueError(
                f"File extension must be one of {cls.VALID_EXTENSIONS}, got {path.suffix}"
            )
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    @abstractmethod
    def __getstate__(self) -> dict[str, Any]:
        """Return state as dict for pickling."""
        pass

    @abstractmethod
    def __setstate__(self, state: dict[str, Any]):
        """Restore state from dict for unpickling."""
        pass


class Structure(Pickleable, ABC):
    name: str | property | None
    data: pd.DataFrame | pd.Series

    def _validate(self):
        # Ensure index levels are named
        if None in self.index.names:
            raise ValueError(
                f"All levels in the row index of {self.__class__.__name__} must be named."
            )
        if hasattr(self.data, "columns") and None in self.data.columns.names:
            raise ValueError(
                f"All levels in the columns of {self.__class__.__name__} must be named."
            )

        # Check for NaN values
        if self.data.isnull().any(axis=None):
            raise ValueError("Data contains NaN values.")

    def _sanitize(self):
        # Force numeric indices where possible
        self.data.index = to_numeric(self.data.index)
        if isinstance(self.data, pd.DataFrame):
            self.data.columns = to_numeric(self.data.columns)

    @property
    def index(self) -> pd.Index:
        return self.data.index

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def values(self) -> npt.NDArray:
        return self.to_numpy()

    def reindex(
        self,
        index: pd.Index,
        columns: pd.Index | None = None,
        verify_overlap: bool = False,
        inplace: bool = False,
        **kwargs,
    ) -> Self:
        """Return a new instance with data reindexed to the specified index and columns."""

        def prepare_data_for_target(
            data: pd.DataFrame | pd.Series, target: pd.Index, axis: int
        ) -> pd.DataFrame | pd.Series:
            current = data.axes[axis]

            if (
                isinstance(target, pd.MultiIndex)
                and not current.nlevels == target.nlevels
            ):
                raise ValueError(
                    f"Cannot reindex: target index has {target.nlevels} levels but data has {current.nlevels} levels."
                )
            if not set(current.names) == set(target.names):
                raise ValueError(
                    f"Cannot reindex: target index names {target.names} do not match data index names {current.names}."
                )
            if current.names != target.names:  # Reorder levels to match target index
                data = data.reorder_levels(target.names, axis=axis)
                current = data.axes[axis]
            if verify_overlap:
                overlap = overlaps(target_idx=target, available_idx=current)
                if not overlap:
                    raise ValueError(
                        f"Target index {target} does not overlap with available index {current}"
                    )
                elif overlap == "partial":
                    warnings.warn(
                        f"Partial overlap between target index {target} and available index {current}. Missing entries will be filled with zeros.",
                        UserWarning,
                        stacklevel=2,
                    )

            return data

        data = prepare_data_for_target(self.data, index, axis=0)

        if columns is not None:
            data = prepare_data_for_target(data, columns, axis=1)
            data = data.reindex(index=index, columns=columns, **kwargs)
        else:
            data = data.reindex(index=index, **kwargs)

        if inplace:
            self.data = data
            return self

        return type(self)(data, name=self.name)  # create new instance

    def round_index(self, decimals: int, axis=0, inplace: bool = False) -> Self:
        """
        Round float indices on the specified axis to given decimals. Non-float indices are left unchanged.
        """
        data = self.data if inplace else self.data.copy(deep=True)

        for ax in resolve_axes(axis):
            if ax == 1 and isinstance(data, pd.Series):
                raise ValueError("Cannot round columns of a Series.")

            new_idx = round_index(data.axes[ax], decimals=decimals)
            data = data.set_axis(new_idx, axis=ax)

        if inplace:
            self.data = data
            return self

        return type(self)(data, name=self.name)

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
        return type(self)(self.data.copy(deep=deep), name=self.name)

    def xs(self, key, axis=0, level=None, drop_level=True):
        """
        Cross-select data based on index/column values.

        Parameters
        ----------
        key : label or list of labels
            The key(s) to select.
        axis : {0, 1, 'index', 'columns', 'both'}, default 0
            Axis to retrieve cross-section from.
        level : int or str, optional
            If the index is a MultiIndex, level(s) to select on.
        drop_level : bool, default True
            Whether to drop the level(s) from the resulting index.

        Returns
        -------
        Series or DataFrame
            The cross-section of the data corresponding to the key(s).
        """
        return xselect(
            self.data, key=key, axis=axis, level=level, drop_level=drop_level
        )

    def to_numpy(self) -> npt.NDArray:
        """Get the underlying data as a NumPy array."""
        return self.data.to_numpy()


class Structure1D(Structure):
    """Base class for 1D structures (Block, Vector)."""

    data: pd.Series

    def __init__(
        self,
        data: ArrayLike,
        name: str | None = None,
        index: pd.Index | None = None,
        dtype: Any = None,
        copy: bool | None = None,
    ):
        # Squeeze to 1D if necessary
        if hasattr(data, "squeeze"):
            data = data.squeeze()

        self.data = pd.Series(data, name=name, index=index, dtype=dtype, copy=copy)

        self._validate()
        self._sanitize()

    def _sanitize(self):
        super()._sanitize()

        # 1D Structures must be all numeric, non nan
        self.data = pd.to_numeric(self.data, errors="raise")

    @property
    def name(self) -> str | None:
        return str(self.data.name)

    def to_series(self) -> pd.Series:
        """Get the underlying Series data."""
        return self.data

    def __getstate__(self):
        """Explicit pickle support: return state as dict."""
        return {"data": self.data}

    def __setstate__(self, state):
        """Explicit pickle support: restore state from dict."""
        self.data = state["data"]


class Structure2D(Structure):
    """Base class for 2D structures (Matrix)."""

    data: pd.DataFrame

    def __init__(
        self,
        data: ArrayLike,
        name: str | None = None,
        index: pd.Index | None = None,
        columns: pd.Index | None = None,
        dtype=None,
        copy=None,
        sparse: bool = False,
    ):
        """
        Initialize a 2D structure with the given data, index, and columns. Validates and sanitizes the data.

        Parameters
        ----------
        data : np.ndarray
            2D data.
        name : str, optional
            Name for the Structure.
        index : pd.Index
            Index for the rows of the DataFrame.
        columns : pd.Index, optional
            Index for the columns of the DataFrame. If None, uses the same as `index`.
        dtype : data type, optional
            Data type to force.
        copy : bool, optional
            Whether to copy the data.
        sparse : bool, default False
            If True, store the data in pandas sparse format (fill_value=0.0).
            The caller is responsible for zeroing out near-zero values before
            setting sparse=True (e.g. via a threshold in the builder).
        """
        self.name = name

        # Detect if the input data is already a sparse DataFrame so we can
        # preserve sparsity when constructing from an existing sparse structure
        # (e.g. inside reindex() which calls type(self)(data, name=self.name)).
        _input_is_sparse = isinstance(data, pd.DataFrame) and any(
            isinstance(dt, pd.SparseDtype) for dt in data.dtypes
        )

        # If columns not provided, assume symmetric matrix with same index for rows and columns
        if columns is None:
            logger.debug(
                "No columns provided, assuming symmetric matrix with same index for rows and columns."
            )
            columns = index

        self.data = pd.DataFrame(
            data, index=index, columns=columns, dtype=dtype, copy=copy
        )

        self._validate()
        self._sanitize()

        # Sparsify after sanitize (index may have been modified by _sanitize)
        if (sparse or _input_is_sparse) and not self.is_sparse:
            self._sparsify()

    def _sparsify(self) -> None:
        """Convert self.data to pandas sparse format in-place.

        The caller is responsible for ensuring near-zero floating-point noise
        has already been zeroed out (via a threshold) before calling this.
        """
        self.data = self.data.astype(pd.SparseDtype(float, fill_value=0.0))

    @property
    def columns(self) -> pd.Index:
        return self.data.columns

    @property
    def values(self) -> npt.NDArray:
        if self.is_sparse:
            return self.data.sparse.to_coo().tocsr()
        return self.data.to_numpy()

    @property
    def is_sparse(self) -> bool:
        """True if the internal DataFrame uses pandas sparse storage."""
        return all(isinstance(dt, pd.SparseDtype) for dt in self.data.dtypes)

    def to_frame(self) -> pd.DataFrame:
        """
        Get the underlying DataFrame data.

        Returns
        -------
        pd.DataFrame
            The DataFrame representation of the matrix.
        """
        return self.data

    def to_dense(self) -> "Structure2D":
        """Return a copy with dense internal storage.

        Returns
        -------
        Structure2D
            New instance backed by a regular dense DataFrame.
        """
        if not self.is_sparse:
            return self
        return type(self)(self.data.sparse.to_dense(), name=self.name)

    def to_sparse(self, threshold: float | None = None) -> "Structure2D":
        """Return a copy with sparse internal storage.

        Parameters
        ----------
        threshold : float | None, optional
            If provided, values whose absolute value is strictly less than
            this threshold are zeroed out before sparsification, avoiding
            storage of floating-point noise as explicit non-zero entries.
            Default is None (no zeroing applied).

        Returns
        -------
        Structure2D
            New instance backed by a sparse DataFrame.
        """
        if self.is_sparse:
            return self
        result = self.copy()
        if threshold is not None:
            result.data = result.data.where(result.data.abs() >= threshold, other=0.0)
        result._sparsify()
        return result

    def __getstate__(self):
        """Explicit pickle support: return state as dict."""
        return {"data": self.data, "name": self.name}

    def __setstate__(self, state):
        """Explicit pickle support: restore state from dict."""
        self.data = state["data"]
        self.name = state["name"]


class SingleBlockMixin(ABC):
    """Base class for low-level structures (Block, MatrixBlock) that wrap raw Series/DataFrames without block levels."""

    data: pd.DataFrame | pd.Series
    name: str | property

    def _validate(self):
        super()._validate()

        # Check for name property
        if self.name is None:
            raise ValueError(
                "Low-level structures (Block, MatrixBlock) must have a name property."
            )

        # Check for absence of 'block' level in index and columns
        if "block" in self.data.index.names:
            raise ValueError("Data should not have a 'block' level in the index.")
        if isinstance(self.data, pd.DataFrame) and "block" in self.data.columns.names:
            raise ValueError(
                "DataFrame should not have a 'block' level in the columns."
            )


class MultiBlockMixin(ABC):
    """Base class for high-level structures (Vector, Matrix) that wrap Series/DataFrames with block levels."""

    data: pd.DataFrame | pd.Series

    @property
    @abstractmethod
    def blocks(self):
        """Accessor for retrieving typed block instances (Block or MatrixBlock) by name."""
        ...

    def _validate(self):
        super()._validate()

        # Check for 'block' level in index for Series, columns for DataFrame
        if "block" not in self.data.index.names:
            raise ValueError("Data must have a 'block' level in the index.")
        if (
            isinstance(self.data, pd.DataFrame)
            and "block" not in self.data.columns.names
        ):
            raise ValueError("DataFrame must have a 'block' level in the columns.")

    def _sanitize(self):
        super()._sanitize()

        def block_as_first_level(index):
            if index.names[0] != "block":
                return index.reorder_levels(
                    ["block"] + [n for n in index.names if n != "block"]
                )
            return index

        self.data.index = block_as_first_level(self.data.index)
        if isinstance(self.data, pd.DataFrame):
            self.data.columns = block_as_first_level(self.data.columns)

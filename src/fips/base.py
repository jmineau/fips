"""
Base classes of the 1D structures, :class:`~fips.Block` and :class:`~fips.Vector`.

This module provides the serialization mixin and the classes the two
deprecated 1D structures are built on. The 2D structures are gone: a matrix
is an :class:`~fips.Operator`.
"""

import logging
import pickle
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Self, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd

from fips.indexes import overlaps, round_index

logger = logging.getLogger(__name__)

ArrayLike: TypeAlias = npt.ArrayLike | pd.Series | pd.DataFrame


def _xselect(series: pd.Series, key, level=None, drop_level=True) -> pd.Series:
    """Return the cross-section of `series` at `key`, without the index levels that are missing on every row."""
    section = series.xs(key, level=level, drop_level=drop_level).copy()
    index = section.index
    empty = [
        i
        for i in range(index.nlevels)
        if np.asarray(index.get_level_values(i).isna()).all()
    ]
    return section.set_axis(index.droplevel(empty)) if empty else section


class Pickleable:
    """
    Mixin to add to_file() and from_file() methods for pickle serialization.

    Provides automatic pickle file I/O with extension validation (.pkl or .pickle).
    """

    VALID_EXTENSIONS = {".pkl", ".pickle"}

    def to_file(self, path: str | Path) -> None:
        """
        Save object to a pickle file.

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
        """
        Load object from a pickle file.

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


def _check_reindex_levels(current: pd.Index, target: pd.Index) -> None:
    """Raise unless `target` has the levels of `current`, in any order."""
    if isinstance(target, pd.MultiIndex) and not current.nlevels == target.nlevels:
        raise ValueError(
            f"Cannot reindex: target index has {target.nlevels} levels but data has {current.nlevels} levels."
        )
    if not set(current.names) == set(target.names):
        raise ValueError(
            f"Cannot reindex: target index names {target.names} do not match data index names {current.names}."
        )


def _verify_overlap(target: pd.Index, current: pd.Index) -> None:
    """Raise if `target` shares no label with `current`; warn if it misses some."""
    overlap = overlaps(target_idx=target, available_idx=current)
    if not overlap:
        raise ValueError(
            f"Target index {target} does not overlap with available index {current}"
        )
    elif overlap == "partial":
        warnings.warn(
            f"Partial overlap between target index {target} and available index {current}. Missing entries will be filled with zeros.",
            UserWarning,
            stacklevel=3,
        )


class Structure(Pickleable, ABC):
    """
    Abstract base class of the 1D structures: a pandas Series with named levels and no NaN.
    """

    name: str | None
    """Name of the structure, used for error messages and block naming."""
    data: pd.Series
    """The underlying data, which must be numeric and non-NaN."""

    def _validate(self):
        """Raise when an index level is unnamed or a value is NaN."""
        if None in self.index.names:
            raise ValueError(
                f"All levels in the index of {self.__class__.__name__} must be named."
            )
        if self.data.isnull().any():
            raise ValueError("Data contains NaN values.")

    def _sanitize(self):
        """Normalize the data after validation; the labels are kept as given."""

    def _set_labels(self, index: pd.Index) -> None:
        """Replace the labels, keeping the data."""
        self.data.index = index

    @property
    def index(self) -> pd.Index:
        """Return the index of the underlying data."""
        return self.data.index

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the underlying data."""
        return self.data.shape

    def __len__(self) -> int:
        """Return the length of the underlying data."""
        return len(self.data)

    @property
    def values(self) -> npt.NDArray:
        """Return the values as a numpy array."""
        return self.to_numpy()

    def _with_data(self, data: pd.Series) -> Self:
        """
        Return a new instance of this class that holds `data`.

        The new instance keeps this one's name, so that every method that
        rebuilds the structure keeps it.
        """
        return type(self)(data, name=self.name)  # type: ignore[call-arg]

    def reindex(
        self,
        index: pd.Index,
        columns: pd.Index | None = None,
        verify_overlap: bool = False,
        inplace: bool = False,
        **kwargs,
    ) -> Self:
        """
        Return a new instance with data reindexed to the specified index.

        Parameters
        ----------
        index : pd.Index
            New index. Its levels must be the structure's, in any order.
        columns : None
            A 1D structure has no columns: anything but None raises.
        verify_overlap : bool, default False
            If True, raise if no label of `index` is in the structure, and warn
            if some are not.
        inplace : bool, default False
            If True, modify the data in place and return self. Otherwise, return a new instance with the reindexed data.
        **kwargs
            Additional keyword arguments to pass to pandas reindex() method (e.g. fill_value, method, limit).

        Returns
        -------
        Structure
            A new instance with the reindexed data, or self if inplace=True.

        Raises
        ------
        ValueError
            If `columns` is given, or the levels of `index` are not the
            structure's.
        """
        if columns is not None:
            raise ValueError(f"{type(self).__name__} is 1D: it has no columns.")
        data = self.data
        _check_reindex_levels(data.index, index)
        if data.index.names != index.names:  # Reorder levels to match target index
            data = data.reorder_levels(list(index.names))
        if verify_overlap:
            _verify_overlap(index, data.index)
        data = data.reindex(index=index, **kwargs)

        if inplace:
            self.data = data
            return self
        return self._with_data(data)

    def round_index(
        self, decimals: int, axis: int | Literal["both"] = 0, inplace: bool = False
    ) -> Self:
        """
        Round float labels to given decimals. Non-float labels are left unchanged.

        Parameters
        ----------
        decimals : int
            Number of decimal places to round to.
        axis : {0, 'index'}, default 0
            A 1D structure has only its index: any other axis raises.
        inplace : bool, default False
            If True, round indices in place and return self. Otherwise, return a new instance with rounded indices.

        Returns
        -------
        Structure
            A new instance with rounded indices, or self if inplace=True.
        """
        if axis not in (0, "index"):
            raise ValueError("Cannot round columns of a Series.")
        data = self.data if inplace else self.data.copy(deep=True)
        data = data.set_axis(round_index(data.index, decimals=decimals))

        if inplace:
            self.data = data
            return self
        return self._with_data(data)

    def copy(self, deep: bool = True) -> Self:
        """
        Create a copy of the structure.

        Parameters
        ----------
        deep : bool, default True
            If True, perform a deep copy of the underlying data.

        Returns
        -------
        Structure
            Copy of the structure instance.
        """
        return self._with_data(self.data.copy(deep=deep))

    def xs(self, key, axis=0, level=None, drop_level=True) -> pd.Series:
        """
        Cross-select data based on index values.

        Parameters
        ----------
        key : label or list of labels
            The key(s) to select.
        axis : {0, 'index'}, default 0
            A 1D structure has only its index: any other axis raises.
        level : int or str, optional
            If the index is a MultiIndex, level(s) to select on.
        drop_level : bool, default True
            Whether to drop the level(s) from the resulting index.

        Returns
        -------
        pd.Series
            The cross-section, without the levels that are missing on every
            one of its rows.
        """
        if axis not in (0, "index"):
            raise ValueError(f"{type(self).__name__} is 1D: it has no columns.")
        return _xselect(self.data, key, level=level, drop_level=drop_level)

    def to_numpy(self) -> npt.NDArray:
        """Get the underlying data as a NumPy array."""
        return self.data.to_numpy()


class Structure1D(Structure):
    """
    Base class for 1D structures (Block, Vector).

    Parameters
    ----------
    data : array-like
        1D data for the structure.
    name : str, optional
        Name for the structure. If None, uses data.name if available.
        Non-None names are converted to strings during construction.
    index : pd.Index, optional
        Index for the data. If None, uses data.index if available.
    dtype : data type, optional
        Data type to force.
    copy : bool, optional
        Whether to copy the data.
    """

    data: pd.Series  # type: ignore[override]
    """The underlying data, which must be numeric and non-NaN."""

    def __init__(
        self,
        data: ArrayLike,
        name: str | None = None,
        index: pd.Index | None = None,
        dtype: Any = None,
        copy: bool | None = None,
    ):
        """
        Initialize a 1D structure with the given data and index. Validates and sanitizes the data.
        """
        # Reduce 2D input to 1D. squeeze() drops every length-1 axis, so on a
        # one-element Series or a one-row, one-column DataFrame it returns a
        # scalar and the index is lost: leave 1D input alone and take a single
        # column as it is.
        if isinstance(data, pd.DataFrame) and data.shape[1] == 1:
            data = data.iloc[:, 0]
        elif getattr(data, "ndim", 1) > 1:
            data = data.squeeze()
            if getattr(data, "ndim", 0) > 1:
                raise ValueError(
                    f"{type(self).__name__} needs 1-dimensional data, "
                    f"got shape {np.shape(data)}"
                )

        self.data = pd.Series(data, name=name, index=index, dtype=dtype, copy=copy)
        if self.data.name is not None:
            self.data.name = str(self.data.name)

        self._validate()
        self._sanitize()

    def _sanitize(self):
        """Coerce the 1-D values to numeric, raising on failure."""
        super()._sanitize()

        # 1D Structures must be all numeric, non nan
        self.data = pd.to_numeric(self.data, errors="raise")

    @property
    def name(self) -> str | None:
        """The stored Series name, a string or None if the Series is unnamed."""
        name = self.data.name
        return None if name is None else str(name)

    def to_series(self) -> pd.Series:
        """Get the underlying Series data."""
        return self.data

    def __getstate__(self):
        """Explicit pickle support: return state as dict."""
        return {"data": self.data}

    def __setstate__(self, state):
        """Explicit pickle support: restore state from dict."""
        self.data = state["data"]


class SingleBlockMixin(ABC):
    """Base class for a structure of one block (Block) that wraps a Series without a block level."""

    data: pd.Series
    name: str | property
    index: pd.Index

    def _validate(self):
        """Raise unless the single-block structure has a name."""
        super()._validate()  # type: ignore[attr-defined]

        # Check for name property
        if self.name is None:
            raise ValueError("A Block must have a name.")
        if "block" in self.index.names:
            raise ValueError("Data should not have a 'block' level in the index.")


class MultiBlockMixin(ABC):
    """Base class for a structure of several blocks (Vector) that wraps a Series with a block level."""

    data: pd.Series
    index: pd.MultiIndex

    @property
    @abstractmethod
    def blocks(self):
        """Accessor for retrieving typed block instances (Block) by name."""
        ...

    def _validate(self):
        """Raise unless the data carries a 'block' index level."""
        super()._validate()  # type: ignore[attr-defined]

        if "block" not in self.index.names:
            raise ValueError("Data must have a 'block' level in the index.")

    def _sanitize(self):
        """Coerce values and move the 'block' level first in the index."""
        super()._sanitize()  # type: ignore[attr-defined]

        def block_as_first_level(index):
            """Return ``index`` with its 'block' level first."""
            if index.names[0] != "block":
                return index.reorder_levels(
                    ["block"] + [n for n in index.names if n != "block"]
                )
            return index

        set_labels: Callable[..., None] = self._set_labels  # type: ignore[attr-defined]
        set_labels(block_as_first_level(self.index))

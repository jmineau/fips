"""
Vectors: values on a space.

A :class:`Vector` holds one value for each label of a :class:`~fips.Space`:
a prior state, the observations, an offset, a posterior estimate or its
variances. The values are a NumPy array and the labels are the space's; a
:class:`pandas.Series` is what :meth:`Vector.to_series` gives.
"""

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from fips.space import Space, _listed


class Vector:
    """
    Values on a :class:`~fips.Space`, one for each label.

    A vector is read-only: its values cannot be changed in place, and every
    method returns a new vector. Labels are kept as given: they are not
    converted to numbers and not rounded unless :meth:`round_labels` is
    called. A vector of one block is what a single state or observation
    component is (``Vector.from_series(series, block="flux")``); several
    blocks are put together with :meth:`from_blocks`.

    Parameters
    ----------
    values : array-like
        One number for each label of `space`, in the space's order. A
        :class:`pandas.Series` is refused here, since its own labels would be
        ignored: use :meth:`from_series`.
    space : Space
        The labels.
    name : str, optional
        What the values are ("prior", "obs", ...). A name that is not a
        string is converted to one.

    Raises
    ------
    TypeError
        If `values` is a pandas object.
    ValueError
        If `values` is not one-dimensional, its length differs from the
        space's, or a value is NaN (the message names its labels).

    See Also
    --------
    Vector.from_series : A vector from a Series, with or without a ``block`` level.
    Vector.from_blocks : A vector from one Series for each block.

    Examples
    --------
    >>> import pandas as pd
    >>> from fips import Vector
    >>> prior = Vector.from_blocks(
    ...     {
    ...         "flux": pd.Series([1.0, 2.0], index=pd.Index([1, 2], name="cell")),
    ...         "bias": pd.Series([0.0], index=pd.Index(["A"], name="site")),
    ...     },
    ...     name="prior",
    ... )
    >>> prior.values
    array([1., 2., 0.])
    >>> prior["flux"].values
    array([1., 2.])
    >>> prior["flux"].space.select("flux")
    Index([1, 2], dtype='int64', name='cell')
    """

    def __init__(self, values: npt.ArrayLike, space: Space, name: str | None = None):
        if isinstance(values, (pd.Series, pd.DataFrame, pd.Index)):
            raise TypeError(
                "Vector takes the values as an array, in the space's order. "
                "Build a vector from a Series with Vector.from_series, which "
                "reads its labels."
            )
        if not isinstance(space, Space):
            raise TypeError(f"space must be a Space, got {type(space).__name__}.")
        array = np.array(values, dtype=float)
        if array.ndim != 1:
            raise ValueError(f"A vector's values are 1D, got shape {array.shape}.")
        if len(array) != len(space):
            raise ValueError(
                f"Got {len(array)} values for a space of {len(space)} labels."
            )
        nan = np.isnan(array)
        if nan.any():
            raise ValueError(
                f"The values are NaN at {int(nan.sum())} labels: "
                f"{_listed(space.index[nan])}."
            )
        array.flags.writeable = False
        self._values = array
        self._space = space
        self._name = None if name is None else str(name)

    @classmethod
    def from_series(cls, series: pd.Series, block: str | None = None) -> "Vector":
        """
        Build a vector from a Series, reading its labels.

        Parameters
        ----------
        series : pd.Series
            Numeric values. With a ``block`` level in its index, the vector has
            those blocks (see :meth:`Space.from_index`); without one, it has
            one block, named by `block`. The Series' name is the vector's.
        block : str, optional
            The name of the one block, for a Series without a ``block``
            level.

        Returns
        -------
        Vector
            The values on the Series' labels. With a ``block`` level, the rows
            of each block are put together, in the order the blocks first
            appear.

        Raises
        ------
        ValueError
            If a Series with a ``block`` level is given `block` too, one
            without is given no `block`, or its labels cannot make a
            :class:`Space`.

        Examples
        --------
        >>> import pandas as pd
        >>> from fips import Vector
        >>> obs = pd.Series(
        ...     [410.2, 412.8], index=pd.Index(["A", "B"], name="site"), name="obs"
        ... )
        >>> Vector.from_series(obs, block="co2").index
        MultiIndex([('co2', 'A'),
                    ('co2', 'B')],
                   names=['block', 'site'])
        """
        index = series.index
        if "block" in index.names:
            if block is not None:
                raise ValueError(
                    "The Series has a 'block' level, which names its blocks; "
                    "leave `block` out."
                )
            space = Space.from_index(index)
            rank = pd.Index(space.names).get_indexer(
                index.get_level_values("block").astype(str)
            )
            values = series.to_numpy(dtype=float)[np.argsort(rank, kind="stable")]
        else:
            if block is None:
                raise ValueError(
                    "A Series without a 'block' level needs `block`, the name "
                    "of its one block."
                )
            space = Space({block: index})
            values = series.to_numpy(dtype=float)
        name = None if series.name is None else str(series.name)
        return cls(values, space, name=name)

    @classmethod
    def from_blocks(
        cls, blocks: "Mapping[str, pd.Series | Vector]", name: str | None = None
    ) -> "Vector":
        """
        Build a vector from one Series (or vector of one block) for each block.

        Parameters
        ----------
        blocks : mapping of str to pd.Series or Vector
            Each block's name mapped to its values, in the order the blocks
            should have: a Series labeled by the block's own index (no
            ``block`` level), or a :class:`Vector` of one block (its labels
            and values; the key names the block).
        name : str, optional
            The vector's name.

        Returns
        -------
        Vector
            The values of every block, on the :class:`Space` of the blocks'
            indexes.

        Raises
        ------
        ValueError
            If a vector given for a block has more than one block, or the
            labels cannot make a :class:`Space`.

        Examples
        --------
        >>> import pandas as pd
        >>> from fips import Vector
        >>> f = Vector.from_blocks(
        ...     {"flux": pd.Series([1.0, 2.0], index=pd.Index([1, 2], name="cell"))}
        ... )
        >>> f.space
        Space(flux: 2 x ['cell'])
        """
        labels, values = {}, []
        for key, part in blocks.items():
            if isinstance(part, Vector):
                if len(part.space.names) != 1:
                    raise ValueError(
                        f"The vector given for block {key!r} has the blocks "
                        f"{list(part.space.names)}; give one block."
                    )
                labels[key] = part.space.select(part.space.names[0])
                values.append(part.values)
            else:
                labels[key] = part.index
                values.append(part.to_numpy(dtype=float))
        return cls(np.concatenate(values), Space(labels), name=name)

    @property
    def values(self) -> np.ndarray:
        """The values, one for each label, in the space's order (read-only)."""
        return self._values

    @property
    def space(self) -> Space:
        """The labels of the values."""
        return self._space

    @property
    def name(self) -> str | None:
        """What the values are: a string, or None."""
        return self._name

    @property
    def index(self) -> pd.MultiIndex:
        """The labels as one MultiIndex with the ``block`` level first (:attr:`Space.index`)."""
        return self._space.index

    def __len__(self) -> int:
        """Return the number of values."""
        return len(self._values)

    def __getitem__(self, block: str) -> "Vector":
        """
        Return the vector of one block.

        Parameters
        ----------
        block : str
            The block's name.

        Returns
        -------
        Vector
            The block's values on a space of that one block, labeled as the
            block was given.

        Raises
        ------
        KeyError
            If the vector has no block of that name.
        """
        labels = self._space.select(block)
        values = self._values[self._space.slice(block)]
        return Vector(values, Space({block: labels}), name=self._name)

    def to_series(self) -> pd.Series:
        """
        Return the values as a Series, labeled by :attr:`index`.

        The index has the ``block`` level first, and the Series holds a copy
        of the values.
        """
        return pd.Series(np.array(self._values), index=self.index, name=self._name)

    def to_xarray(self) -> xr.DataArray | xr.Dataset:
        """
        Return the values as xarray, each block on its own levels.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            For a vector of one block, a DataArray with a dimension for each
            of the block's levels (and no ``block`` dimension), named after
            the block. For several blocks, a Dataset with one such variable
            per block.

        Examples
        --------
        >>> import pandas as pd
        >>> from fips import Vector
        >>> flux = pd.Series(
        ...     [1.0, 2.0, 3.0, 4.0],
        ...     index=pd.MultiIndex.from_product(
        ...         [[40.0, 41.0], [-112.0, -111.0]], names=["lat", "lon"]
        ...     ),
        ... )
        >>> Vector.from_series(flux, block="flux").to_xarray().dims
        ('lat', 'lon')
        """
        arrays = {
            block: pd.Series(
                self._values[self._space.slice(block)],
                index=self._space.select(block),
                name=block,
            ).to_xarray()
            for block in self._space.names
        }
        if len(arrays) == 1:
            return next(iter(arrays.values()))
        return xr.Dataset(arrays)

    def align(self, space: Space) -> "Vector":
        """
        Return the values on `space`, which must have exactly the same labels.

        Parameters
        ----------
        space : Space
            The labels to put the values on, in any order (see
            :meth:`Space.align`).

        Returns
        -------
        Vector
            The same values, reordered to `space`.

        Raises
        ------
        ValueError
            If `space` and the vector's space do not have the same labels; the
            message names those missing from either.
        """
        return Vector(self._values[space.align(self._space)], space, name=self._name)

    def combine_first(self, other: "Vector") -> "Vector":
        """
        Return `other` with this vector's values where this vector has the label.

        The values of a smaller vector put back on a larger space: the
        estimate of a problem restricted to some of the state
        (:meth:`Problem.restrict <fips.Problem.restrict>`), on the whole state,
        with the prior where the state was held at it.

        Parameters
        ----------
        other : Vector
            The values for the labels this vector lacks. Every label of this
            vector must be one of `other`'s.

        Returns
        -------
        Vector
            On `other`'s space, named as this vector.

        Raises
        ------
        ValueError
            If this vector has a block or a label `other` lacks.

        Examples
        --------
        >>> import pandas as pd
        >>> from fips import Vector
        >>> cells = pd.Index(["a", "b", "c"], name="cell")
        >>> prior = Vector.from_series(
        ...     pd.Series([1.0, 2.0, 3.0], index=cells), block="flux"
        ... )
        >>> part = Vector.from_series(
        ...     pd.Series([20.0], index=cells[[1]], name="estimate"), block="flux"
        ... )
        >>> part.combine_first(prior).values
        array([ 1., 20.,  3.])
        """
        positions = other.space._subset(self._space, "This vector")
        values = np.array(other.values)
        values[positions] = self._values
        return Vector(values, other.space, name=self._name)

    def round_labels(self, decimals: int) -> "Vector":
        """
        Return the vector with its float labels rounded (:meth:`Space.round_labels`).

        Parameters
        ----------
        decimals : int
            Number of decimal places.

        Returns
        -------
        Vector
            The same values on the rounded labels.

        Raises
        ------
        ValueError
            If rounding makes two labels of a block equal.
        """
        return Vector(self._values, self._space.round_labels(decimals), name=self._name)

    def __repr__(self) -> str:
        """Return the name and space, then the values as a Series."""
        header = f"Vector(name={self._name!r}, {self._space!r})\n"
        return header + repr(self.to_series())

    def __getstate__(self) -> dict:
        """Pickle the values, the space and the name."""
        return {"values": self._values, "space": self._space, "name": self._name}

    def __setstate__(self, state: dict) -> None:
        """Restore from :meth:`__getstate__`, read-only again."""
        values = np.array(state["values"], dtype=float)
        values.flags.writeable = False
        self._values = values
        self._space = state["space"]
        self._name = state["name"]

"""
Correlation models: how correlated two labels are, from their coordinates.

A model is a correlation (one on the diagonal), read from the levels of a
space's labels. :class:`CorrelationModel` is the base of every one;
:class:`Independent` is the identity, :class:`Kronecker` the product of models
on separate levels, kept as factors, and the kernels of :mod:`fips.kernels`
(:class:`~fips.Exponential`, :class:`~fips.Gaussian`,
:class:`~fips.Matern`, :class:`~fips.Constant`) correlate
labels by their distance.

A model becomes a covariance with :meth:`~CorrelationModel.with_std`: given
the standard deviations as a :class:`~fips.Vector`, it is a
:class:`~fips.Covariance` at once; given a scalar or an array, it is a
:class:`~fips.models.CovarianceModel`, which
:meth:`~fips.models.CovarianceModel.build` puts on a space.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, overload

import numpy as np
import numpy.typing as npt
import pandas as pd

from fips.kronecker import KroneckerOperator
from fips.linalg import DenseOperator, DiagonalOperator, LinearOperator
from fips.models import CovarianceModel, _Scaled
from fips.vector import Vector

if TYPE_CHECKING:
    from fips.covariance import Covariance

logger = logging.getLogger(__name__)


def _names(value: str | Sequence[str] | None, what: str) -> tuple[str, ...]:
    """Return level names as a tuple, checking that each is a string named once."""
    if value is None:
        return ()
    names = (value,) if isinstance(value, str) else tuple(value)
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValueError(f"{what} must be level names (strings), got {name!r}.")
    if len(set(names)) != len(names):
        raise ValueError(f"{what} names a level twice: {list(names)}.")
    return names


@dataclass(frozen=True, init=False)
class CorrelationModel(ABC):
    """
    A correlation model: how correlated two labels are, from their coordinates.

    The base of :class:`Exponential`, :class:`Gaussian`, :class:`Matern`,
    :class:`Constant`, :class:`Independent` and :class:`Kronecker`. A model
    has no labels of its own: :meth:`with_std` scales it into a covariance,
    which reads the model's levels from the labels of the space it is put on.

    A new model is a frozen dataclass that subclasses this one, names the
    levels it reads in :attr:`levels` and implements :meth:`correlation`.

    See Also
    --------
    fips.models.CovarianceModel : A model with its standard deviations.
    """

    levels: tuple[str, ...]
    """The levels of the labels the correlation is a function of."""

    def _reads(self) -> tuple[str, ...]:
        """Return every level the model reads: :attr:`levels`, and any it groups by."""
        return self.levels

    @abstractmethod
    def correlation(self, coords: pd.DataFrame) -> np.ndarray:
        """
        Return the correlation between the rows of `coords`.

        Parameters
        ----------
        coords : pd.DataFrame
            One row per label, with a column for each level the model reads.

        Returns
        -------
        np.ndarray
            The ``n x n`` correlation matrix, one on the diagonal.
        """

    def _form(self, coords: pd.DataFrame) -> LinearOperator:
        """Return the correlation of the rows of `coords` in its cheapest form."""
        return DenseOperator(np.asarray(self.correlation(coords), dtype=float))

    @overload
    def with_std(self, std: Vector) -> "Covariance":  # noqa: D418
        """With a Vector: the covariance on its space."""

    @overload
    def with_std(self, std: "float | npt.ArrayLike") -> CovarianceModel:  # noqa: D418
        """With a scalar or an array: a model to build on a space."""

    def with_std(
        self, std: "float | npt.ArrayLike | Vector"
    ) -> "Covariance | CovarianceModel":
        r"""
        Return the covariance of this correlation with standard deviations `std`.

        Entry ``(i, j)`` is ``std[i] * C[i, j] * std[j]``. This is the one
        way to make a covariance from a model.

        Parameters
        ----------
        std : float, 1D array-like or Vector
            The standard deviation of each label, not negative. A scalar is
            the same for every label, an array is in the order of the space
            the covariance is built on, and a :class:`~fips.Vector` brings its
            own space.

        Returns
        -------
        Covariance or CovarianceModel
            For a Vector, the :class:`~fips.Covariance` on its space. For a
            scalar or an array, a :class:`~fips.models.CovarianceModel`, which
            :meth:`~fips.models.CovarianceModel.build` puts on a space.

        Raises
        ------
        ValueError
            If a standard deviation is negative or not finite, or `std` is
            not a scalar or 1D.

        Examples
        --------
        >>> import pandas as pd
        >>> from fips import Exponential, Space, Vector
        >>> space = Space({"flux": pd.Index([0.0, 1.0, 2.0], name="x")})
        >>> S = Exponential("x", 1.0).with_std(2.0).build(space)
        >>> S.dense().round(3)
        array([[4.   , 1.472, 0.541],
               [1.472, 4.   , 1.472],
               [0.541, 1.472, 4.   ]])
        >>> std = Vector([1.0, 1.0, 2.0], space)
        >>> Exponential("x", 1.0).with_std(std).variances.values
        array([1., 1., 4.])
        """
        if isinstance(std, Vector):
            return _Scaled(self, std.values).build(std.space)
        return _Scaled(self, std)


@dataclass(frozen=True, init=False)
class Independent(CorrelationModel):
    """
    The identity correlation: each label is correlated with itself only.

    By label, not by coordinates: two labels at the same place and time are
    still independent. ``Independent().with_std(std)`` is a diagonal
    covariance, and a nugget is a sum, ``model.with_std(a) +
    Independent().with_std(b)``.

    Examples
    --------
    >>> import pandas as pd
    >>> from fips import Independent, Space
    >>> space = Space({"co2": pd.Index(["a", "b"], name="site")})
    >>> Independent().with_std(0.5).build(space).dense()
    array([[0.25, 0.  ],
           [0.  , 0.25]])
    """

    levels: tuple[str, ...] = field(default=(), repr=False)
    """No levels, since the identity reads none."""

    def __init__(self):
        pass

    def correlation(self, coords: pd.DataFrame) -> np.ndarray:
        """
        Return the identity matrix, one row per row of `coords`.

        Parameters
        ----------
        coords : pd.DataFrame
            One row per label.

        Returns
        -------
        np.ndarray
            The identity.
        """
        return np.eye(len(coords))

    def _form(self, coords: pd.DataFrame) -> LinearOperator:
        """Return the identity as a diagonal."""
        return DiagonalOperator(np.ones(len(coords)))


@dataclass(frozen=True, init=False)
class Kronecker(CorrelationModel):
    """
    The product of correlation models on separate levels, kept as its factors.

    The correlation of two labels is the product of the factors'
    correlations of them. Each factor reads its own levels (no level is read
    by two) and is evaluated once, on the unique coordinates of its levels:
    the covariance is a :class:`~fips.kronecker.KroneckerOperator`, never
    the ``n x n`` matrix. The labels can be in any order and any subset of
    the product of the factors' coordinates, but each label must have its own
    combination of them.

    Parameters
    ----------
    *factors : CorrelationModel
        Two or more models, each reading at least one level (so not
        :class:`Independent`). A Kronecker factor brings its own factors.

    Raises
    ------
    ValueError
        If fewer than two factors are given, a factor reads no level, or two
        factors read the same level.

    Examples
    --------
    >>> import pandas as pd
    >>> from fips import Exponential, Kronecker, Space
    >>> labels = pd.MultiIndex.from_product(
    ...     [pd.date_range("2024-01-01", periods=3), [40.0, 40.5], [-112.0]],
    ...     names=["time", "lat", "lon"],
    ... )
    >>> model = Kronecker(
    ...     Exponential("time", "2D"),
    ...     Exponential(["lat", "lon"], 50.0, metric="haversine"),
    ... )
    >>> S_0 = model.with_std(0.5).build(Space({"flux": labels}))
    >>> S_0.data
    <6x6 KroneckerOperator with dtype=float64>
    """

    factors: tuple[CorrelationModel, ...]
    """The factors, each a model on its own levels."""
    levels: tuple[str, ...] = field(repr=False, compare=False)
    """The levels the factors read, factor by factor."""

    def __init__(self, *factors: CorrelationModel):
        flat: list[CorrelationModel] = []
        for factor in factors:
            if not isinstance(factor, CorrelationModel):
                raise TypeError(
                    f"Kronecker takes correlation models, got {type(factor).__name__}."
                )
            flat.extend(factor.factors if isinstance(factor, Kronecker) else [factor])
        if len(flat) < 2:
            raise ValueError("Kronecker needs at least two factors.")
        seen: dict[str, CorrelationModel] = {}
        for factor in flat:
            if not factor._reads():
                raise ValueError(
                    f"A Kronecker factor must read a level; {factor!r} reads none."
                )
            for level in factor._reads():
                if level in seen:
                    raise ValueError(
                        f"Level {level!r} is read by two factors, {seen[level]!r} "
                        f"and {factor!r}: the factors must read separate levels."
                    )
                seen[level] = factor
        object.__setattr__(self, "factors", tuple(flat))
        object.__setattr__(self, "levels", tuple(seen))

    def correlation(self, coords: pd.DataFrame) -> np.ndarray:
        """
        Return the correlation between the rows of `coords`, as a dense matrix.

        Parameters
        ----------
        coords : pd.DataFrame
            One row per label, with a column for each level the factors read.

        Returns
        -------
        np.ndarray
            The ``n x n`` product of the factors' correlations.
        """
        return self._form(coords).dense()

    def _form(self, coords: pd.DataFrame) -> KroneckerOperator:
        """
        Return the correlation as its factors, and each row's position in them.

        Each factor is evaluated on the unique coordinates of its levels, in
        the order they first appear in `coords`.
        """
        repeated = int(coords.duplicated(subset=list(self.levels)).sum())
        if repeated:
            raise ValueError(
                f"A Kronecker model needs each label to have its own combination "
                f"of the factors' levels {list(self.levels)}, but {repeated} "
                "labels repeat one. Labels that differ only on another level "
                "would be perfectly correlated."
            )
        matrices, positions = [], []
        for factor in self.factors:
            reads = list(factor._reads())
            pos = coords.groupby(reads, sort=False, dropna=False).ngroup().to_numpy()
            first = np.unique(pos, return_index=True)[1]
            unique = coords[reads].iloc[first].reset_index(drop=True)
            matrix = factor._form(unique).dense()
            if matrix.shape != (len(unique), len(unique)):
                raise ValueError(
                    f"{factor!r} gave a matrix of shape {matrix.shape} for "
                    f"{len(unique)} unique coordinates."
                )
            matrices.append(matrix)
            positions.append(pos)
        return KroneckerOperator(matrices, positions=positions)

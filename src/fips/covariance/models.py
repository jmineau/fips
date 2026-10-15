"""
Correlation models and covariance models: a covariance written before its labels.

A :class:`CorrelationModel` says how correlated two labels are, read from the
levels of a space's labels. :class:`Independent` is the identity,
:class:`Kronecker` the product of models on separate levels, kept as factors,
and the kernels of :mod:`fips.covariance.kernels` (:class:`~fips.Exponential`,
:class:`~fips.Gaussian`, :class:`~fips.Matern`, :class:`~fips.Constant`)
correlate labels by their distance.

A model becomes a covariance with :meth:`~CorrelationModel.with_std`: given the
standard deviations as a :class:`~fips.Vector`, it is a
:class:`~fips.Covariance` at once; given a scalar or an array, it is a
:class:`CovarianceModel`: everything a covariance needs except the
:class:`~fips.Space` it lives on. A sum of them, or
:meth:`Covariance.block_diag <fips.Covariance.block_diag>` of parts some of
which are models, is one too. :meth:`CovarianceModel.build` puts it on a space
and gives the :class:`~fips.Covariance`, in its cheapest form (Kronecker
factors stay factors, an independent model is a diagonal, a model ``within``
groups is sparse).
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import overload

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sp

from fips.covariance.covariance import Covariance, _side_by_side
from fips.linalg import (
    DenseOperator,
    DiagonalOperator,
    KroneckerOperator,
    LinearOperator,
    SparseOperator,
)
from fips.space import Space, _listed
from fips.vector import Vector

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
    fips.covariance.CovarianceModel : A model with its standard deviations.
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
    def with_std(self, std: "float | npt.ArrayLike") -> "CovarianceModel":  # noqa: D418
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
            scalar or an array, a :class:`~fips.covariance.CovarianceModel`, which
            :meth:`~fips.covariance.CovarianceModel.build` puts on a space.

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
        return DiagonalOperator._adopt(np.ones(len(coords)))


@dataclass(frozen=True, init=False)
class Kronecker(CorrelationModel):
    """
    The product of correlation models on separate levels, kept as its factors.

    The correlation of two labels is the product of the factors'
    correlations of them. Each factor reads its own levels (no level is read
    by two) and is evaluated once, on the unique coordinates of its levels:
    the covariance is a :class:`~fips.linalg.KroneckerOperator`, never
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
        return KroneckerOperator._adopt(matrices, positions=positions)


def _coords(space: Space, levels: Iterable[str]) -> pd.DataFrame:
    """
    Return the labels of `space` on `levels`, one row per label, in the space's order.

    Raises
    ------
    KeyError
        If the space has no such level.
    ValueError
        If a block of the space does not have a level: the model would read
        a missing label there.
    """
    needed = list(dict.fromkeys(levels))
    for level in needed:
        if level == "block":
            continue
        if level not in space.levels:
            raise KeyError(
                f"The model reads level {level!r}, which the space does not "
                f"have; its levels are {_listed(space.levels)}."
            )
        lacking = [b for b in space.names if level not in space.select(b).names]
        if lacking:
            raise ValueError(
                f"The model reads level {level!r}, which block(s) "
                f"{_listed(lacking)} do not have. Put independent blocks side "
                "by side with Covariance.block_diag."
            )
    return space.index.to_frame(index=False)[needed]


def _scaled(form: LinearOperator, std: np.ndarray) -> LinearOperator:
    """Return ``D C D`` for the correlation `form` and ``D = diag(std)``, keeping the form."""
    if isinstance(form, KroneckerOperator):
        return KroneckerOperator._adopt(
            form.factors, std=form.std * std, positions=form.positions
        )
    if isinstance(form, DiagonalOperator):
        return DiagonalOperator._adopt(std * form.values * std)
    if isinstance(form, SparseOperator):
        coo = sp.coo_array(form.matrix)
        data = std[coo.row] * coo.data * std[coo.col]
        return SparseOperator._adopt(
            sp.csr_array((data, (coo.row, coo.col)), shape=coo.shape)
        )
    if isinstance(form, DenseOperator):
        return DenseOperator._adopt(std[:, None] * form.array * std[None, :])
    raise TypeError(f"Cannot scale a {type(form).__name__}.")  # pragma: no cover


class CovarianceModel(ABC):
    """
    A covariance without its labels: it becomes a Covariance on a space.

    What :meth:`~fips.covariance.CorrelationModel.with_std` gives for a scalar
    or an array, a sum of covariance models (``a + b``), and
    :meth:`Covariance.block_diag <fips.Covariance.block_diag>` of parts
    some of which are models. A :class:`~fips.Covariance` may be added to a
    model, or be a part of one: it is aligned to the space by an exact join.

    See Also
    --------
    fips.covariance.CorrelationModel.with_std : A covariance model from a correlation.

    Examples
    --------
    The model-data mismatch of one site: an instrument error, independent,
    and a transport error correlated over 2.8 hours that day:

    >>> import pandas as pd
    >>> from fips import Exponential, Independent, Space
    >>> times = pd.date_range("2024-01-01 22:00", periods=4, freq="h")
    >>> obs = Space({"co2": pd.Index(times, name="obs_time")})
    >>> S_z = Independent().with_std(0.1) + Exponential(
    ...     "obs_time", "2.8h", within="obs_time.date"
    ... ).with_std(0.5)
    >>> S_z.build(obs).dense().round(3)
    array([[0.26 , 0.175, 0.   , 0.   ],
           [0.175, 0.26 , 0.   , 0.   ],
           [0.   , 0.   , 0.26 , 0.175],
           [0.   , 0.   , 0.175, 0.26 ]])
    """

    @abstractmethod
    def build(self, space: Space, name: str | None = None) -> Covariance:
        """
        Return the covariance on `space`.

        Parameters
        ----------
        space : Space
            The labels the covariance is put on. Each model reads its levels
            from them; a covariance given as a part is aligned to them.
        name : str, optional
            The covariance's name.

        Returns
        -------
        Covariance
            The covariance, rows and columns in the order of `space`, in its
            cheapest form.

        Raises
        ------
        KeyError
            If a model reads a level the space does not have, or a block
            the model names is not in the space.
        ValueError
            If a block lacks a level a model reads, an array of standard
            deviations is not as long as the space, or the labels of a
            covariance given as a part do not match the space's.
        """

    def __add__(self, other: "CovarianceModel | Covariance") -> "CovarianceModel":
        """Return the sum with a covariance model or a covariance."""
        if not isinstance(other, (CovarianceModel, Covariance)):
            return NotImplemented
        return _Sum((*_terms(self), *_terms(other)))

    def __radd__(self, other: "CovarianceModel | Covariance") -> "CovarianceModel":
        """Return the sum of a covariance and this model."""
        if not isinstance(other, (CovarianceModel, Covariance)):
            return NotImplemented
        return _Sum((*_terms(other), *_terms(self)))


def _terms(part: "CovarianceModel | Covariance") -> tuple:
    """Return the terms of a sum, or the one term."""
    return part.terms if isinstance(part, _Sum) else (part,)


def _check_space(space: Space) -> None:
    """Raise a TypeError unless `space` is a Space."""
    if not isinstance(space, Space):
        raise TypeError(f"build takes a Space, got {type(space).__name__}.")


@dataclass(frozen=True, eq=False)
class _Scaled(CovarianceModel):
    """A correlation model times its standard deviations: ``D C D``."""

    model: "CorrelationModel"
    std: "float | npt.ArrayLike"

    def __post_init__(self):
        std = np.asarray(self.std, dtype=float)
        if std.ndim > 1:
            raise ValueError(f"std must be a scalar or 1D, got shape {std.shape}.")
        if not np.isfinite(std).all() or (std < 0).any():
            raise ValueError("The standard deviations must be finite and not negative.")
        if std.ndim == 1:
            std = std.copy()
            std.flags.writeable = False
        object.__setattr__(self, "std", std)

    def build(self, space: Space, name: str | None = None) -> Covariance:
        """Return the covariance on `space` (see :meth:`CovarianceModel.build`)."""
        _check_space(space)
        std = np.asarray(self.std)
        if std.ndim == 0:
            std = np.full(len(space), float(std))
        elif len(std) != len(space):
            raise ValueError(
                f"Got {len(std)} standard deviations for a space of "
                f"{len(space)} labels."
            )
        form = self.model._form(_coords(space, self.model._reads()))
        return Covariance(_scaled(form, std), space, name=name)

    def __repr__(self) -> str:
        """Return how the model was made."""
        std = self.std if np.ndim(self.std) == 0 else f"<{np.size(self.std)} values>"
        return f"{self.model!r}.with_std({std})"


def _on(part: "CovarianceModel | Covariance", space: Space) -> Covariance:
    """Return `part` on `space`: a model built there, a covariance aligned to it."""
    if isinstance(part, Covariance):
        return Covariance(part._on(space, space), space)
    return part.build(space)


@dataclass(frozen=True, eq=False)
class _Sum(CovarianceModel):
    """A sum of covariance models (and covariances), each built on the same space."""

    terms: tuple

    def build(self, space: Space, name: str | None = None) -> Covariance:
        """Return the sum on `space` (see :meth:`CovarianceModel.build`)."""
        _check_space(space)
        total = _on(self.terms[0], space)
        for term in self.terms[1:]:
            total = total + _on(term, space)
        return Covariance(total.data, space, name=name)

    def __repr__(self) -> str:
        """Return the terms joined by ``+``."""
        return " + ".join(repr(term) for term in self.terms)


@dataclass(frozen=True, eq=False)
class _BlockDiagonal(CovarianceModel):
    """Independent parts side by side, each on its own blocks of the space."""

    parts: tuple  # of (block names, CovarianceModel or Covariance)
    name: str | None = None

    def build(self, space: Space, name: str | None = None) -> Covariance:
        """Return the block-diagonal covariance on `space` (see :meth:`CovarianceModel.build`)."""
        _check_space(space)
        named = [block for names, _ in self.parts for block in names]
        missing = [block for block in space.names if block not in named]
        extra = [block for block in named if block not in space]
        if missing or extra:
            raise ValueError(
                "The block diagonal covariance and the space must have the same "
                f"blocks; the space has {_listed(space.names)}, the parts cover "
                f"{_listed(named)}."
            )
        built = [
            _on(part, Space({b: space.select(b) for b in names}))
            for names, part in self.parts
        ]
        form, joined = _side_by_side(built)
        name = self.name if name is None else name
        if joined.names != space.names:  # put the blocks in the space's order
            form = Covariance(form, joined)._on(space, space)
        return Covariance(form, space, name=name)

    def __repr__(self) -> str:
        """Return the parts by block."""
        parts = ", ".join(f"{'+'.join(names)}={part!r}" for names, part in self.parts)
        return f"Covariance.block_diag({parts})"


def _block_diagonal(
    covariances: tuple, blocks: dict, name: str | None
) -> _BlockDiagonal:
    """Return the model :meth:`Covariance.block_diag <fips.Covariance.block_diag>` gives when a part is a model."""
    parts: list[tuple[tuple[str, ...], Covariance | CovarianceModel]] = []
    for cov in covariances:
        if not isinstance(cov, Covariance):
            raise TypeError(
                "block_diag takes Covariances in order, and covariance models "
                f"by block name (flux=...); got {type(cov).__name__}."
            )
        parts.append((cov.space.names, cov))
    for key, part in blocks.items():
        if isinstance(part, Covariance) and part.space.names != (key,):
            raise ValueError(f"The covariance given as {key}= is on {part.space}.")
        if not isinstance(part, (Covariance, CovarianceModel)):
            raise TypeError(
                f"block_diag takes a Covariance or a CovarianceModel as {key}=, "
                f"got {type(part).__name__}."
            )
        parts.append(((str(key),), part))
    names = [block for blocks_of_part, _ in parts for block in blocks_of_part]
    twice = sorted({block for block in names if names.count(block) > 1})
    if twice:
        raise ValueError(f"Two parts have a block named {_listed(twice)}.")
    return _BlockDiagonal(tuple(parts), name=name)


# ``Covariance`` cannot import this module (this one imports it), so it asks
# for the block diagonal of parts some of which are models through this hook.
Covariance._block_diag_models = staticmethod(_block_diagonal)

"""
Covariance models: covariances waiting for their labels.

A correlation model (:mod:`fips.correlation`, :mod:`fips.kernels`) scaled by
:meth:`~fips.correlation.CorrelationModel.with_std` (with a scalar or an array),
a sum of them, or :meth:`Covariance.block_diag <fips.Covariance.block_diag>`
of parts some of which are models, is a :class:`CovarianceModel`: everything
a covariance needs except the :class:`~fips.Space` it lives on.
:meth:`CovarianceModel.build` puts it on one and gives the
:class:`~fips.Covariance`, in its cheapest form (Kronecker factors stay
factors, an independent model is a diagonal, a model ``within`` groups is
sparse).
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sp

from fips.covariance import Covariance, _side_by_side
from fips.kronecker import KroneckerOperator
from fips.linalg import (
    DenseOperator,
    DiagonalOperator,
    LinearOperator,
    SparseOperator,
)
from fips.space import Space, _listed

if TYPE_CHECKING:
    from fips.kernels import CorrelationModel

logger = logging.getLogger(__name__)


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

    What :meth:`~fips.correlation.CorrelationModel.with_std` gives for a scalar
    or an array, a sum of covariance models (``a + b``), and
    :meth:`Covariance.block_diag <fips.Covariance.block_diag>` of parts
    some of which are models. A :class:`~fips.Covariance` may be added to a
    model, or be a part of one: it is aligned to the space by an exact join.

    See Also
    --------
    fips.correlation.CorrelationModel.with_std : A covariance model from a correlation.

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

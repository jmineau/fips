"""
Covariances: symmetric operators from a space to itself.

A :class:`Covariance` is a symmetric :class:`~fips.Operator` from a space to
itself, kept in its cheapest form, with the variances, solves,
log-determinant, quadratic form and square root a covariance needs. The
correlation models of :mod:`fips.covariance` make one from the labels
(``model.with_std(std)``).
"""

import logging
from collections.abc import Callable
from typing import ClassVar, Protocol, Self

import numpy as np
import numpy.typing as npt
import pandas as pd

from fips.linalg import (
    BlockDiagonalOperator,
    DiagonalOperator,
    LinearOperator,
    LinearOperatorLike,
    as_linear_operator,
)
from fips.linalg.forms import _adopt
from fips.operator import Operator
from fips.space import Space
from fips.vector import Vector

logger = logging.getLogger(__name__)


class _Model(Protocol):
    """A covariance model, as :meth:`Covariance.block_diag` sees one: it builds a covariance on a space."""

    def build(self, space: Space, name: str | None = None) -> "Covariance":
        """Return the covariance on `space`."""
        ...


def _side_by_side(parts: "list[Covariance]") -> tuple[LinearOperator, Space]:
    """Return the block diagonal form of `parts`, and the space of their blocks in order."""
    if not parts:
        raise ValueError("block_diag needs at least one covariance.")
    blocks: dict[str, pd.Index] = {}
    for part in parts:
        for name in part.space.names:
            if name in blocks:
                raise ValueError(f"Two blocks are named {name!r}.")
            blocks[name] = part.space.select(name)
    forms = [part.data for part in parts]
    if len(forms) == 1:
        return forms[0], Space(blocks)
    if all(isinstance(form, DiagonalOperator) for form in forms):
        values = np.concatenate([form.diagonal() for form in forms])
        return DiagonalOperator._adopt(values), Space(blocks)
    return BlockDiagonalOperator(forms), Space(blocks)


class Covariance(Operator):
    """
    A covariance: a symmetric operator from a :class:`~fips.Space` to itself.

    The matrix is kept in its cheapest form, as an :class:`Operator`'s is: a
    diagonal covariance is its variances, a Kronecker one its factors, a
    block diagonal one each block in its own form. On top of an operator's
    methods it gives the :attr:`variances` and, for a positive-definite
    covariance, :meth:`solve`, :meth:`logdet`, :meth:`quadratic_form` and
    :meth:`sqrt`, each done the way its form makes cheap.

    Parameters
    ----------
    data : array-like, SciPy sparse matrix or LinearOperator
        The matrix, square, rows and columns in the order of `space`. An array
        or a sparse matrix is copied, and the copy is read-only, as for an
        :class:`~fips.Operator`.
    space : Space
        The labels of both axes.
    name : str, optional
        What the covariance is ("prior_cov", "obs_cov", ...).

    Raises
    ------
    TypeError
        If `data` is a pandas object or `space` is not a Space.
    ValueError
        If the matrix does not have one row and one column per label, or it
        stores a NaN.

    See Also
    --------
    Covariance.diagonal : A covariance from its variances.
    Covariance.block_diag : Independent covariances side by side.
    fips.Exponential : A correlation model, which ``with_std`` makes a covariance.
    Operator.from_frame : A covariance from a labeled DataFrame.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from fips import Covariance, Space, Vector
    >>> space = Space({"flux": pd.Index(["a", "b"], name="cell")})
    >>> S = Covariance(np.array([[2.0, 1.0], [1.0, 2.0]]), space)
    >>> S.variances.values
    array([2., 2.])
    >>> S.solve(Vector([3.0, 3.0], space)).values
    array([1., 1.])
    """

    _block_diag_models: ClassVar[Callable[[tuple, dict, str | None], _Model]]
    """Makes the model :meth:`block_diag` gives when a part is a model; :mod:`fips.covariance.models` sets it."""

    def __init__(self, data: LinearOperatorLike, space: Space, name: str | None = None):
        super().__init__(data, space, space, name=name)

    @classmethod
    def _build(
        cls, data: LinearOperatorLike, rows: Space, cols: Space, name: str | None
    ) -> Self:
        """Return a covariance on `rows`, with its columns put in the order of the rows."""
        if rows != cols:
            try:
                order = rows.align(cols)
            except ValueError as err:
                raise ValueError(
                    f"A covariance has the same labels on both axes. {err}"
                ) from None
            data = as_linear_operator(data)._select(np.arange(len(rows)), order)
        return cls(data, rows, name=name)

    def _sum_like(
        self, other: Operator, total: LinearOperator, name: str | None
    ) -> Operator:
        """Return the sum `total`: a covariance when `other` is one too."""
        if isinstance(other, Covariance):
            return Covariance(total, self._rows, name=name)
        return super()._sum_like(other, total, name)

    def _like(self, data: LinearOperatorLike, rows: Space, cols: Space) -> Operator:
        """Return a covariance when the two axes are one space, else an operator."""
        if rows is cols or rows == cols:
            return Covariance(data, rows, name=self._name)
        return Operator(data, rows, cols, name=self._name)

    def __repr__(self) -> str:
        """Return the name, shape and form, then the space."""
        return (
            f"Covariance(name={self._name!r}, shape={self.shape}, "
            f"form={type(self._data).__name__})\n  space: {self._rows!r}"
        )

    @property
    def space(self) -> Space:
        """The labels of both axes."""
        return self._rows

    @classmethod
    def diagonal(
        cls,
        values: "float | npt.ArrayLike | Vector",
        space: Space | None = None,
        name: str | None = None,
    ) -> Self:
        """
        Build a diagonal covariance from its variances.

        The covariance is kept as its diagonal, so it costs one number per
        label whatever the size.

        Parameters
        ----------
        values : float, 1D array-like or Vector
            The variances (an array is copied). A scalar is the same for every
            label; a :class:`~fips.Vector` brings its own space.
        space : Space, optional
            The labels. Required for a scalar or an array; for a Vector it
            must be left out, or have exactly the vector's labels (the
            variances are put in its order).
        name : str, optional
            The covariance's name.

        Returns
        -------
        Covariance
            The diagonal covariance.

        Raises
        ------
        ValueError
            If a scalar or an array comes without a space, or `values` is not
            a scalar or 1D, or has a length other than the space's.

        Examples
        --------
        >>> import pandas as pd
        >>> from fips import Covariance, Space
        >>> space = Space({"co2": pd.Index(["a", "b", "c"], name="site")})
        >>> S = Covariance.diagonal(0.25, space)
        >>> S.dense()
        array([[0.25, 0.  , 0.  ],
               [0.  , 0.25, 0.  ],
               [0.  , 0.  , 0.25]])
        >>> Covariance.diagonal(S.variances).variances.values
        array([0.25, 0.25, 0.25])
        """
        if isinstance(values, Vector):
            vector = values if space is None else values.align(space)
            return cls(DiagonalOperator._adopt(vector.values), vector.space, name=name)
        if space is None:
            raise ValueError(
                "A diagonal from a scalar or an array needs `space`, its labels."
            )
        diagonal = np.asarray(values, dtype=float)
        if diagonal.ndim == 0:
            diagonal = np.full(len(space), float(diagonal))
        elif diagonal.ndim != 1:
            raise ValueError(
                f"The variances must be a scalar or 1D, got shape {diagonal.shape}."
            )
        elif len(diagonal) != len(space):
            raise ValueError(
                f"Got {len(diagonal)} variances for a space of {len(space)} labels."
            )
        return cls(DiagonalOperator(diagonal), space, name=name)

    @classmethod
    def block_diag(
        cls,
        *covariances: "Covariance",
        name: str | None = None,
        **blocks: "Covariance | _Model",
    ) -> "Self | _Model":
        """
        Put independent covariances side by side: each on its own blocks, zero between.

        Each part keeps its form (a Kronecker block stays factors, a diagonal
        one its variances), so the result costs what its parts do.

        Parameters
        ----------
        *covariances : Covariance
            Covariances that know their blocks, in order.
        name : str, optional
            The covariance's name.
        **blocks : Covariance or CovarianceModel
            Parts named by block, after the positional ones: a covariance on
            that one block, or a :class:`~fips.covariance.CovarianceModel` (as
            ``model.with_std(std)``), which gets the block's labels when built.

        Returns
        -------
        Covariance or CovarianceModel
            The covariance on the blocks of every part, one after the other;
            while a part is a model, a :class:`~fips.covariance.CovarianceModel`
            to :meth:`~fips.covariance.CovarianceModel.build` on a space.

        Raises
        ------
        ValueError
            If no part is given, two have a block of one name, or a covariance
            given as `block=` is on another block.

        Examples
        --------
        >>> import pandas as pd
        >>> from fips import Covariance, Independent, Space
        >>> cells = Space({"flux": pd.Index(["a", "b"], name="cell")})
        >>> sites = Space({"bias": pd.Index(["A"], name="site")})
        >>> flux = Covariance.diagonal(4.0, cells)
        >>> bias = Covariance.diagonal(1.0, sites)
        >>> S = Covariance.block_diag(flux, bias)
        >>> S.space
        Space(flux: 2 x ['cell'], bias: 1 x ['site'])
        >>> S.variances.values
        array([4., 4., 1.])
        >>> model = Covariance.block_diag(flux, bias=Independent().with_std(2.0))
        >>> model.build(S.space).variances.values
        array([4., 4., 4.])
        """
        parts: list[Covariance] = []
        for part in (*covariances, *blocks.values()):
            if not isinstance(part, Covariance):
                return cls._block_diag_models(covariances, blocks, name)
            parts.append(part)
        for key, cov in zip(blocks, parts[len(covariances) :], strict=True):
            if cov.space.names != (key,):
                raise ValueError(f"The covariance given as {key}= is on {cov.space}.")
        return cls(*_side_by_side(parts), name=name)

    @property
    def variances(self) -> Vector:
        """
        The variances: the diagonal, as a vector on the covariance's space.

        Read from the form without forming the matrix (from the variances of
        a diagonal covariance, the factors of a Kronecker one).
        """
        return Vector(self._data.diagonal(), self._rows, name="variance")

    def solve(self, b: "Vector | Operator") -> "Vector | Operator":
        r"""
        Return :math:`S^{-1} b`, for a positive-definite covariance :math:`S`.

        Parameters
        ----------
        b : Vector or Operator
            A vector on the covariance's space, or an operator whose rows are
            on it (both in any order).

        Returns
        -------
        Vector or Operator
            The solution, on the covariance's space (and `b`'s columns).

        Raises
        ------
        TypeError
            If `b` is an array: it has no labels to check.
        ValueError
            If the labels of `b` do not match the covariance's.
        numpy.linalg.LinAlgError
            If the covariance is singular.

        See Also
        --------
        fips.linalg.LinearOperator.solve : How each form solves.
        """
        if isinstance(b, Vector):
            return Vector(self._data.solve(b.align(self._rows).values), self._rows)
        if isinstance(b, Operator):
            rhs = b._on(self._rows, b.cols)
            return Operator(_adopt(self._data.solve(rhs)), self._rows, b.cols)
        raise TypeError(
            f"Covariance.solve takes a Vector or an Operator, not {type(b).__name__}: "
            "an array has no labels to check. Use cov.data.solve(b) to solve "
            "by position."
        )

    def logdet(self) -> float:
        """
        Return ``log(det(S))``, from the form's factors (see :meth:`fips.linalg.LinearOperator.logdet`).

        Raises
        ------
        numpy.linalg.LinAlgError
            If the covariance is not positive-definite.
        """
        return self._data.logdet()

    def quadratic_form(self, x: Vector) -> float:
        r"""
        Return :math:`x^T S^{-1} x`, the squared Mahalanobis length of `x`.

        Parameters
        ----------
        x : Vector
            A vector on the covariance's space, in any order.

        Returns
        -------
        float
            The quadratic form of the inverse covariance.

        Raises
        ------
        ValueError
            If the labels of `x` do not match the covariance's.
        numpy.linalg.LinAlgError
            If the covariance is singular.

        Examples
        --------
        >>> import pandas as pd
        >>> from fips import Covariance, Space, Vector
        >>> space = Space({"co2": pd.Index(["a", "b"], name="site")})
        >>> S = Covariance.diagonal([4.0, 1.0], space)
        >>> S.quadratic_form(Vector([2.0, 1.0], space))
        2.0
        """
        values = x.align(self._rows).values
        return float(values @ self._data.solve(values))

    def sqrt(self) -> Operator:
        r"""
        Return a square root :math:`L`, with :math:`L L^T = S`.

        A diagonal covariance gives the diagonal of standard deviations, a
        block diagonal one the root of each block, any other form the lower
        Cholesky factor of its dense matrix. :math:`L` maps independent unit
        variables, one per label, to the covariance's space: ``L @ w`` with
        ``w`` standard normal is a draw from :math:`N(0, S)`.

        Returns
        -------
        Operator
            :math:`L`, with the covariance's space on both axes.

        Raises
        ------
        numpy.linalg.LinAlgError
            If the covariance is not positive-definite.
        """
        return Operator(self._data.sqrt(), self._rows, self._rows, name=self._name)

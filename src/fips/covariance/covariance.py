"""
Covariances: symmetric operators from a space to itself.

A :class:`Covariance` is a symmetric :class:`~fips.Operator` from a space to
itself, kept in its cheapest form, with the variances, solves,
log-determinant, quadratic form and square root a covariance needs. The
correlation models of :mod:`fips.covariance` make one from the labels
(``model.with_std(std)``).
"""

import logging
from collections.abc import Callable, Container, Mapping
from typing import ClassVar, Protocol, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from numpy.linalg import LinAlgError

from fips.linalg import (
    BlockDiagonalOperator,
    DenseOperator,
    DiagonalOperator,
    KroneckerOperator,
    LinearOperator,
    LinearOperatorLike,
    SparseOperator,
    as_linear_operator,
)
from fips.linalg.forms import _adopt, _asymmetry
from fips.operator import Operator, _difference
from fips.space import Space, _listed, _plain
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


def _check_symmetric(form: LinearOperator, space: Space, start: int = 0) -> None:
    """
    Raise a ValueError if a dense or sparse form (or block of one) is not symmetric.

    The check is ``max|A - A^T| <= 1e-10 * max|A|``, without forming a dense
    matrix from a sparse one. A diagonal form is symmetric; a block diagonal
    one is checked block by block; a Kronecker form checked its factors when
    it was built; any other form is not checked.
    """
    if isinstance(form, BlockDiagonalOperator):
        for block in form.blocks:
            _check_symmetric(block, space, start)
            start += block.shape[0]
        return
    if isinstance(form, DenseOperator):
        found = _asymmetry(form.array)
    elif isinstance(form, SparseOperator):
        found = _asymmetry(form.matrix)
    else:
        return
    if found is not None:
        i, j, a_ij, a_ji, scale = found
        row, col = (_plain(space.index[start + k]) for k in (i, j))
        raise ValueError(
            f"A covariance must be symmetric, but the entry at ({row!r}, {col!r}) "
            f"is {a_ij:.6g} and the one at ({col!r}, {row!r}) is "
            f"{a_ji:.6g}: a difference of {abs(a_ij - a_ji):.3g}, the largest, "
            f"more than 1e-10 of the largest entry ({scale:.3g}). Give the "
            "whole matrix (both triangles), or (A + A.T) / 2 if the difference "
            "is roundoff."
        )


def _check_positive_semidefinite(form: LinearOperator) -> None:
    """
    Raise a ValueError if a form fips holds is not positive semidefinite.

    The check is the form's draw root (``form._root``), built the way the form
    makes cheapest: the square roots of a diagonal, the Cholesky factor of
    each group of correlated rows of a sparse matrix, of each block of a block
    diagonal one and of each factor of a Kronecker one, and of a dense matrix.
    Where there is no Cholesky factor (a singular matrix) it is the
    eigendecomposition, which refuses an eigenvalue below zero by more than
    roundoff, ``n * eps * max|lambda|``. The root is kept, so a draw from the
    covariance factors nothing again. A form of another class (a SciPy
    operator) is not checked: its root would need its dense matrix.
    """
    if not isinstance(
        form,
        (
            DenseOperator,
            SparseOperator,
            DiagonalOperator,
            BlockDiagonalOperator,
            KroneckerOperator,
        ),
    ):
        return
    try:
        _ = form._root  # built, and kept for sample
    except LinAlgError as err:
        raise ValueError(
            f"{err} A covariance estimated or tapered by hand can have negative "
            "eigenvalues: build it from a correlation model, or set its negative "
            "eigenvalues to zero before giving it."
        ) from None


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
        If `data` is a pandas object or an :class:`~fips.Operator` (its
        labels would be ignored: use ``cov.align(space, space)``, or
        ``cov.data`` for the matrix), or `space` is not a Space.
    ValueError
        If the matrix does not have one row and one column per label, it
        stores a NaN, or a dense or sparse matrix is not symmetric: its
        largest ``|A - A^T|`` is more than 1e-10 of its largest entry (the
        message names the two entries and their labels). A diagonal or
        Kronecker form is symmetric by construction, and a block diagonal one
        is checked block by block. Also if the matrix is not positive
        semidefinite: an eigenvalue is below zero by more than roundoff,
        ``n * eps`` times the largest (the message names the smallest).

    Notes
    -----
    The positive semidefinite check is the root :meth:`sample` draws with,
    built here and kept: the Cholesky factor of a dense matrix (``n^3 / 3``
    operations, and ``n x n`` more memory), of each group of correlated rows
    of a sparse one, of each block of a block diagonal one and of each factor
    of a Kronecker one, or the square roots of a diagonal. A singular matrix,
    which has no Cholesky factor, is eigendecomposed instead (several times
    the cost), with one INFO line. A covariance fips builds from a correlation
    model or computes (a sum, a reordering, a posterior) is positive
    semidefinite by construction and is not checked.

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
        _check_symmetric(self._data, space)
        _check_positive_semidefinite(self._data)

    @classmethod
    def _computed(
        cls, data: LinearOperatorLike, space: Space, name: str | None = None
    ) -> Self:
        """
        Return a covariance fips made from covariances or models, without the symmetry check.

        A sum, a reordering, a block diagonal or a product such as
        ``W S W^T`` of symmetric matrices is symmetric up to roundoff, which in
        a posterior covariance can be more than the check's 1e-10; what a
        caller hands over goes through ``Covariance(...)``, which checks it.
        """
        cov = cls.__new__(cls)
        Operator.__init__(cov, data, space, space, name=name)
        return cov

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

    @classmethod
    def _check_blocks(
        cls,
        labeled: Mapping[tuple[str, str], Operator],
        unlabeled: Container[tuple[str, str]],
        rows: Space,
        cols: Space,
    ) -> None:
        """
        Raise if a block of a covariance built from blocks lacks a variance.

        A variance is the entry of a label on the diagonal, which only the
        block's own pair ``(block, block)`` can give. A pair that does not
        cover every label of its block (or is not there) would leave those
        entries 0 and the covariance singular, so it raises, naming them. An
        unlabeled pair already has one row and column per label. Rows and
        columns on other blocks are not a covariance at all, and
        :meth:`_build` says so.
        """
        if set(rows.names) != set(cols.names):
            return
        for block in rows.names:
            pair = (block, block)
            if pair in unlabeled:
                continue
            if pair not in labeled:
                raise ValueError(
                    f"Block {block!r} has no {pair!r} pair, so its "
                    f"{len(rows.select(block))} variances would be 0. Give the "
                    "pair, with a variance for every label."
                )
            for axis, space in (("rows", rows), ("columns", cols)):
                have = labeled[pair].rows if axis == "rows" else labeled[pair].cols
                lacking, _ = _difference(space.select(block), have.select(block))
                if len(lacking):
                    raise ValueError(
                        f"The {axis} of the matrix of {pair!r} lack "
                        f"{len(lacking)} of the {len(space.select(block))} labels "
                        f"of block {block!r}: {_listed(lacking)}. A label without "
                        "a row and column there gets a variance of 0, which makes "
                        "the covariance singular. Give the pair every label of "
                        "the block."
                    )

    def _like(self, data: LinearOperatorLike, rows: Space, cols: Space) -> Operator:
        """Return a covariance when the two axes are one space, else an operator."""
        if rows is cols or rows == cols:
            return Covariance._computed(data, rows, name=self._name)
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
            a scalar or 1D, or has a length other than the space's, or a
            variance is below zero by more than roundoff.

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
            The covariance's name. So a block called ``"name"`` cannot be
            given by keyword: build its part on the block and give the
            Covariance in order.
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
        TypeError
            If `name` is a covariance or a model: a part for a block called
            ``"name"``, which the keyword cannot take.
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
        if isinstance(name, Covariance) or hasattr(name, "build"):
            raise TypeError(
                "block_diag's `name` is the covariance's name, so it cannot "
                "take the part of a block called 'name'. Build that part on its "
                "block (model.build(Space({'name': labels}))) and give the "
                "Covariance in order, before the parts named by block."
            )
        parts: list[Covariance] = []
        for part in (*covariances, *blocks.values()):
            if not isinstance(part, Covariance):
                return cls._block_diag_models(covariances, blocks, name)
            parts.append(part)
        for key, cov in zip(blocks, parts[len(covariances) :], strict=True):
            if cov.space.names != (key,):
                raise ValueError(f"The covariance given as {key}= is on {cov.space}.")
        return cls._computed(*_side_by_side(parts), name=name)

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
        block diagonal one the root of each block, and a Kronecker one whose
        rows are its whole grid a root applied through its factors. Any
        other form, a Kronecker one on a subset of its grid among them, gives
        the lower Cholesky factor of its dense matrix.
        :math:`L` maps independent unit variables, one per label, to the
        covariance's space: ``L @ w`` with ``w`` standard normal is a draw
        from :math:`N(0, S)`.

        Returns
        -------
        Operator
            :math:`L`, with the covariance's space on both axes.

        Raises
        ------
        numpy.linalg.LinAlgError
            If the covariance is not positive-definite.

        See Also
        --------
        sample : A draw, made without forming :math:`L` where a form has a cheaper root.
        """
        return Operator(self._data.sqrt(), self._rows, self._rows, name=self._name)

    def aggregate(self, W: Operator) -> "Covariance":
        r"""
        Return the covariance of aggregates, :math:`W S W^T`.

        For :math:`x` with covariance :math:`S`, :math:`W x` has covariance
        :math:`W S W^T`: one row and one column per aggregate (a total per
        year, the mean over a region, a trend). Its diagonal is the variance
        of each aggregate, and its off-diagonal entries are what makes a
        combination of aggregates (the difference of two years) more or less
        uncertain than its parts. The product is taken in the forms of
        :math:`W` and :math:`S` (through the factors of a Kronecker
        covariance), so nothing of the covariance's size is formed when
        :math:`W` has few rows.

        Parameters
        ----------
        W : Operator
            An aggregation, ``k x n``, such as :func:`fips.aggregate.sum`'s:
            its columns the covariance's labels (in any order), its rows the
            aggregates.

        Returns
        -------
        Covariance
            :math:`W S W^T`, on the rows of `W`, with this covariance's name.

        Raises
        ------
        TypeError
            If `W` is not an Operator.
        ValueError
            If the columns of `W` do not have exactly the covariance's labels.

        See Also
        --------
        fips.Posterior.cov : The posterior covariance of aggregates of the state.
        fips.Posterior.uncertainty : The prior and posterior uncertainty of aggregates.
        fips.Problem.aggregate : The problem of aggregated observations.

        Examples
        --------
        The total of two cells, each with variance 1, correlated by 0.5:

        >>> import numpy as np
        >>> import pandas as pd
        >>> import fips
        >>> space = fips.Space({"flux": pd.Index(["a", "b"], name="cell")})
        >>> S = fips.Covariance(np.array([[1.0, 0.5], [0.5, 1.0]]), space)
        >>> total = fips.Operator(
        ...     np.ones((1, 2)),
        ...     fips.Space({"total": pd.Index(["all"], name="region")}),
        ...     space,
        ... )
        >>> S.aggregate(total).dense()
        array([[3.]])
        """
        if not isinstance(W, Operator):
            raise TypeError(
                f"An aggregation is an Operator, not {type(W).__name__}: build one "
                "with fips.aggregate.sum or fips.aggregate.mean."
            )
        form = W._on(
            W.rows, self._rows, "the aggregation", ("its rows", "the covariance")
        )
        return Covariance._computed(
            (form @ self._data) @ form._transpose(), W.rows, name=self._name
        )

    def sample(self, rng: int | np.random.Generator | None = None) -> Vector:
        r"""
        Return one draw from :math:`N(0, S)`, a vector on the covariance's space.

        The draw is :math:`R w` with :math:`R R^T = S` and :math:`w` standard
        normal, and :math:`R` is the root the form makes cheapest, never the
        ``n x n`` matrix where the form is cheaper than that. A diagonal
        covariance scales its normals by the standard deviations; a block
        diagonal one draws each block in turn; a Kronecker one applies the
        Cholesky factors of its factors to normals on the whole product grid
        and keeps its rows (a part of a Gaussian vector has the part of its
        covariance), so a subset of the grid costs what the whole grid does;
        a sparse one draws each group of correlated rows (a ``within=``
        kernel's site-days) from the Cholesky factor of its own block; a dense
        one uses its Cholesky factor. The root is built once and kept, so a
        loop of draws factors once: by ``Covariance(...)``, whose positive
        semidefinite check it is, or by the first draw of a covariance fips
        built from a model.

        A semi-definite covariance, a ``Constant()`` kernel's rank-one block
        for one, has no Cholesky factor, so it is drawn from its
        eigendecomposition, with the eigenvalues that are negative by roundoff
        set to zero and one INFO line logged.

        Parameters
        ----------
        rng : int, numpy.random.Generator or None, default None
            A seed, or the generator to draw from (used as given, so one
            generator passed to every call gives independent draws). None
            draws from fresh entropy. It goes through
            :func:`numpy.random.default_rng`, as SciPy's ``rng`` argument
            does. The same seed gives the same draw for the same form of
            covariance; a Kronecker covariance on a subset of its grid uses
            one normal for each point of the whole grid, so changing a
            covariance's form changes the draw.

        Returns
        -------
        Vector
            The draw, on the covariance's space.

        Raises
        ------
        numpy.linalg.LinAlgError
            If the covariance has a clearly negative eigenvalue (not
            positive semi-definite).

        See Also
        --------
        sqrt : The root :math:`L`, as an operator.

        Examples
        --------
        >>> import pandas as pd
        >>> from fips import Covariance, Space
        >>> space = Space({"flux": pd.Index(["a", "b", "c"], name="cell")})
        >>> S = Covariance.diagonal([4.0, 1.0, 0.25], space)
        >>> S.sample(rng=0).to_series().round(2)
        block  cell
        flux   a       0.25
               b      -0.13
               c       0.32
        dtype: float64
        >>> S.sample(rng=0).values.tolist() == S.sample(rng=0).values.tolist()
        True
        """
        return Vector(self._data._sample(np.random.default_rng(rng)), self._rows)

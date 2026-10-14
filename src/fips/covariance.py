"""
Covariances, and the error components that build them.

A :class:`Covariance` is a symmetric :class:`~fips.Operator` from a space to
itself, kept in its cheapest form, with the variances, solves,
log-determinant, quadratic form and square root a covariance needs.

The error components (:class:`DiagonalError`, :class:`BlockDecayError`,
:class:`KroneckerError`, summed by :class:`CovarianceBuilder`) build a
covariance's matrix from the labels of its rows, as a DataFrame that
:meth:`Covariance.from_frame <fips.Operator.from_frame>` reads.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from joblib import Parallel, delayed

from fips.field import Field
from fips.kronecker import KroneckerOperator
from fips.linalg import (
    BlockDiagonalOperator,
    DiagonalOperator,
    LinearOperator,
    LinearOperatorLike,
    as_linear_operator,
)
from fips.operator import Operator
from fips.space import Space

logger = logging.getLogger(__name__)


def _joined(spaces: list[Space]) -> Space:
    """Return the space of the blocks of `spaces`, one after the other."""
    blocks: dict[str, pd.Index] = {}
    for space in spaces:
        for name in space.names:
            if name in blocks:
                raise ValueError(f"Two blocks are named {name!r}.")
            blocks[name] = space.select(name)
    return Space(blocks)


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
        The matrix, square, rows and columns in the order of `space`.
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
    Operator.from_frame : A covariance from a labeled DataFrame.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from fips import Covariance, Field, Space
    >>> space = Space({"flux": pd.Index(["a", "b"], name="cell")})
    >>> S = Covariance(np.array([[2.0, 1.0], [1.0, 2.0]]), space)
    >>> S.variances.values
    array([2., 2.])
    >>> S.solve(Field([3.0, 3.0], space)).values
    array([1., 1.])
    """

    def __init__(
        self,
        data: LinearOperatorLike,
        space: Space,
        name: str | None = None,
    ):
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
        values: "float | npt.ArrayLike | Field",
        space: Space | None = None,
        name: str | None = None,
    ) -> Self:
        """
        Build a diagonal covariance from its variances.

        The covariance is kept as its diagonal, so it costs one number per
        label whatever the size.

        Parameters
        ----------
        values : float, 1D array-like or Field
            The variances. A scalar is the same for every label; a
            :class:`~fips.Field` brings its own space.
        space : Space, optional
            The labels. Required for a scalar or an array; for a Field it
            must be left out, or have exactly the field's labels (the
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
        if isinstance(values, Field):
            field = values if space is None else values.align(space)
            return cls(DiagonalOperator(field.values), field.space, name=name)
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
    def block_diag(cls, *covariances: "Covariance", name: str | None = None) -> Self:
        """
        Put independent covariances side by side: each on its own blocks, zero between.

        Each covariance keeps its form (a Kronecker block stays factors, a
        diagonal one its variances), so the result costs what its parts do.

        Parameters
        ----------
        *covariances : Covariance
            The covariances, in order. Their blocks must have different names.
        name : str, optional
            The covariance's name.

        Returns
        -------
        Covariance
            The covariance on the blocks of every part, one after the other.

        Raises
        ------
        ValueError
            If no covariance is given, or two have a block of the same name.

        Examples
        --------
        >>> import pandas as pd
        >>> from fips import Covariance, Space
        >>> flux = Covariance.diagonal(
        ...     4.0, Space({"flux": pd.Index(["a", "b"], name="cell")})
        ... )
        >>> bias = Covariance.diagonal(
        ...     1.0, Space({"bias": pd.Index(["A"], name="site")})
        ... )
        >>> S = Covariance.block_diag(flux, bias)
        >>> S.space
        Space(flux: 2 x ['cell'], bias: 1 x ['site'])
        >>> S.variances.values
        array([4., 4., 1.])
        """
        if not covariances:
            raise ValueError("block_diag needs at least one covariance.")
        for cov in covariances:
            if not isinstance(cov, Covariance):
                raise TypeError(
                    f"block_diag takes Covariances, got {type(cov).__name__}."
                )
        space = _joined([cov.space for cov in covariances])
        forms = [cov.data for cov in covariances]
        if len(forms) == 1:
            return cls(forms[0], space, name=name)
        diagonals = [form for form in forms if isinstance(form, DiagonalOperator)]
        if len(diagonals) == len(forms):
            values = np.concatenate([form.values for form in diagonals])
            return cls(DiagonalOperator(values), space, name=name)
        return cls(BlockDiagonalOperator(forms), space, name=name)

    @property
    def variances(self) -> Field:
        """
        The variances: the diagonal, as a field on the covariance's space.

        Read from the form without forming the matrix (from the variances of
        a diagonal covariance, the factors of a Kronecker one).
        """
        return Field(self._data.diagonal(), self._rows, name="variance")

    def solve(self, b: "Field | Operator") -> "Field | Operator":
        r"""
        Return :math:`S^{-1} b`, for a positive-definite covariance :math:`S`.

        Parameters
        ----------
        b : Field or Operator
            A field on the covariance's space, or an operator whose rows are
            on it (both in any order).

        Returns
        -------
        Field or Operator
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
        if isinstance(b, Field):
            return Field(self._data.solve(b.align(self._rows).values), self._rows)
        if isinstance(b, Operator):
            rhs = b._on(self._rows, b.cols)
            return Operator(self._data.solve(rhs), self._rows, b.cols)
        raise TypeError(
            f"Covariance.solve takes a Field or an Operator, not {type(b).__name__}: "
            "an array has no labels to check. Use cov.data.solve(b) to solve "
            "by position."
        )

    def logdet(self) -> float:
        """
        Return the logarithm of the determinant.

        Returns
        -------
        float
            ``log(det(S))``, from the form's factors (see
            :meth:`fips.linalg.LinearOperator.logdet`).

        Raises
        ------
        numpy.linalg.LinAlgError
            If the covariance is not positive-definite.
        """
        return self._data.logdet()

    def quadratic_form(self, x: Field) -> float:
        r"""
        Return :math:`x^T S^{-1} x`, the squared Mahalanobis length of `x`.

        Parameters
        ----------
        x : Field
            A field on the covariance's space, in any order.

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
        >>> from fips import Covariance, Field, Space
        >>> space = Space({"co2": pd.Index(["a", "b"], name="site")})
        >>> S = Covariance.diagonal([4.0, 1.0], space)
        >>> S.quadratic_form(Field([2.0, 1.0], space))
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


def _kronecker_marginals(
    index: pd.MultiIndex, marginal_kernels: list[tuple[str | list[str], Callable]]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Return each marginal's correlation matrix and each row's position in it.

    For each kernel, the unique coordinates of its levels are numbered in order
    of first appearance in `index`; the kernel gets them as a DataFrame (the
    rows of their first appearance) and returns their correlation matrix, and
    each row of the index gets the number of its coordinates. This is the
    position mapping `KroneckerError.build` uses (it calls this), so that
    ``KroneckerOperator(factors, std, positions).dense()`` is that ``build``.

    Raises
    ------
    ValueError
        If no marginal kernels are given, if two rows of the index have the
        same coordinates on every kernel's levels, or if a kernel does not
        return a square matrix with one row per unique coordinate.
    """
    if not marginal_kernels:
        raise ValueError("KroneckerError needs at least one marginal kernel.")
    coords = index.to_frame(index=False)
    dims_list = [
        [dims] if isinstance(dims, str) else list(dims) for dims, _ in marginal_kernels
    ]
    kernel_dims = list(dict.fromkeys(d for dims in dims_list for d in dims))
    n_repeated = int(coords.duplicated(subset=kernel_dims).sum())
    if n_repeated:
        raise ValueError(
            f"KroneckerError needs each row of the index to have its own "
            f"combination of the kernels' coordinates {kernel_dims}, but "
            f"{n_repeated} rows repeat one. Rows that differ only on another "
            f"level would be perfectly correlated."
        )
    factors, positions = [], []
    for dims, (_, kernel_func) in zip(dims_list, marginal_kernels, strict=True):
        pos = coords.groupby(dims, sort=False, dropna=False).ngroup().to_numpy()
        first = np.unique(pos, return_index=True)[1]
        C = np.asarray(kernel_func(coords[dims].iloc[first]), dtype=float)
        if C.shape != (len(first), len(first)):
            raise ValueError(
                f"The kernel for {dims} returned a matrix of shape {C.shape} "
                f"for {len(first)} unique coordinates; expected "
                f"({len(first)}, {len(first)})."
            )
        factors.append(C)
        positions.append(pos)
    return factors, positions


class ErrorComponent(ABC):
    """
    Abstract base class for error components in covariance matrices.
    """

    name: str
    """Name of the error component."""
    variances: float | pd.Series
    """Variances for the diagonal of the covariance matrix. Can be a single float or a Series indexed by the same MultiIndex as the covariance matrix."""

    def __init__(self, name: str, variances: float | pd.Series):
        self.name = name
        self.variances = variances

    def _align_variances(self, index: pd.MultiIndex) -> pd.Series:
        """Return the variances reindexed onto ``index``, filling gaps with zeros."""
        if isinstance(self.variances, pd.Series):
            if self.variances.index.equals(index):
                return self.variances.fillna(0.0)

            # Check if the series has names for all its levels
            target_levels = self.variances.index.names
            if any(lvl is None for lvl in target_levels):
                raise ValueError(
                    "All levels in the variance Series index must be named."
                )

            # Check if the target index has any unnamed levels
            if any(name is None for name in index.names):
                raise ValueError(
                    "All levels in the target index must be named when aligning variances."
                )

            # Ensure the Series has a name for join operation
            variances = self.variances
            if variances.name is None:
                variances = variances.copy()
                variances.name = "variances"

            # Convert the target index to a dataframe just to use its join capabilities
            target_df = pd.DataFrame(index=index)
            # pandas-stubs take only str names in join(on=), though a level name
            # is any Hashable
            # pyrefly: ignore[bad-argument-type]
            aligned = target_df.join(variances, on=target_levels)
            return aligned[variances.name].fillna(0.0)

        return pd.Series(self.variances, index=index)

    @abstractmethod
    def build(self, index: pd.MultiIndex, **kwargs) -> pd.DataFrame:
        """Must return a pd.DataFrame with the given index on both rows and columns."""
        pass

    def __add__(self, other):
        """Add error components to create a CovarianceBuilder."""
        if isinstance(other, ErrorComponent):
            # Adding two components creates a new Builder
            return CovarianceBuilder([self, other])
        elif isinstance(other, CovarianceBuilder):
            # Adding a component to a Builder appends it
            other.components.append(self)
            return other
        raise TypeError(f"Cannot add ErrorComponent to {type(other)}")


class CovarianceBuilder:
    """
    Builds covariance matrices by combining multiple error components.
    """

    components: list[ErrorComponent]
    """The list of error components to build and sum."""

    def __init__(self, components: list[ErrorComponent]):
        self.components = components

    def build(self, index: pd.MultiIndex, **kwargs) -> pd.DataFrame:
        """
        Build and sum all error components into a single DataFrame.

        Parameters
        ----------
        index : pd.MultiIndex
            The state/observation index to build the covariance over.
        **kwargs
            Passed to each component's ``build``.

        Returns
        -------
        pd.DataFrame
            The dense covariance, labeled by `index` on both axes. Label it
            with :meth:`Covariance.from_frame <fips.Operator.from_frame>`; a
            covariance that is mostly zeros is stored sparse by
            ``Covariance(scipy.sparse.csr_matrix(frame.to_numpy()), space)``.
        """
        if not self.components:
            raise ValueError("No components provided to build.")

        S = np.add.reduce(
            [c.build(index, **kwargs).to_numpy(dtype=float) for c in self.components]
        )

        return pd.DataFrame(S, index=index, columns=index)

    def __add__(self, other):
        """Add error components or builders together."""
        if isinstance(other, ErrorComponent):
            return CovarianceBuilder(self.components + [other])
        elif isinstance(other, CovarianceBuilder):
            return CovarianceBuilder(self.components + other.components)
        raise TypeError(f"Cannot add CovarianceBuilder to {type(other)}")

    def __radd__(self, other):
        """Right-hand addition for error components."""
        if isinstance(other, ErrorComponent):
            return CovarianceBuilder([other] + self.components)
        raise TypeError(f"Cannot add {type(other)} to CovarianceBuilder")

    def __repr__(self) -> str:
        """Return string representation."""
        names = [c.name for c in self.components]
        return f"CovarianceBuilder(n_components={len(self.components)}, components={names})"


class DiagonalError(ErrorComponent):
    """
    Error component that produces diagonal covariance matrices.
    """

    def build(self, index: pd.MultiIndex, **kwargs) -> pd.DataFrame:
        """Build diagonal covariance matrix."""
        variances = self._align_variances(index).to_numpy(dtype=float)
        return pd.DataFrame(np.diag(variances), index=index, columns=index)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"DiagonalError(name='{self.name}')"


class BlockDecayError(ErrorComponent):
    """
    Error component with block-structured correlation decay.
    """

    groupers: list
    """List of groupers to use for block structure."""

    corr_func: Callable
    """Function that takes a DataFrame group and returns a correlation matrix for that block."""

    def __init__(
        self,
        name: str,
        variances: float | pd.Series,
        groupers: list,
        corr_func: Callable,
    ):
        super().__init__(name, variances)
        self.groupers = groupers
        self.corr_func = corr_func

    def _compute_block(self, group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Compute block correlation matrix for a group."""
        idx = group.index.to_numpy()
        return idx, self.corr_func(group)

    def build(self, index: pd.MultiIndex, n_jobs: int = 1, **kwargs) -> pd.DataFrame:
        """Build block-structured covariance matrix with correlation decay."""
        variances = self._align_variances(index)
        N = len(index)
        corr_matrix = np.eye(N)

        coords = index.to_frame(index=False)

        # Parallelize the loop over groups using joblib
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._compute_block)(group)
            for _, group in coords.groupby(self.groupers)
        )

        for idx, block in results or []:
            corr_matrix[np.ix_(idx, idx)] = block

        std_dev = np.sqrt(variances.to_numpy())
        cov_matrix = std_dev[:, None] * corr_matrix * std_dev[None, :]

        return pd.DataFrame(cov_matrix, index=index, columns=index)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"BlockDecayError(name='{self.name}', groupers={self.groupers})"


class KroneckerError(ErrorComponent):
    """
    Builds a covariance matrix from a product of marginal correlation matrices.

    Each marginal kernel gives the correlation along some of the index's levels
    (time, or ``lat`` and ``lon`` together), and the correlation between two
    elements is the product of their marginal correlations. On the full product
    of the marginal coordinates, in the kernels' order, this is the Kronecker
    product of the marginal correlation matrices.

    Parameters
    ----------
    name : str
        Name of the error component.
    variances : float or pd.Series
        Variances for the diagonal of the covariance matrix.
        Can be a single float or a Series indexed by the same MultiIndex as the covariance matrix.
    marginal_kernels : list of tuples
        Each tuple contains:

        1. The dimension name(s) as a string or list of strings.
        2. The Callable kernel function that takes a DataFrame of those
           unique coordinates and returns a 2D correlation matrix.

    Notes
    -----
    The index can hold its rows in any order, and any subset of the product of
    the marginal coordinates: a grid with some cells left out, at every time or
    at only some. Entry ``(i, j)`` of the result is
    ``std[i] * std[j] * C_1[i, j] * C_2[i, j] * ...``, where ``C_k[i, j]`` is
    the correlation the k-th kernel gives between the coordinates of rows i and
    j on that kernel's levels. Each kernel receives the unique coordinates of
    its levels in the order they first appear in the index. Every row must have
    its own combination of the kernels' coordinates: two rows that differ only
    on a level no kernel names (two blocks on the same grid, say) would be
    perfectly correlated, so `build` raises a `ValueError` instead.
    """

    marginal_kernels: list[tuple[str | list[str], Callable]]
    """List of marginal kernels, each defined by a tuple of dimension name(s) and a kernel function."""

    def __init__(
        self,
        name: str,
        variances: float | pd.Series,
        marginal_kernels: list[tuple[str | list[str], Callable]],
    ):
        super().__init__(name, variances)
        self.marginal_kernels = marginal_kernels

    def build(self, index: pd.MultiIndex, **kwargs) -> pd.DataFrame:
        """
        Build covariance matrix using product of marginal kernels.

        Parameters
        ----------
        index : pd.MultiIndex
            The MultiIndex of the covariance matrix. It must contain the
            levels named in ``marginal_kernels``; its rows can be in any order.
        **kwargs
            Additional keyword arguments that may be needed by the kernel functions.

        Returns
        -------
        pd.DataFrame
            The constructed covariance matrix with the given index on both rows and columns.

        Raises
        ------
        ValueError
            If no marginal kernels were given, if two rows of the index have the
            same coordinates on every kernel's levels, or if a kernel does not
            return a square matrix with one row per unique coordinate.
        """
        factors, positions = _kronecker_marginals(index, self.marginal_kernels)
        variances = self._align_variances(index)

        # corr[i, j] = prod_k C_k[pos_k[i], pos_k[j]], for any row order and any
        # subset of the product. On the full product in the kernels' order it
        # equals the Kronecker product of the C_k.
        corr_matrix = factors[0].take(positions[0], axis=0).take(positions[0], axis=1)
        for C, pos in zip(factors[1:], positions[1:], strict=True):
            corr_matrix *= C.take(pos, axis=0).take(pos, axis=1)

        # Scale by variances
        std_dev = np.sqrt(variances.to_numpy())
        corr_matrix *= std_dev[:, None]
        corr_matrix *= std_dev[None, :]

        return pd.DataFrame(corr_matrix, index=index, columns=index)

    def operator(self, index: pd.MultiIndex) -> KroneckerOperator:
        """
        Return the covariance on `index` as Kronecker factors, without forming it.

        The rows can be in any order and any subset of the product of the
        marginal coordinates. Entry ``(i, j)`` is ``std[i] * std[j] *
        C_1[i, j] * C_2[i, j] * ...``, where ``C_k[i, j]`` is the correlation the
        k-th kernel gives between the coordinates of rows i and j. On an index
        that is the product of the marginal coordinates in the kernels' order,
        its :meth:`~KroneckerOperator.dense` is :meth:`build`, number for
        number.

        Parameters
        ----------
        index : pd.MultiIndex
            The index of the covariance. It must contain the levels named in
            ``marginal_kernels``.

        Returns
        -------
        KroneckerOperator
            The factors (each kernel's matrix over the unique coordinates of
            its levels, in order of first appearance in `index`), each row's
            position in them, and the standard deviations.

        Raises
        ------
        ValueError
            If no marginal kernels were given, if two rows of the index have the
            same coordinates on every kernel's levels, or if a kernel does not
            return a square matrix with one row per unique coordinate.

        See Also
        --------
        fips.Covariance : Label it, to use as a prior error.

        Examples
        --------
        >>> import pandas as pd
        >>> from fips import Covariance, Space
        >>> from fips.covariance import KroneckerError
        >>> from fips.kernels import GridSpatialDecay, GridTimeDecay
        >>> index = pd.MultiIndex.from_product(
        ...     [
        ...         ["flux"],
        ...         pd.date_range("2024-01-01", periods=3),
        ...         [40.0, 40.5],
        ...         [-112.0],
        ...     ],
        ...     names=["block", "time", "lat", "lon"],
        ... )
        >>> error = KroneckerError(
        ...     "prior_error",
        ...     0.25,
        ...     [
        ...         ("time", GridTimeDecay("2D")),
        ...         (["lat", "lon"], GridSpatialDecay("lat", "lon", 50.0)),
        ...     ],
        ... )
        >>> S_0 = Covariance(error.operator(index), Space.from_index(index))
        >>> S_0.data
        <6x6 KroneckerOperator with dtype=float64>
        """
        factors, positions = _kronecker_marginals(index, self.marginal_kernels)
        std = np.sqrt(self._align_variances(index).to_numpy())
        return KroneckerOperator(factors, std=std, positions=positions)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"KroneckerError(name='{self.name}', "
            f"n_marginals={len(self.marginal_kernels)})"
        )

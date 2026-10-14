"""
Covariance matrix structures and generation.

This module provides the `CovarianceMatrix` class and related utilities for
representing, building, and manipulating error covariance matrices used in
Bayesian estimation.
"""

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from functools import cached_property
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from joblib import Parallel, delayed
from numpy.linalg import LinAlgError
from scipy import sparse
from scipy.sparse import csr_matrix

from fips._sparse import frame_from_spmatrix
from fips.base import diagonal_frame
from fips.matrix import Matrix
from fips.operator import DenseOperator, Operator, _as_array, _solve_psd
from fips.vector import Vector

logger = logging.getLogger(__name__)


class CovarianceMatrix(Matrix):
    """
    Represents a symmetric Covariance Matrix.

    Covariance matrices are used to represent error covariances in the inversion framework.
    They can be constructed from variances and correlation matrices.
    """

    @property
    def variances(self) -> Vector:
        """
        The variances (diagonal elements) of the covariance matrix.

        Read from the stored entries of a sparse covariance, without forming
        the dense matrix.
        """
        return Vector(self.operator.diagonal(), index=self.index, name="variance")

    def force_symmetry(self, keep: str = "lower") -> "CovarianceMatrix":
        """
        Force the matrix to be perfectly symmetric by copying one triangle to the other.

        Useful for eliminating floating-point asymmetry.

        Parameters
        ----------
        keep : {'lower', 'upper'}, default 'lower'
            Which triangle of the matrix to preserve and copy.

        Returns
        -------
        CovarianceMatrix
            A new, perfectly symmetric covariance matrix, sparse if this one is.
        """
        if keep not in ("lower", "upper"):
            raise ValueError("keep must be 'lower' or 'upper'")

        if self.is_sparse:
            # The kept triangle, diagonal included, plus the transpose of the
            # same triangle without its diagonal
            mat = self.values
            if keep == "lower":
                sym = sparse.tril(mat) + sparse.tril(mat, k=-1).T
            else:
                sym = sparse.triu(mat) + sparse.triu(mat, k=1).T
            frame = frame_from_spmatrix(sparse.csr_matrix(sym), self.index, self.index)
            return type(self)(frame, name=self.name)

        mat = self.values.copy()

        if keep == "lower":
            # Extract lower triangle and add its transpose (minus the diagonal so it doesn't double)
            lower = np.tril(mat)
            sym_mat = lower + lower.T - np.diag(np.diag(lower))
        else:
            upper = np.triu(mat)
            sym_mat = upper + upper.T - np.diag(np.diag(upper))

        return type(self)(sym_mat, index=self.index, name=self.name)


_CHUNK = 2**24
"""
How many entries of the product grid a :class:`KroneckerOperator` product holds at once.

A product with many columns (``H @ S_0`` has one per observation) runs over
blocks of columns, so its work arrays stay near 128 MB (float64) whatever the
number of columns.
"""


class KroneckerOperator(Operator):
    r"""
    A covariance kept as the Kronecker product of its marginal correlations.

    .. math::

        S = D \, P \, (C_1 \otimes C_2 \otimes \dots \otimes C_K) \, P^T D

    Each factor :math:`C_k` is the correlation matrix of one marginal (the
    times, or the grid cells), :math:`P` picks the rows of the full product
    that the operator has, in its order (any subset of the product, in any
    order), and :math:`D` is the diagonal of standard deviations. Entry
    ``(i, j)`` is ``std[i] * std[j] * prod_k C_k[pos_k[i], pos_k[j]]``.

    The ``n x n`` matrix is never formed unless :meth:`dense` is called. A
    product with ``p`` columns costs ``N * (N_1 + ... + N_K)`` per column,
    where ``N_k`` is the size of factor k and ``N`` their product, and holds
    work arrays of about ``N`` entries per column. That is less than the dense
    matrix as long as the rows fill most of the product.

    :meth:`~fips.covariance.KroneckerError.operator` builds one from marginal
    kernels and an index, and :meth:`~fips.CovarianceMatrix.from_operator`
    puts it behind labels.

    Parameters
    ----------
    factors : sequence of 2D array-like
        The marginal correlation matrices :math:`C_k`, square, symmetric and
        positive-definite (positive-definite only for :meth:`solve`).
    std : float or 1D array-like, default 1.0
        The standard deviation of each row; a scalar is the same for every row.
    positions : sequence of 1D array-like of int, optional
        For each factor, the position of each row in that factor (``pos_k``).
        If not given, the rows are the full product in order, the first
        factor varying slowest, as ``np.kron`` lays it out.

    Raises
    ------
    ValueError
        If there is no factor, a factor is not square or not finite, the
        positions do not match the factors (one array per factor, all the same
        length, each within its factor), two rows have the same positions, or
        a standard deviation is negative or not finite.

    See Also
    --------
    KroneckerError.operator : The operator of a Kronecker error component on an index.
    fips.CovarianceMatrix.from_operator : A labeled covariance that holds an operator.

    Examples
    --------
    >>> import numpy as np
    >>> from fips.covariance import KroneckerOperator
    >>> C_time = np.array([[1.0, 0.5], [0.5, 1.0]])
    >>> C_space = np.array([[1.0, 0.2, 0.0], [0.2, 1.0, 0.2], [0.0, 0.2, 1.0]])
    >>> S = KroneckerOperator([C_time, C_space], std=2.0)
    >>> S.shape
    (6, 6)
    >>> np.allclose(S.dense(), 4.0 * np.kron(C_time, C_space))
    True
    >>> x = np.arange(6.0)
    >>> np.allclose(S.solve(S @ x), x)
    True
    """

    factors: tuple[np.ndarray, ...]
    """The marginal correlation matrices, one per marginal."""
    positions: tuple[np.ndarray, ...]
    """For each factor, the position of each row in it (``pos_k``)."""
    std: np.ndarray
    """The standard deviation of each row."""

    def __init__(
        self,
        factors: Sequence[npt.ArrayLike],
        std: npt.ArrayLike = 1.0,
        positions: Sequence[npt.ArrayLike] | None = None,
    ):
        mats = tuple(np.asarray(C, dtype=float) for C in factors)
        if not mats:
            raise ValueError("A KroneckerOperator needs at least one factor.")
        for C in mats:
            if C.ndim != 2 or C.shape[0] != C.shape[1]:
                raise ValueError(f"Each factor must be square; got shape {C.shape}.")
            if not np.isfinite(C).all():
                raise ValueError("The factors must be finite.")
        sizes = tuple(len(C) for C in mats)
        size = math.prod(sizes)

        if positions is None:
            pos = tuple(np.unravel_index(np.arange(size), sizes))
        else:
            pos = tuple(np.asarray(p) for p in positions)
            if len(pos) != len(mats):
                raise ValueError(
                    f"Got {len(pos)} arrays of positions for {len(mats)} factors."
                )
            if any(p.ndim != 1 or len(p) != len(pos[0]) for p in pos):
                raise ValueError("The positions must be 1D arrays of one length.")
            if any(not np.issubdtype(p.dtype, np.integer) for p in pos):
                raise ValueError("The positions must be integers.")
            for p, n_k in zip(pos, sizes, strict=True):
                if len(p) and (p.min() < 0 or p.max() >= n_k):
                    raise ValueError(
                        f"A position is outside its factor, which has {n_k} rows."
                    )
        rows = np.ravel_multi_index(pos, sizes)
        if len(np.unique(rows)) != len(rows):
            raise ValueError(
                "Two rows have the same position in every factor; they would be "
                "perfectly correlated."
            )

        n = len(rows)
        std = np.asarray(std, dtype=float)
        if std.ndim == 0:
            std = np.full(n, float(std))
        if std.shape != (n,):
            raise ValueError(f"std must be a scalar or have {n} entries.")
        if not np.isfinite(std).all() or (std < 0).any():
            raise ValueError("The standard deviations must be finite and not negative.")

        self.factors = mats
        self.positions = tuple(p.astype(np.intp, copy=False) for p in pos)
        self.std = std
        self._sizes = sizes
        # The rows' places in the product grid, or None when they are the
        # whole grid in order (no scatter or gather needed)
        self._rows = None if np.array_equal(rows, np.arange(size)) else rows
        super().__init__((n, n))

    @property
    def _size(self) -> int:
        """The number of entries of the full product, ``N``."""
        return math.prod(self._sizes)

    def _scatter(self, Y: np.ndarray) -> np.ndarray:
        """Put the rows of `Y` at their places in the product grid, zeros elsewhere."""
        if self._rows is None:
            return np.ascontiguousarray(Y)
        Z = np.zeros((self._size, Y.shape[1]))
        Z[self._rows] = Y
        return Z

    def _gather(self, Z: np.ndarray) -> np.ndarray:
        """Take the operator's rows out of the product grid `Z`."""
        return Z if self._rows is None else Z[self._rows]

    def _through(self, Z: np.ndarray, matrices) -> np.ndarray:
        """
        Return ``(M_1 kron ... kron M_K) Z`` for `Z` on the product grid.

        Each matrix is applied along its own axis of the grid, never their
        Kronecker product.
        """
        q = Z.shape[1]
        for k, M in enumerate(matrices):
            outer = math.prod(self._sizes[:k])
            Z = np.matmul(M, Z.reshape(outer, self._sizes[k], -1))
        return Z.reshape(-1, q)

    def _by_columns(self, X, func) -> np.ndarray:
        """Return `func` applied to `X` (an array or a SciPy sparse matrix), a block of columns at a time."""
        if sparse.issparse(X):
            X = sparse.csc_array(X)
        out = np.empty((self.shape[0], X.shape[1]))
        step = max(1, _CHUNK // self._size)
        for j in range(0, X.shape[1], step):
            block = X[:, j : j + step]
            block = block.toarray() if sparse.issparse(block) else np.asarray(block)
            out[:, j : j + step] = func(block.astype(float, copy=False))
        return out

    def _matmat(self, X):
        """Return ``self @ X`` through the factors."""
        s = self.std[:, None]

        def apply(Y):
            """Return ``S Y`` for a block of columns."""
            return s * self._gather(self._through(self._scatter(s * Y), self.factors))

        return self._by_columns(X, apply)

    def _transpose(self):
        """Return the transpose: the same rows, each factor transposed."""
        return KroneckerOperator(
            [C.T for C in self.factors], self.std, positions=self.positions
        )

    def _elementwise(self, func):
        """Return the operator times a scalar; a positive scalar keeps the factors."""
        scale = float(func(1.0))
        if scale > 0:
            return KroneckerOperator(
                self.factors, self.std * np.sqrt(scale), positions=self.positions
            )
        return DenseOperator(func(self.dense()))

    def _select(self, rows, cols):
        """Return the entries at `rows` and `cols`; the same rows and columns keep the factors."""
        if np.array_equal(rows, cols):
            return KroneckerOperator(
                self.factors,
                self.std[rows],
                positions=[p[rows] for p in self.positions],
            )
        return super()._select(rows, cols)

    def dense(self) -> np.ndarray:
        """
        Return the operator as a dense NumPy array.

        Entry ``(i, j)`` is ``std[i] * std[j] * prod_k C_k[pos_k[i], pos_k[j]]``,
        computed as :meth:`KroneckerError.build` computes it, so the two give the
        same numbers. This forms the ``n x n`` matrix.

        Returns
        -------
        np.ndarray
            The full matrix.
        """
        C, p = self.factors[0], self.positions[0]
        corr = C.take(p, axis=0).take(p, axis=1)
        for C, p in zip(self.factors[1:], self.positions[1:], strict=True):
            corr *= C.take(p, axis=0).take(p, axis=1)
        corr *= self.std[:, None]
        corr *= self.std[None, :]
        return corr

    def diagonal(self) -> np.ndarray:
        """
        Return the main diagonal, ``std**2 * prod_k diag(C_k)[pos_k]``.

        Returns
        -------
        np.ndarray
            The variances, without forming the matrix.
        """
        d = np.diagonal(self.factors[0])[self.positions[0]].copy()
        for C, p in zip(self.factors[1:], self.positions[1:], strict=True):
            d *= np.diagonal(C)[p]
        d *= self.std
        d *= self.std
        return d

    @cached_property
    def _inverses(self) -> tuple[np.ndarray, ...]:
        """The inverse of each factor, each factored once (Cholesky, or ``LDL^T`` with a warning)."""
        return tuple(_solve_psd(C, np.eye(len(C))) for C in self.factors)

    @cached_property
    def _complement(self) -> tuple[np.ndarray, np.ndarray]:
        """
        The product grid's missing entries, and the inverse's columns at them.

        For rows that are a strict subset of the product: ``W = G[:, missing]``,
        where ``G`` is the inverse of the full product's correlation.
        """
        kept = np.arange(self._size) if self._rows is None else self._rows
        missing = np.setdiff1d(np.arange(self._size), kept)
        E = np.zeros((self._size, len(missing)))
        E[missing, np.arange(len(missing))] = 1.0
        return missing, self._through(E, self._inverses)

    @cached_property
    def _norm(self) -> float:
        """An upper bound on the 2-norm of the rows' correlation: the product of the factors' 2-norms."""
        return math.prod(float(np.linalg.norm(C, 2)) for C in self.factors)

    def solve(self, B) -> np.ndarray:
        r"""
        Solve ``self @ X = B`` through the factors.

        Each factor is factored once (Cholesky, or a symmetric ``LDL^T``
        with a logged warning when roundoff has tipped it off
        positive-definite) and the factorization is kept. When the rows are
        the whole product, in any order, the inverse is
        :math:`D^{-1} P (C_1^{-1} \otimes \dots \otimes C_K^{-1}) P^T D^{-1}`.

        When the rows are a strict subset of the product, with ``d`` entries
        of the product missing, the inverse of the kept rows' correlation is
        the Schur complement :math:`G_{kk} - G_{kd} G_{dd}^{-1} G_{dk}` of
        the full product's inverse :math:`G`, at the cost of ``d`` products
        with the factors' inverses and one ``d x d`` solve. That is the
        inverse in exact arithmetic, but in floating point it loses accuracy
        when the factors are ill-conditioned (the complement subtracts large
        terms of :math:`G`). So the solve checks its result: when the
        backward error of any column, :math:`\|R y - c\| / (\|R\| \|y\| +
        \|c\|)` for the rows' correlation :math:`R`, is larger than
        ``n * eps`` (the bound a dense Cholesky solve meets), it logs a
        warning and solves the dense matrix instead. When the complement
        would hold more than the dense matrix (``N d > n^2``), the dense
        matrix is solved from the start.

        Parameters
        ----------
        B : array-like, SciPy sparse matrix or Operator
            The right-hand side: a vector, or a matrix with as many rows as the
            operator.

        Returns
        -------
        np.ndarray
            ``X``, dense, with the shape of `B`.

        Raises
        ------
        numpy.linalg.LinAlgError
            If a standard deviation is zero: the operator is singular.
        """
        if not np.all(self.std):
            raise LinAlgError(
                "A standard deviation is zero, so the operator is singular."
            )
        B = _as_array(B)
        n = self.shape[0]
        if self._size * (self._size - n) > n * n:
            logger.info(
                "Solving the %d x %d Kronecker covariance densely: its rows fill "
                "too little of the product for the factored solve.",
                n,
                n,
            )
            return _solve_psd(self.dense(), B)

        inv_s = 1.0 / self.std[:, None]
        tol = n * np.finfo(float).eps
        worst = 0.0  # the largest backward error of the Schur complement

        def apply(Y):
            """Return ``S^-1 Y`` for a block of columns."""
            nonlocal worst
            c = inv_s * Y  # the right-hand side in the correlation's units
            Z = self._through(self._scatter(c), self._inverses)
            y = self._gather(Z)
            if n < self._size:  # a strict subset: the Schur complement
                missing, W = self._complement
                y = y - W[self._rows] @ _solve_psd(W[missing], Z[missing])
                Ry = self._gather(self._through(self._scatter(y), self.factors))
                scale = self._norm * np.linalg.norm(y, axis=0) + np.linalg.norm(
                    c, axis=0
                )
                error = np.linalg.norm(Ry - c, axis=0)
                worst = max(worst, float(np.max(error / np.maximum(scale, 1e-300))))
            return inv_s * y

        X = self._by_columns(B[:, None] if B.ndim == 1 else B, apply)
        if worst > tol:
            logger.warning(
                "The factored solve of the %d x %d Kronecker covariance lost "
                "accuracy (backward error %.1e, more than %.1e), likely from "
                "ill-conditioned factors; solving the dense matrix instead.",
                n,
                n,
                worst,
                tol,
            )
            return _solve_psd(self.dense(), B)
        return X[:, 0] if B.ndim == 1 else X

    def __getstate__(self) -> dict[str, Any]:
        """Return the state for pickling, without the cached inverses."""
        state = self.__dict__.copy()
        state.pop("_inverses", None)
        state.pop("_complement", None)
        state.pop("_norm", None)
        return state


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

    def build(
        self, index: pd.MultiIndex, sparse: bool = False, **kwargs
    ) -> pd.DataFrame:
        """
        Build and sum all error components into a single DataFrame.

        Parameters
        ----------
        index : pd.MultiIndex
            The state/observation index to build the covariance over.
        sparse : bool, default False
            If True, return a pandas sparse DataFrame.  Apply threshold zeroing
            in each ErrorComponent.build() before using this option.
        """
        if not self.components:
            raise ValueError("No components provided to build.")

        S = np.add.reduce(
            [c.build(index, **kwargs).to_numpy(dtype=float) for c in self.components]
        )

        if sparse:
            return frame_from_spmatrix(csr_matrix(S), index=index, columns=index)
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
        variances = self._align_variances(index)
        return diagonal_frame(variances.to_numpy(), index)

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
        fips.CovarianceMatrix.from_operator : Label it, to use as a prior error.

        Examples
        --------
        >>> import pandas as pd
        >>> from fips import CovarianceMatrix
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
        >>> S_0 = CovarianceMatrix.from_operator(error.operator(index), index)
        >>> S_0.operator
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

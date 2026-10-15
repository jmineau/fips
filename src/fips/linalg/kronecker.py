"""
Kronecker covariances: a covariance kept as the product of its marginal correlations.

A covariance over a grid of times and places is often the product of a
correlation in time and one in space. :class:`KroneckerOperator` keeps the
two (or more) factors and never forms the ``n x n`` matrix: products, solves,
the diagonal and the log-determinant go through the factors. It is one of the
forms of :mod:`fips.linalg`, so a :class:`~fips.Covariance` holds one as it
holds a dense or a diagonal matrix.
"""

import logging
import math
from collections.abc import Sequence
from functools import cached_property
from typing import Any, Self

import numpy as np
import numpy.typing as npt
from numpy.linalg import LinAlgError
from scipy import sparse

from fips.linalg.forms import (
    DenseOperator,
    LinearOperator,
    _as_array,
    _asymmetry,
    _read_only,
    _solve_psd,
)

logger = logging.getLogger(__name__)


_CHUNK = 2**24
"""
How many entries of the product grid a :class:`KroneckerOperator` product holds at once.

A product with many columns (``H @ S_0`` has one per observation) runs over
blocks of columns, so its work arrays stay near 128 MB (float64) whatever the
number of columns.
"""


class KroneckerOperator(LinearOperator):
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

    The operator copies the factors, the standard deviations and the positions
    it is given, and the copies are read-only: changing the arrays afterwards
    leaves the operator as it was.

    A :class:`~fips.Kronecker` model builds one from the labels of a
    space, and :class:`~fips.Covariance` puts it behind them.

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
        If there is no factor, a factor is not square, not finite or not
        symmetric (to 1e-10 of its largest entry), the positions do not match
        the factors (one array per factor, all the same length, each within
        its factor), two rows have the same positions, or a standard deviation
        is negative or not finite.

    See Also
    --------
    fips.Kronecker : The correlation model whose covariance this is.
    fips.Covariance : A labeled covariance, which holds a form.

    Examples
    --------
    >>> import numpy as np
    >>> from fips.linalg import KroneckerOperator
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
    """The marginal correlation matrices, one per marginal, read-only."""
    positions: tuple[np.ndarray, ...]
    """For each factor, the position of each row in it (``pos_k``), read-only."""
    std: np.ndarray
    """The standard deviation of each row, read-only."""

    def __init__(
        self,
        factors: Sequence[npt.ArrayLike],
        std: npt.ArrayLike = 1.0,
        positions: Sequence[npt.ArrayLike] | None = None,
    ):
        self._hold(
            [np.array(C, dtype=float) for C in factors],
            np.array(std, dtype=float),
            None if positions is None else [np.array(p) for p in positions],
        )

    @classmethod
    def _adopt(
        cls,
        factors: Sequence[np.ndarray],
        std: "float | np.ndarray" = 1.0,
        positions: Sequence[np.ndarray] | None = None,
    ) -> Self:
        """Return the operator that holds these arrays themselves, which fips made or are read-only, without copying them."""
        form = cls.__new__(cls)
        form._hold(factors, std, positions)
        return form

    def _hold(
        self,
        factors: Sequence[np.ndarray],
        std: "float | np.ndarray",
        positions: Sequence[np.ndarray] | None,
    ) -> None:
        """Check the factors, the standard deviations and the positions, and hold them, read-only."""
        mats = tuple(np.asarray(C, dtype=float) for C in factors)
        if not mats:
            raise ValueError("A KroneckerOperator needs at least one factor.")
        for k, C in enumerate(mats):
            if C.ndim != 2 or C.shape[0] != C.shape[1]:
                raise ValueError(f"Each factor must be square; got shape {C.shape}.")
            if not np.isfinite(C).all():
                raise ValueError("The factors must be finite.")
            found = _asymmetry(C)
            if found is not None:
                i, j, a_ij, a_ji, _ = found
                raise ValueError(
                    f"Factor {k} is not symmetric: its entry ({i}, {j}) is "
                    f"{a_ij:.6g} and ({j}, {i}) is {a_ji:.6g}."
                )
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
        self._freeze()
        self._sizes = sizes
        # The rows' places in the product grid, or None when they are the
        # whole grid in order (no scatter or gather needed)
        self._rows = None if np.array_equal(rows, np.arange(size)) else rows
        super().__init__((n, n))

    def _freeze(self) -> None:
        """Make the factors, the positions and the standard deviations read-only."""
        for array in (*self.factors, *self.positions, self.std):
            _read_only(array)

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
        return KroneckerOperator._adopt(
            [C.T for C in self.factors], self.std, positions=self.positions
        )

    def _elementwise(self, func):
        """Return the operator times a scalar; a positive scalar keeps the factors."""
        scale = float(func(1.0))
        if scale > 0:
            return KroneckerOperator._adopt(
                self.factors, self.std * np.sqrt(scale), positions=self.positions
            )
        return DenseOperator._adopt(func(self.dense()))

    def _select(self, rows, cols):
        """Return the entries at `rows` and `cols`; the same rows and columns (none of them -1) keep the factors."""
        rows = np.asarray(rows, dtype=np.intp)
        if np.array_equal(rows, cols) and (rows >= 0).all():
            return KroneckerOperator._adopt(
                self.factors,
                self.std[rows],
                positions=[p[rows] for p in self.positions],
            )
        return super()._select(rows, cols)

    def dense(self) -> np.ndarray:
        """
        Return the operator as a dense NumPy array.

        Entry ``(i, j)`` is ``std[i] * std[j] * prod_k C_k[pos_k[i], pos_k[j]]``,
        computed factor by factor, then scaled by the standard deviations.
        This forms the ``n x n`` matrix.

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

    def logdet(self) -> float:
        r"""
        Return the log-determinant through the factors.

        When the rows are the whole product, in any order,
        :math:`\log\det S = 2 \sum_i \log s_i + \sum_k (N / N_k) \log\det C_k`,
        where :math:`s` are the standard deviations, :math:`N_k` the size of
        factor k and :math:`N` their product. For a strict subset of the
        product the dense matrix is factored.

        Returns
        -------
        float
            ``log(det(S))``.

        Raises
        ------
        numpy.linalg.LinAlgError
            If a factor is not positive-definite or a standard deviation is zero.
        """
        if self.shape[0] < self._size:
            return super().logdet()
        if not np.all(self.std):
            raise LinAlgError(
                "A standard deviation is zero, so the operator is singular."
            )
        total = 2.0 * float(np.log(self.std).sum())
        for C, n_k in zip(self.factors, self._sizes, strict=True):
            total += (self._size // n_k) * DenseOperator._adopt(C).logdet()
        return total

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

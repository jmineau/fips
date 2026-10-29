r"""
The analytic method: the closed-form Gaussian posterior of a linear problem.

``_analytic`` factors the problem once and returns a :class:`~fips.Posterior`
that asks the factors for everything it reports. It factors in one of two
forms, chosen by the problem's size, which give the same posterior:

- ``Whitened`` whitens the observations with the model-data mismatch and
  works with an ``m x m`` matrix (``m`` observations);
- ``Information`` whitens the state with the prior error covariance and
  works with an ``n x n`` matrix (``n`` state elements). It applies the
  inverse of the model-data mismatch, which it can for a mismatch that is
  dense, diagonal, sparse, block diagonal or Kronecker.

The information form is used when there are fewer state elements than
observations and the mismatch is dense, diagonal, sparse, block diagonal or
Kronecker (and the prior error covariance has a square root); otherwise the
whitened form (``_choose``). :class:`AnalyticOptions` holds :math:`\gamma`.

Either form's matrix is factored at the solve's :math:`\gamma` by Cholesky
(``_decompose``): most problems are solved at one :math:`\gamma`, and the
Cholesky costs a fraction of an eigendecomposition. The eigendecomposition
reads the factors at any :math:`\gamma`, so the first ``post.with_gamma``
does it and keeps it, and later ones share it. Neither depends on the values
of the observations, so the posterior of other observations
(``post.with_data``) shares the factors instead of solving again. The
eigendecomposition uses LAPACK's ``evd`` driver, and redoes it with ``evr``
when there is not enough memory for ``evd``'s workspace (``_eigh``).
"""

import dataclasses
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import ClassVar, Self

import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import cholesky, eigh, solve_triangular
from scipy.linalg.blas import get_blas_funcs
from scipy.linalg.lapack import get_lapack_funcs

from fips.linalg import (
    BlockDiagonalOperator,
    DenseOperator,
    DiagonalOperator,
    KroneckerOperator,
    LinearOperator,
    SparseOperator,
)
from fips.linalg.forms import _read_only
from fips.posterior import Posterior
from fips.problem import Problem

logger = logging.getLogger(__name__)

_LARGE_M = 5000
"""
The number of observations above which a whitened solve with many more observations than state elements logs a warning.

The whitened form factors an ``m x m`` matrix, in a time that grows as
``m^3``. Measured on one core of a cluster node (``n = 100``) when every solve
eigendecomposed it, the whole solve took 2 s at ``m = 2,205``, 25 s at
``5,000``, 85 s at ``7,500`` and 203 s at ``10,000``; the Cholesky of a
one-gamma solve is several times faster, and the first ``with_gamma`` still
eigendecomposes. When the information form could not be used and ``m`` is
more than twice ``n``, a solve above this size says why.
"""

_INVERTIBLE = (
    DenseOperator,
    DiagonalOperator,
    SparseOperator,
    BlockDiagonalOperator,
    KroneckerOperator,
)
"""
The forms of the model-data mismatch whose inverse the information form applies.

A dense mismatch is applied through its Cholesky factor. The whitened form
needs that factor too, so the information form saves the ``m x m``
factorization and the ``m x m`` products. A Kronecker mismatch is applied
through its factors; the whitened form would form it as a dense ``m x m``
matrix.
"""


@dataclass(frozen=True)
class AnalyticOptions:
    r"""
    The options of the ``"analytic"`` method.

    Parameters
    ----------
    gamma : float, default 1.0
        The regularization factor :math:`\gamma`, which multiplies the
        observation term of the cost:
        :math:`J(x) = (x - x_0)^T S_0^{-1} (x - x_0) + \gamma (z - H x - c)^T S_z^{-1} (z - H x - c)`.
        It is the same as dividing :math:`S_z` by :math:`\gamma`: above 1 the
        estimate fits the observations more closely, below 1 it stays closer
        to the prior. It must be positive and finite.

    Raises
    ------
    ValueError
        If `gamma` is not a positive, finite number.
    """

    gamma: float = 1.0
    """The regularization factor: the model-data mismatch is divided by it."""

    def __post_init__(self):
        try:
            gamma = float(self.gamma)
        except (TypeError, ValueError):
            raise ValueError(f"gamma must be a number, got {self.gamma!r}.") from None
        if not (math.isfinite(gamma) and gamma > 0):
            raise ValueError(f"gamma must be positive and finite, got {gamma}.")
        object.__setattr__(self, "gamma", gamma)


@dataclass(frozen=True, eq=False)
class _Factors:
    r"""
    What both forms share: the weights, DOFS, and the eigendecomposition :meth:`with_gamma` needs.

    A form factors its matrix :math:`A` (``M`` or ``N``) so that
    :math:`(A + I/\gamma)^{-1} = X \operatorname{diag}(w) X^T` (``_decompose``).
    ``fips.solve`` factors by Cholesky at its gamma: :attr:`eigenvalues` is
    None, the weights are 1 and DOFS was computed with the factor. The first
    :meth:`with_gamma` eigendecomposes the same problem, keeps that, and reads
    it at the new gamma: :math:`w_k = \gamma / (\gamma \lambda_k + 1)`.
    """

    estimate_is_linear: ClassVar[bool] = True
    r"""The estimate is linear in the observations, :math:`x_0 + K (z - H x_0 - c)` with the gain :math:`K` of ``gain``."""

    problem: Problem
    """The problem that was factored."""
    gamma: float
    r"""The regularization factor :math:`\gamma` the factors are read at."""
    eigenvalues: np.ndarray | None
    r"""The eigenvalues :math:`\lambda_k` of the form's matrix, clipped at zero; None after a Cholesky."""
    innovation: np.ndarray
    """The innovation in the form's basis (:math:`e` or :math:`b`)."""
    dofs_at_gamma: float | None = field(default=None, kw_only=True)
    """DOFS at :attr:`gamma`, computed with a Cholesky; None after an eigendecomposition."""

    @classmethod
    def factor(
        cls, problem: Problem, gamma: float, eigendecompose: bool = False
    ) -> "Self":
        """Factor a problem (each form defines it)."""
        raise NotImplementedError

    def weights(self) -> np.ndarray:
        r"""Return :math:`w_k = \gamma / (\gamma \lambda_k + 1)`, or ones after a Cholesky (its factors are read at gamma already)."""
        if self.eigenvalues is None:
            return np.ones(len(self.innovation))
        return self.gamma / (self.gamma * self.eigenvalues + 1.0)

    def dofs(self) -> float:
        r"""Return :math:`\sum_k \gamma \lambda_k / (\gamma \lambda_k + 1)`, or after a Cholesky the DOFS computed with it."""
        if self.eigenvalues is None:
            assert self.dofs_at_gamma is not None  # set with every Cholesky
            return self.dofs_at_gamma
        return float(np.sum(self.weights() * self.eigenvalues))

    def with_gamma(self, gamma: float) -> "Self":
        """Return the same factors, read at `gamma`, from the eigendecomposition (done on the first call after a Cholesky, and kept)."""
        if self.eigenvalues is not None:
            return dataclasses.replace(self, gamma=gamma)
        eigendecomposed = self.__dict__.get("_eigendecomposed")
        if eigendecomposed is None:
            logger.debug(
                "Eigendecomposing the %s form's matrix, for another gamma.",
                type(self).__name__,
            )
            eigendecomposed = type(self).factor(
                self.problem, self.gamma, eigendecompose=True
            )
            # Kept outside the fields (the dataclass is frozen): later sweeps share it
            self.__dict__["_eigendecomposed"] = eigendecomposed
        return eigendecomposed.with_gamma(gamma)

    def __getstate__(self) -> dict:
        """Return the fields for a pickle or a copy, without the kept eigendecomposition (the first ``with_gamma`` after loading redoes it)."""
        state = dict(self.__dict__)
        state.pop("_eigendecomposed", None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore from a pickle or a copy, the arrays read-only again."""
        self.__dict__.update(state)
        for value in state.values():
            if isinstance(value, np.ndarray):
                _read_only(value)


@dataclass(frozen=True, eq=False)
class Whitened(_Factors):
    r"""
    The analytic method's solution in the whitened form: the factors of one problem, read at one gamma.

    ``_analytic`` factors its problem into this (:meth:`factor`) unless it
    takes the :class:`Information` form, and the :class:`~fips.Posterior`
    asks it for every array it reports: it is the analytic method's
    :class:`~fips.posterior.Solution`. It factors an :math:`m \times m`
    matrix. Nothing :math:`n \times n` is formed unless the full posterior
    covariance or the gain is asked for. The arrays are read-only, also after
    a pickle or a copy.

    With :math:`Q = S_0` the prior error covariance (:math:`n \times n`),
    :math:`S_z = L L^T` the model-data mismatch (:math:`m \times m`; :math:`L` its
    Cholesky factor, or the square root of its diagonal) and :math:`S_z / \gamma`
    the mismatch the solve uses:

    .. math::

        M = L^{-1} H Q H^T L^{-T}, \qquad
        H Q H^T + S_z / \gamma = L (M + I / \gamma) L^T.

    :math:`M` is symmetric positive semidefinite, and the solve needs
    :math:`(M + I/\gamma)^{-1} = X \operatorname{diag}(w) X^T`. ``fips.solve``
    takes it from the Cholesky factor at its gamma, :math:`M + I/\gamma = R
    R^T`: :math:`X = R^{-T}` and :math:`w = 1`. The first :meth:`with_gamma`
    takes it from the eigendecomposition :math:`M = V \Lambda V^T`, which holds
    at any gamma: :math:`X = V` and :math:`w_k = \gamma / (\gamma \lambda_k +
    1)`, the eigenvalues clipped at zero (roundoff can make a few slightly
    negative). With the whitener :math:`B = X^T L^{-1}`, :math:`W = B H Q`
    (:math:`m \times n`) and the whitened innovation :math:`e = B d` with
    :math:`d = z - H x_0 - c`:

    .. math::

        \hat{x} - x_0 = W^T (w \circ e), \qquad
        \hat{S} = Q - W^T \operatorname{diag}(w) W, \qquad
        \operatorname{diag}(\hat{S}) = \operatorname{diag}(Q) - \textstyle\sum_k w_k W_{k\cdot}^2,

    .. math::

        K = W^T \operatorname{diag}(w) B, \qquad
        d^T (H Q H^T + S_z/\gamma)^{-1} d = \textstyle\sum_k w_k e_k^2.

    The last is the minimum of the cost, the sum of the data and model misfits of
    the reduced chi-squared. DOFS, :math:`\operatorname{tr}(K H) =
    \operatorname{tr}(M (M + I/\gamma)^{-1})`, is :math:`\sum_k \gamma\lambda_k /
    (\gamma\lambda_k + 1)` from the eigenvalues, and :math:`\operatorname{tr}(R^{-1}
    M R^{-T})` from the Cholesky factor. Once the eigendecomposition
    is done only :math:`w` depends on :math:`\gamma`, so another :math:`\gamma`
    costs :math:`O(m n)`.
    """

    estimate_is_mean: ClassVar[bool] = True
    """The posterior is Gaussian, so its estimate is its mean."""

    # And from _Factors: problem, gamma, eigenvalues (of M, m), innovation
    # (e = B (z - H x_0 - c), m), dofs_at_gamma
    whitener: np.ndarray
    """:math:`B = X^T L^{-1}`, which whitens an observation-space vector (``m x m``)."""
    W: np.ndarray
    """:math:`W = B H Q` (``m x n``)."""
    prior_variances: np.ndarray
    """The diagonal of :math:`Q` (``n``)."""

    @classmethod
    def factor(
        cls, problem: Problem, gamma: float, eigendecompose: bool = False
    ) -> "Whitened":
        r"""
        Factor a problem.

        :math:`H S_0` and :math:`H S_0 H^T` are formed in the cheapest form of
        their factors (through a Kronecker :math:`S_0`'s factors), then dense.

        Parameters
        ----------
        problem : Problem
            The problem to factor.
        gamma : float
            The regularization factor to read the factors at.
        eigendecompose : bool, default False
            Eigendecompose :math:`M`, which :meth:`with_gamma` reads at any
            gamma, instead of the Cholesky at `gamma` (``_decompose``).

        Returns
        -------
        Whitened
            The factors.

        Raises
        ------
        numpy.linalg.LinAlgError
            If the model-data mismatch is not positive-definite.
        """
        H = problem.forward.data
        S_0 = problem.prior_cov.data
        HS_0 = H @ S_0
        HS_0H = HS_0 @ H._transpose()
        d = _innovation(problem)
        HQ = HS_0.dense()
        HQH = HS_0H.dense()
        S_z = problem.obs_cov.data
        if isinstance(S_z, DiagonalOperator):
            if not np.all(S_z.values > 0):
                raise LinAlgError(
                    "The model-data mismatch has a variance that is not positive, "
                    "so it cannot whiten the observations."
                )
            s = np.sqrt(S_z.values)
            M = HQH / s[:, None] / s[None, :]

            def whiten(X):
                """Return ``X^T L^{-1}`` for the diagonal ``L``."""
                return X.T / s[None, :]

        else:
            try:
                L = cholesky(S_z.dense(), lower=True)
            except LinAlgError as err:
                raise LinAlgError(
                    "The model-data mismatch is not positive-definite, so it "
                    "cannot whiten the observations."
                ) from err
            # np.asarray: SciPy is not annotated, and gives no array type
            X = np.asarray(solve_triangular(L, HQH, lower=True))
            M = np.asarray(solve_triangular(L, X.T, lower=True))

            def whiten(X):
                """Return ``X^T L^{-1}``, as the transpose of ``L^{-T} X``."""
                return np.asarray(solve_triangular(L, X, lower=True, trans="T")).T

        M = 0.5 * (M + M.T)
        X, eigenvalues, dofs = _decompose(
            M,
            gamma,
            eigendecompose,
            "The whitened prior in observation space",
            "the prior error covariance",
        )
        B = whiten(X)
        return cls(
            problem=problem,
            gamma=gamma,
            eigenvalues=None if eigenvalues is None else _read_only(eigenvalues),
            whitener=_read_only(B),
            W=_read_only(B @ HQ),
            innovation=_read_only(B @ d),
            prior_variances=_read_only(np.asarray(S_0.diagonal(), dtype=float)),
            dofs_at_gamma=dofs,
        )

    def estimate(self) -> np.ndarray:
        r"""Return :math:`\hat{x} = x_0 + W^T (w \circ e)`."""
        increment = self.W.T @ (self.weights() * self.innovation)
        return self.problem.prior.values + increment

    def variances(self) -> np.ndarray:
        r"""Return the diagonal of :math:`\hat{S}`, :math:`O(m n)`."""
        w = self.weights()
        return self.prior_variances - np.einsum("k,kj,kj->j", w, self.W, self.W)

    def cov(self, W: LinearOperator | None = None) -> np.ndarray:
        r"""
        Return :math:`\hat{S}`, ``n x n``, or for an aggregation :math:`W \hat{S} W^T` without it.

        :math:`W \hat{S} W^T = W Q W^T - (W W_f^T) \operatorname{diag}(w) (W W_f^T)^T`,
        with :math:`W_f` the factor :attr:`W`: no product bigger than ``k x m``.
        """
        S_0 = self.problem.prior_cov.data
        if W is None:
            Y = np.sqrt(self.weights())[:, None] * self.W
            S_hat = Y.T @ Y  # Q - S_hat, until the subtraction
            np.subtract(S_0.dense(), S_hat, out=S_hat)
            return S_hat
        prior = np.asarray(W @ (S_0 @ W.T))
        WWf = W @ self.W.T  # k x m
        return prior - (WWf * self.weights()) @ WWf.T

    def gain(self, W: LinearOperator | None = None) -> np.ndarray:
        r"""Return the gain :math:`K = W_f^T \operatorname{diag}(w) B`, ``n x m``, or for an aggregation :math:`W K` without it."""
        if W is None:
            return self.W.T @ (self.weights()[:, None] * self.whitener)
        return ((W @ self.W.T) * self.weights()) @ self.whitener

    def chi2(self) -> float:
        r"""Return :math:`d^T (H Q H^T + S_z/\gamma)^{-1} d = \sum_k w_k e_k^2`."""
        return float(np.sum(self.weights() * self.innovation**2))

    def innovation_precision(self, groups: Sequence[np.ndarray]) -> list[np.ndarray]:
        r"""
        Return each group's block of :math:`P = (H Q H^T + S_z/\gamma)^{-1} = B^T \operatorname{diag}(w) B`.

        For a group of observations at positions :math:`g`,
        :math:`P_{gg} = B_g^T \operatorname{diag}(w) B_g` from the columns
        :math:`B_g` of the whitener: :math:`O(m\, m_g^2)`.
        """
        w = self.weights()[:, None]
        blocks = []
        for rows in groups:
            B_g = self.whitener[:, rows]
            blocks.append((B_g * w).T @ B_g)
        return blocks

    def self_sensitivity(self) -> np.ndarray:
        r"""
        Return :math:`\operatorname{diag}(H K)`, each observation's share of DOFS.

        :math:`H K = I - (S_z/\gamma) P` and :math:`S_z B^T = L X`, so
        :math:`(H K)_{ii} = 1 - \sum_k (S_z B^T)_{ik} w_k B_{ki} / \gamma`:
        :math:`O(m^2)` for a diagonal :math:`S_z`, one product of
        :math:`S_z` with :math:`B^T` (``m x m``) otherwise.
        """
        B, w = self.whitener, self.weights()
        S_z = self.problem.obs_cov.data
        if isinstance(S_z, DiagonalOperator):
            SP = S_z.values * np.einsum("k,ki,ki->i", w, B, B)
        else:
            SP = np.einsum("ik,k,ki->i", np.asarray(S_z @ B.T), w, B)
        return 1.0 - SP / self.gamma

    def with_data(self, problem: Problem) -> "Whitened":
        """
        Return the factors of `problem`, this one's with other observation values.

        Only the whitened innovation changes: one ``m x m`` product.
        """
        d = _innovation(problem)
        innovation = _read_only(self.whitener @ d)
        return dataclasses.replace(self, problem=problem, innovation=innovation)


@dataclass(frozen=True, eq=False)
class Information(_Factors):
    r"""
    The analytic method's solution in the information form: the factors of one problem, read at one gamma.

    ``_analytic`` factors its problem into this (:meth:`factor`) when there
    are fewer state elements than observations and the model-data mismatch
    is dense, diagonal, sparse, block diagonal or Kronecker; otherwise into
    :class:`Whitened`. It factors an :math:`n \times n` matrix instead of an
    :math:`m \times m` one, and gives the same posterior. It holds one
    :math:`n \times n` array; it forms nothing :math:`m \times m` of its own
    (a dense mismatch is factored by Cholesky inside each solve with it; a
    Kronecker one is applied through its factors). The arrays are read-only,
    also after a pickle or a copy.

    With :math:`S_0 = R_0 R_0^T` the prior error covariance (:math:`R_0` its
    square root in its own form: :meth:`~fips.linalg.LinearOperator.sqrt`),
    :math:`S_z` the model-data mismatch and :math:`S_z / \gamma` the mismatch
    the solve uses:

    .. math::

        N = R_0^T H^T S_z^{-1} H R_0, \qquad
        \hat{S} = R_0 (\gamma N + I)^{-1} R_0^T
        = R_0 (N + I/\gamma)^{-1} R_0^T / \gamma.

    The solve needs :math:`(N + I/\gamma)^{-1} = X \operatorname{diag}(w) X^T`,
    which it takes as the whitened form does: from the Cholesky factor at
    ``fips.solve``'s gamma, :math:`N + I/\gamma = R R^T` (:math:`X = R^{-T}`,
    :math:`w = 1`), or, from the first :meth:`with_gamma` on, from the
    eigendecomposition :math:`N = U \Lambda U^T` (:math:`X = U`, :math:`w_k =
    \gamma / (\gamma \lambda_k + 1)`, the eigenvalues clipped at zero). The
    nonzero eigenvalues of :math:`N` are those of the whitened form's
    :math:`M` (both are the squared singular values of :math:`S_z^{-1/2} H
    R_0`), so the weights and DOFS are the same. With :math:`G = R_0 X`
    (:math:`n \times n`), the innovation :math:`d = z - H x_0 - c` and
    :math:`b = G^T H^T S_z^{-1} d`:

    .. math::

        \hat{x} - x_0 = G (w \circ b), \qquad
        \hat{S} = G \operatorname{diag}(w / \gamma) G^T, \qquad
        \operatorname{diag}(\hat{S})_i = \textstyle\sum_k G_{ik}^2 w_k / \gamma,

    .. math::

        K = G \operatorname{diag}(w) G^T H^T S_z^{-1}, \qquad
        \mathrm{DOFS} = \operatorname{tr}(N (N + I/\gamma)^{-1}),

    DOFS from the eigenvalues or the Cholesky factor as in the whitened form.
    The chi-squared is the minimum of the cost, :math:`d^T (H S_0 H^T +
    S_z/\gamma)^{-1} d = \gamma d^T S_z^{-1} r`, with :math:`r = d - H
    (\hat{x} - x_0)` the residual. Once the eigendecomposition is done only
    :math:`w` depends on :math:`\gamma`, so another :math:`\gamma` costs
    :math:`O(n^2)`.
    """

    estimate_is_mean: ClassVar[bool] = True
    """The posterior is Gaussian, so its estimate is its mean."""

    # And from _Factors: problem, gamma, eigenvalues (of N, n), innovation
    # (b = G^T H^T S_z^-1 (z - H x_0 - c), n), dofs_at_gamma
    G: np.ndarray
    """:math:`G = R_0 X`, the prior's square root in the basis :math:`X` (``n x n``)."""

    @classmethod
    def factor(
        cls, problem: Problem, gamma: float, eigendecompose: bool = False
    ) -> "Information":
        r"""
        Factor a problem.

        :math:`H R_0` is formed in the cheapest form of its factors (through a
        Kronecker :math:`S_0`'s factors), then dense (:math:`m \times n`), and
        :math:`S_z^{-1}` is applied by its form's solve (for a dense
        :math:`S_z`, by Cholesky; for a Kronecker one, through its factors).

        Parameters
        ----------
        problem : Problem
            The problem to factor.
        gamma : float
            The regularization factor to read the factors at.
        eigendecompose : bool, default False
            Eigendecompose :math:`N`, which :meth:`with_gamma` reads at any
            gamma, instead of the Cholesky at `gamma` (``_decompose``).

        Returns
        -------
        Information
            The factors.

        Raises
        ------
        numpy.linalg.LinAlgError
            If the model-data mismatch is not positive-definite.
        _NoSquareRoot
            If the prior error covariance has no square root (it is not
            positive-definite to working precision).
        """
        H = problem.forward.data
        S_z = problem.obs_cov.data
        try:
            S_z.logdet()  # raises unless positive-definite
        except LinAlgError as err:
            raise LinAlgError(
                "The model-data mismatch is not positive-definite, so it cannot "
                "weight the observations."
            ) from err
        try:
            R_0 = problem.prior_cov.data.sqrt()
        except LinAlgError as err:
            raise _NoSquareRoot(
                "The prior error covariance has no square root: it is not "
                "positive-definite to working precision."
            ) from err
        HR = (H @ R_0).dense()
        N = HR.T @ S_z.solve(HR)
        N = 0.5 * (N + N.T)
        X, eigenvalues, dofs = _decompose(
            N,
            gamma,
            eigendecompose,
            "The observations' information on the whitened state",
            "the model-data mismatch",
        )
        G = np.asarray(R_0 @ X)
        return cls(
            problem=problem,
            gamma=gamma,
            eigenvalues=None if eigenvalues is None else _read_only(eigenvalues),
            G=_read_only(G),
            innovation=_read_only(_carried(problem, G)),
            dofs_at_gamma=dofs,
        )

    def _increment(self) -> np.ndarray:
        r"""Return :math:`\hat{x} - x_0 = G (w \circ b)`."""
        return self.G @ (self.weights() * self.innovation)

    def estimate(self) -> np.ndarray:
        r"""Return :math:`\hat{x} = x_0 + G (w \circ b)`."""
        return self.problem.prior.values + self._increment()

    def variances(self) -> np.ndarray:
        r"""Return the diagonal of :math:`\hat{S}`, :math:`O(n^2)`."""
        return np.einsum("ik,k,ik->i", self.G, self.weights() / self.gamma, self.G)

    def cov(self, W: LinearOperator | None = None) -> np.ndarray:
        r"""
        Return :math:`\hat{S}`, ``n x n``, or for an aggregation :math:`W \hat{S} W^T` without it.

        :math:`W \hat{S} W^T = (W G) \operatorname{diag}(w / \gamma) (W G)^T`:
        no product bigger than ``k x n``.
        """
        Y = self.G if W is None else np.asarray(W @ self.G)
        Y = Y * np.sqrt(self.weights() / self.gamma)
        return Y @ Y.T

    def gain(self, W: LinearOperator | None = None) -> np.ndarray:
        r"""
        Return the gain :math:`K = G \operatorname{diag}(w) (S_z^{-1} H G)^T`, ``n x m``, or for an aggregation :math:`W K` without it.

        :math:`S_z^{-1} H G` (``m x n``) is formed for each call.
        """
        problem = self.problem
        HG = np.asarray(problem.forward.data @ self.G)
        Z = problem.obs_cov.data.solve(HG)
        WG = self.G if W is None else np.asarray(W @ self.G)
        return (WG * self.weights()) @ Z.T

    def chi2(self) -> float:
        r"""Return the minimum of the cost, :math:`\gamma d^T S_z^{-1} r` with :math:`r = d - H (\hat{x} - x_0)`."""
        problem = self.problem
        d = _innovation(problem)
        r = d - problem.forward.data @ self._increment()
        return self.gamma * float(d @ problem.obs_cov.data.solve(r))

    def _weighted_obs(self) -> tuple[np.ndarray, np.ndarray]:
        r"""Return :math:`H G` and :math:`Z = S_z^{-1} H G` (both ``m x n``)."""
        HG = np.asarray(self.problem.forward.data @ self.G)
        return HG, np.asarray(self.problem.obs_cov.data.solve(HG))

    def innovation_precision(self, groups: Sequence[np.ndarray]) -> list[np.ndarray]:
        r"""
        Return each group's block of :math:`P = (H S_0 H^T + S_z/\gamma)^{-1}`.

        By Woodbury, :math:`P = \gamma S_z^{-1} - \gamma Z \operatorname{diag}(w) Z^T`
        with :math:`Z = S_z^{-1} H G` (``m x n``, formed once), so
        :math:`P_{gg} = \gamma \big((S_z^{-1})_{gg} - Z_g \operatorname{diag}(w) Z_g^T\big)`.
        The block :math:`(S_z^{-1})_{gg}` is read from a diagonal
        :math:`S_z`, from one inverse of a dense one, and otherwise from a
        solve with the group's columns of the identity, in the form's own way.
        """
        _, Z = self._weighted_obs()
        w = self.weights()
        S_z = self.problem.obs_cov.data
        m = S_z.shape[0]
        inverse = S_z.solve(np.eye(m)) if isinstance(S_z, DenseOperator) else None
        blocks = []
        for rows in groups:
            if isinstance(S_z, DiagonalOperator):
                S_inv = np.diag(1.0 / S_z.values[rows])
            elif inverse is not None:
                S_inv = inverse[np.ix_(rows, rows)]
            else:
                E = np.zeros((m, len(rows)))
                E[rows, np.arange(len(rows))] = 1.0
                S_inv = np.asarray(S_z.solve(E))[rows]
            Z_g = Z[rows]
            blocks.append(self.gamma * (S_inv - (Z_g * w) @ Z_g.T))
        return blocks

    def self_sensitivity(self) -> np.ndarray:
        r"""
        Return :math:`\operatorname{diag}(H K)`, each observation's share of DOFS.

        :math:`H K = (H G) \operatorname{diag}(w) Z^T`, so
        :math:`(H K)_{ii} = \sum_k (H G)_{ik} w_k Z_{ik}`: :math:`O(m n)`.
        """
        HG, Z = self._weighted_obs()
        return np.einsum("ik,k,ik->i", HG, self.weights(), Z)

    def with_data(self, problem: Problem) -> "Information":
        """
        Return the factors of `problem`, this one's with other observation values.

        Only :math:`b` changes: a solve with the model-data mismatch, a
        product with :math:`H^T` and one with :math:`G^T` (``n x n``).
        """
        innovation = _read_only(_carried(problem, self.G))
        return dataclasses.replace(self, problem=problem, innovation=innovation)


class _NoSquareRoot(LinAlgError):
    """The prior error covariance has no square root, so the information form cannot factor the problem."""


def _innovation(problem: Problem) -> np.ndarray:
    """Return the innovation :math:`d = z - H x_0 - c`, through ``problem.predict``."""
    return problem.obs.values - problem.predict(problem.prior).values


def _carried(problem: Problem, G: np.ndarray) -> np.ndarray:
    r"""Return :math:`b = G^T H^T S_z^{-1} d`, the innovation carried into the information form's eigenvectors."""
    y = problem.obs_cov.data.solve(_innovation(problem))
    return G.T @ (problem.forward.data._transpose() @ y)


def _decompose(
    A: np.ndarray, gamma: float, eigendecompose: bool, matrix: str, covariance: str
) -> tuple[np.ndarray, np.ndarray | None, float | None]:
    r"""
    Return :math:`X`, the eigenvalues and DOFS, with :math:`(A + I/\gamma)^{-1} = X \operatorname{diag}(w) X^T`.

    `A` is a form's symmetric positive semidefinite matrix (``M`` or ``N``,
    ``k x k``), which this may overwrite.

    Here a solve chooses how to factor. Most problems are solved at one
    gamma, and a Cholesky :math:`A + I/\gamma = R R^T` costs a fraction of an
    eigendecomposition (about a tenth at ``k = 10,000``): :math:`X = R^{-T}`,
    :math:`w = 1`, no eigenvalues, and DOFS is :math:`\operatorname{tr}(A (A +
    I/\gamma)^{-1}) = \operatorname{tr}(R^{-1} A R^{-T})`, by a triangular
    product (:math:`k^3` flops, less than the form spends making `A`). It
    equals :math:`k - \lVert R^{-1} \rVert_F^2 / \gamma`, but that loses
    digits when DOFS is much smaller than `k` (two in paper 1's monthly
    inversion). The
    eigendecomposition :math:`A = V \Lambda V^T` holds at any gamma
    (:math:`X = V`, :math:`w_k = \gamma / (\gamma \lambda_k + 1)`; DOFS is
    then read from the eigenvalues, so None here). It is done when
    `eigendecompose` (the first ``with_gamma`` asks for it), and when `A` is
    not positive semidefinite to roundoff (``_cholesky_inverse``), so that
    ``_clip`` clips its negative eigenvalues and names `matrix` and
    `covariance` in its warning, as for every solve before.
    """
    if not eigendecompose:
        R_inv = _cholesky_inverse(A, gamma)
        if R_inv is not None:
            (trmm,) = get_blas_funcs(("trmm",), (R_inv,))  # triangular times full
            dofs = float(np.vdot(trmm(1.0, R_inv, A, lower=1), R_inv))
            return R_inv.T, None, dofs
        logger.debug(
            "%s is not positive semidefinite to roundoff: eigendecomposing it.", matrix
        )
    eigenvalues, V = _eigh(A)
    _clip(eigenvalues, matrix, covariance)
    return V, eigenvalues, None


def _cholesky_inverse(A: np.ndarray, gamma: float) -> np.ndarray | None:
    r"""
    Return :math:`R^{-1}`, with :math:`A + I/\gamma = R R^T` and :math:`R` lower triangular, or None when `A` is not positive semidefinite to roundoff.

    A Cholesky factor of :math:`A + I/\gamma` exists whenever no eigenvalue
    of `A` is below :math:`-1/\gamma`, so it cannot show the negative
    eigenvalues that ``_clip`` warns about. :math:`A + \tau I` is factored
    first, with :math:`\tau = k \epsilon \max_i A_{ii}`: it has a factor
    only if no eigenvalue of `A` is below about :math:`-\tau`. Since
    :math:`\max_i A_{ii} \le \lambda_{max}`, :math:`\tau` is at most the
    roundoff ``_clip`` allows, so an `A` that passes would have been clipped
    without a warning. One that fails (or a gamma so large that :math:`A +
    I/\gamma` has no factor) gives None, and is eigendecomposed.
    """
    k = len(A)
    tau = k * np.finfo(float).eps * float(np.diagonal(A).max(initial=0.0))
    shifted = np.array(A, order="F")  # Fortran order: LAPACK factors it in place
    try:
        shifted.flat[:: k + 1] += tau
        cholesky(shifted, lower=True, overwrite_a=True, check_finite=False)
        np.copyto(shifted, A)
        shifted.flat[:: k + 1] += 1.0 / gamma
        R = cholesky(shifted, lower=True, overwrite_a=True, check_finite=False)
    except LinAlgError:
        return None
    (trtri,) = get_lapack_funcs(("trtri",), (R,))  # inverts a triangular matrix
    R_inv, info = trtri(R, lower=1, overwrite_c=1)
    if info != 0:  # a Cholesky factor has a positive diagonal: never
        raise LinAlgError(f"The Cholesky factor could not be inverted (info {info}).")
    # np.asarray: SciPy is not annotated, and gives no array type
    return np.asarray(R_inv)


def _eigh(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the eigenvalues and eigenvectors of the symmetric matrix `a`, which it may overwrite.

    LAPACK's divide-and-conquer driver (``evd``) is fast whatever the
    eigenvalues are, but needs a workspace of about ``m^2`` more numbers than
    the default driver (``evr``), which slows down when many eigenvalues are
    equal. When the workspace cannot be allocated, SciPy raises
    ``MemoryError`` before the decomposition starts, `a` is untouched, and the
    same decomposition is redone with ``evr``, logged at INFO.
    """
    try:
        w, V = eigh(a, overwrite_a=True, driver="evd")
    except MemoryError:
        logger.info(
            "There is not enough memory to eigendecompose the %d x %d matrix with "
            "the 'evd' driver; using 'evr', which needs less memory and is slower "
            "when many eigenvalues are equal.",
            len(a),
            len(a),
        )
        w, V = eigh(a, overwrite_a=True, driver="evr")
    # np.asarray: SciPy is not annotated, and gives no array type
    return np.asarray(w), np.asarray(V)


def _clip(eigenvalues: np.ndarray, matrix: str, covariance: str) -> None:
    """
    Set the negative eigenvalues of a positive semidefinite matrix to zero, in place.

    Roundoff leaves eigenvalues of order ``m * eps * max|lambda|`` that should
    be zero, some negative. One beyond that is logged, naming the `matrix`:
    the inputs are not what they should be (`covariance` is not positive
    semidefinite).
    """
    low = eigenvalues.min(initial=0.0)
    if low < 0:
        roundoff = len(eigenvalues) * np.finfo(float).eps * np.abs(eigenvalues).max()
        if -low > roundoff:
            logger.warning(
                "%s has a negative eigenvalue, %.3g, larger than roundoff (%.3g); it "
                "and the other negative eigenvalues are set to zero. Check that %s "
                "is positive semidefinite.",
                matrix,
                low,
                roundoff,
                covariance,
            )
        np.maximum(eigenvalues, 0.0, out=eigenvalues)


def _choose(problem: Problem) -> tuple[type, str]:
    """
    Return the form to factor `problem` in, and why.

    The information form when there are fewer state elements than
    observations and the model-data mismatch is dense, diagonal, sparse, block
    diagonal or Kronecker; otherwise the whitened form. The tests replace this
    function to hold both forms to the same outputs.
    """
    m, n = problem.forward.data.shape
    S_z = problem.obs_cov.data
    if n >= m:
        return (
            Whitened,
            f"the n = {n} state elements are not fewer than the m = {m} observations",
        )
    if not isinstance(S_z, _INVERTIBLE):
        return Whitened, (
            f"the model-data mismatch is a {type(S_z).__name__}, which the "
            "information form does not take (it takes dense, diagonal, sparse, "
            "block diagonal and Kronecker)"
        )
    return Information, (
        f"the n = {n} state elements are fewer than the m = {m} observations, and "
        f"the model-data mismatch is a {type(S_z).__name__}"
    )


def _factor(problem: Problem, gamma: float) -> "Whitened | Information":
    """
    Factor `problem` in the form :func:`_choose` gives, logging which and why.

    The information form needs a square root of the prior error covariance;
    when it has none (it is not positive-definite to working precision), the
    whitened form, which needs none, factors the problem. A whitened solve
    with many more observations than state elements and more than
    ``_LARGE_M`` of them logs a warning that says why the information form was
    not used.
    """
    form, why = _choose(problem)
    if form is Information:
        try:
            solution = Information.factor(problem, gamma)
        except _NoSquareRoot:
            why = (
                "the prior error covariance has no square root (it is not "
                "positive-definite to working precision), which the information "
                "form needs"
            )
        else:
            logger.debug("The analytic method uses the information form: %s.", why)
            return solution
    logger.debug("The analytic method uses the whitened form: %s.", why)
    m, n = problem.forward.data.shape
    if m > _LARGE_M and m > 2 * n:
        logger.warning(
            "The whitened form factors an m x m matrix, with m = %d "
            "observations, more than twice the n = %d state elements: its time "
            "grows as m^3. The information form (n x n) would be cheaper, but %s. "
            "Aggregating the observations makes m smaller.",
            m,
            n,
            why,
        )
    return Whitened.factor(problem, gamma)


def _analytic(problem: Problem, options: AnalyticOptions) -> Posterior:
    """Factor the problem once, in the form its size calls for; the posterior asks the factors for everything."""
    return Posterior(problem, _factor(problem, options.gamma), options)

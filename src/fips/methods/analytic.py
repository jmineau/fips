r"""
The analytic method: the closed-form Gaussian posterior of a linear problem.

``_analytic`` factors the problem once and returns a :class:`~fips.Posterior`
that asks the factors for everything it reports. It factors in one of two
forms, chosen by the problem's size, which give the same posterior:

- ``Whitened`` whitens the observations with the model-data mismatch and
  eigendecomposes an ``m x m`` matrix (``m`` observations);
- ``Information`` whitens the state with the prior error covariance and
  eigendecomposes an ``n x n`` matrix (``n`` state elements). It applies the
  inverse of the model-data mismatch, which it can for a mismatch that is
  dense, diagonal, sparse, block diagonal or Kronecker.

The information form is used when there are fewer state elements than
observations and the mismatch is dense, diagonal, sparse, block diagonal or
Kronecker (and the prior error covariance has a square root); otherwise the
whitened form (``_choose``). The factors depend neither on the regularization factor
:math:`\gamma` nor on the values of the observations, so the posterior of
another :math:`\gamma` (``post.with_gamma``) or of other observations
(``post.with_data``) shares them instead of solving again.
:class:`AnalyticOptions` holds :math:`\gamma`.

Either form's eigendecomposition uses LAPACK's ``evd`` driver, and redoes it
with ``evr`` when there is not enough memory for ``evd``'s workspace
(``_eigh``).
"""

import dataclasses
import logging
import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import cholesky, eigh, solve_triangular

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

The whitened form eigendecomposes an ``m x m`` matrix, in a time that grows as
``m^3``. Measured on one core of a cluster node (``n = 100``), the whole solve
took 2 s at ``m = 2,205``, 25 s at ``5,000``, 85 s at ``7,500`` and 203 s at
``10,000``. When the information form could not be used and ``m`` is more
than twice ``n``, a solve above this size says why.
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
eigendecomposition and the ``m x m`` products. A Kronecker mismatch is applied
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
class Whitened:
    r"""
    The analytic method's solution in the whitened form: the factors of one problem, read at one gamma.

    ``_analytic`` factors its problem into this once (:meth:`factor`) unless
    it takes the :class:`Information` form, and the
    :class:`~fips.Posterior` asks it for every array it reports: it is the
    analytic method's :class:`~fips.posterior.Solution`. It eigendecomposes an
    :math:`m \times m` matrix. The factors depend neither on gamma nor on the values of the observations, so
    :meth:`with_gamma` and :meth:`with_data` share them. Nothing
    :math:`n \times n` is formed unless the full posterior covariance or the
    gain is asked for. The arrays are read-only, also after a pickle or a copy.

    With :math:`Q = S_0` the prior error covariance (:math:`n \times n`),
    :math:`S_z = L L^T` the model-data mismatch (:math:`m \times m`; :math:`L` its
    Cholesky factor, or the square root of its diagonal) and :math:`S_z / \gamma`
    the mismatch the solve uses:

    .. math::

        M = L^{-1} H Q H^T L^{-T} = V \Lambda V^T, \qquad
        H Q H^T + S_z / \gamma = L V (\Lambda + I / \gamma) V^T L^T.

    :math:`M` is symmetric positive semidefinite, so its eigenvalues are clipped at
    zero (roundoff can make a few slightly negative). With the whitener
    :math:`B = V^T L^{-1}`, :math:`W = B H Q` (:math:`m \times n`), the whitened
    innovation :math:`e = B d` with :math:`d = z - H x_0 - c`, and the weights
    :math:`w_k = \gamma / (\gamma \lambda_k + 1)`:

    .. math::

        \hat{x} - x_0 = W^T (w \circ e), \qquad
        \hat{S} = Q - W^T \operatorname{diag}(w) W, \qquad
        \operatorname{diag}(\hat{S}) = \operatorname{diag}(Q) - \textstyle\sum_k w_k W_{k\cdot}^2,

    .. math::

        K = W^T \operatorname{diag}(w) B, \qquad
        \mathrm{DOFS} = \operatorname{tr}(K H) = \textstyle\sum_k \gamma\lambda_k / (\gamma\lambda_k + 1), \qquad
        d^T (H Q H^T + S_z/\gamma)^{-1} d = \textstyle\sum_k w_k e_k^2.

    The last is the minimum of the cost, the sum of the data and model misfits of
    the reduced chi-squared. Only :math:`w` depends on :math:`\gamma`, so a new
    :math:`\gamma` costs :math:`O(m n)`.
    """

    estimate_is_mean: ClassVar[bool] = True
    """The posterior is Gaussian, so its estimate is its mean."""

    problem: Problem
    """The problem that was factored (the estimate and the covariance read its prior)."""
    gamma: float
    r"""The regularization factor :math:`\gamma` the factors are read at."""
    eigenvalues: np.ndarray
    r"""The eigenvalues :math:`\lambda_k` of the whitened :math:`M`, clipped at zero (``m``)."""
    whitener: np.ndarray
    """:math:`B = V^T L^{-1}`, which whitens an observation-space vector (``m x m``)."""
    W: np.ndarray
    """:math:`W = B H Q` (``m x n``)."""
    innovation: np.ndarray
    """The whitened innovation :math:`e = B (z - H x_0 - c)` (``m``)."""
    prior_variances: np.ndarray
    """The diagonal of :math:`Q` (``n``)."""

    @classmethod
    def factor(cls, problem: Problem, gamma: float) -> "Whitened":
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

            def whiten(V):
                """Return ``V^T L^{-1}`` for the diagonal ``L``."""
                return V.T / s[None, :]

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

            def whiten(V):
                """Return ``V^T L^{-1}``, as the transpose of ``L^{-T} V``."""
                return np.asarray(solve_triangular(L, V, lower=True, trans="T")).T

        M = 0.5 * (M + M.T)
        eigenvalues, V = _eigh(M)
        _clip(
            eigenvalues,
            "The whitened prior in observation space",
            "the prior error covariance",
        )
        B = whiten(V)
        return cls(
            problem=problem,
            gamma=gamma,
            eigenvalues=_read_only(eigenvalues),
            whitener=_read_only(B),
            W=_read_only(B @ HQ),
            innovation=_read_only(B @ d),
            prior_variances=_read_only(np.asarray(S_0.diagonal(), dtype=float)),
        )

    def weights(self) -> np.ndarray:
        r"""Return :math:`w_k = \gamma / (\gamma \lambda_k + 1)`."""
        return self.gamma / (self.gamma * self.eigenvalues + 1.0)

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

    def dofs(self) -> float:
        r"""Return :math:`\sum_k \gamma \lambda_k / (\gamma \lambda_k + 1)`."""
        return float(np.sum(self.weights() * self.eigenvalues))

    def chi2(self) -> float:
        r"""Return :math:`d^T (H Q H^T + S_z/\gamma)^{-1} d = \sum_k w_k e_k^2`."""
        return float(np.sum(self.weights() * self.innovation**2))

    def with_gamma(self, gamma: float) -> "Whitened":
        """Return the same factors, read at `gamma`."""
        return dataclasses.replace(self, gamma=gamma)

    def with_data(self, problem: Problem) -> "Whitened":
        """
        Return the factors of `problem`, this one's with other observation values.

        Only the whitened innovation changes: one ``m x m`` product.
        """
        d = _innovation(problem)
        innovation = _read_only(self.whitener @ d)
        return dataclasses.replace(self, problem=problem, innovation=innovation)

    def __setstate__(self, state: dict) -> None:
        """Restore from a pickle or a copy, the arrays read-only again."""
        self.__dict__.update(state)
        for array in (
            self.eigenvalues,
            self.whitener,
            self.W,
            self.innovation,
            self.prior_variances,
        ):
            _read_only(array)


@dataclass(frozen=True, eq=False)
class Information:
    r"""
    The analytic method's solution in the information form: the factors of one problem, read at one gamma.

    ``_analytic`` factors its problem into this once (:meth:`factor`) when
    there are fewer state elements than observations and the model-data
    mismatch is dense, diagonal, sparse, block diagonal or Kronecker;
    otherwise into :class:`Whitened`. It eigendecomposes an
    :math:`n \times n` matrix instead of an :math:`m \times m` one, and gives
    the same posterior. The factors depend neither on gamma nor on the
    values of the observations, so :meth:`with_gamma` and :meth:`with_data`
    share them. It holds one :math:`n \times n` array; it forms nothing
    :math:`m \times m` of its own (a dense mismatch is factored by Cholesky
    inside each solve with it; a Kronecker one is applied through its factors). The arrays are read-only, also after a
    pickle or a copy.

    With :math:`S_0 = R R^T` the prior error covariance (:math:`R` its square
    root in its own form: :meth:`~fips.linalg.LinearOperator.sqrt`), :math:`S_z`
    the model-data mismatch and :math:`S_z / \gamma` the mismatch the solve
    uses:

    .. math::

        N = R^T H^T S_z^{-1} H R = U \Lambda U^T.

    The nonzero eigenvalues of :math:`N` are those of the whitened form's
    :math:`M` (both are the squared singular values of
    :math:`S_z^{-1/2} H R`), so the weights :math:`w_k = \gamma / (\gamma
    \lambda_k + 1)` and DOFS are the same. With :math:`G = R U`
    (:math:`n \times n`), the innovation :math:`d = z - H x_0 - c` and
    :math:`b = G^T H^T S_z^{-1} d`:

    .. math::

        \hat{x} - x_0 = G (w \circ b), \qquad
        \hat{S} = G \operatorname{diag}(w / \gamma) G^T, \qquad
        \operatorname{diag}(\hat{S})_i = \textstyle\sum_k G_{ik}^2 w_k / \gamma,

    .. math::

        K = G \operatorname{diag}(w) G^T H^T S_z^{-1}, \qquad
        \mathrm{DOFS} = \textstyle\sum_k \gamma\lambda_k / (\gamma\lambda_k + 1).

    The chi-squared is the minimum of the cost, the sum of its two misfits:
    :math:`\lVert w \circ b \rVert^2` for the state (:math:`G^{-1} (\hat{x}
    - x_0) = U^T (w \circ b)` is orthogonal) and :math:`\gamma r^T S_z^{-1} r`
    for the observations, with :math:`r = d - H (\hat{x} - x_0)`. Only
    :math:`w` depends on :math:`\gamma`, so a new :math:`\gamma` costs
    :math:`O(n^2)`.
    """

    estimate_is_mean: ClassVar[bool] = True
    """The posterior is Gaussian, so its estimate is its mean."""

    problem: Problem
    """The problem that was factored (the estimate and the gain read it)."""
    gamma: float
    r"""The regularization factor :math:`\gamma` the factors are read at."""
    eigenvalues: np.ndarray
    r"""The eigenvalues :math:`\lambda_k` of :math:`N`, clipped at zero (``n``)."""
    G: np.ndarray
    """:math:`G = R U`, the prior's square root in the eigenvectors of :math:`N` (``n x n``)."""
    innovation: np.ndarray
    r""":math:`b = G^T H^T S_z^{-1} (z - H x_0 - c)` (``n``)."""

    @classmethod
    def factor(cls, problem: Problem, gamma: float) -> "Information":
        r"""
        Factor a problem.

        :math:`H R` is formed in the cheapest form of its factors (through a
        Kronecker :math:`S_0`'s factors), then dense (:math:`m \times n`), and
        :math:`S_z^{-1}` is applied by its form's solve (for a dense
        :math:`S_z`, by Cholesky; for a Kronecker one, through its factors).

        Parameters
        ----------
        problem : Problem
            The problem to factor.
        gamma : float
            The regularization factor to read the factors at.

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
            R = problem.prior_cov.data.sqrt()
        except LinAlgError as err:
            raise _NoSquareRoot(
                "The prior error covariance has no square root: it is not "
                "positive-definite to working precision."
            ) from err
        HR = (H @ R).dense()
        N = HR.T @ S_z.solve(HR)
        N = 0.5 * (N + N.T)
        eigenvalues, U = _eigh(N)
        _clip(
            eigenvalues,
            "The observations' information on the whitened state",
            "the model-data mismatch",
        )
        G = np.asarray(R @ U)
        return cls(
            problem=problem,
            gamma=gamma,
            eigenvalues=_read_only(eigenvalues),
            G=_read_only(G),
            innovation=_read_only(_carried(problem, G)),
        )

    def weights(self) -> np.ndarray:
        r"""Return :math:`w_k = \gamma / (\gamma \lambda_k + 1)`."""
        return self.gamma / (self.gamma * self.eigenvalues + 1.0)

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

    def dofs(self) -> float:
        r"""Return :math:`\sum_k \gamma \lambda_k / (\gamma \lambda_k + 1)`."""
        return float(np.sum(self.weights() * self.eigenvalues))

    def chi2(self) -> float:
        r"""Return :math:`\lVert w \circ b \rVert^2 + \gamma r^T S_z^{-1} r`, the minimum of the cost."""
        problem = self.problem
        r = _innovation(problem) - problem.forward.data @ self._increment()
        state = float(np.sum((self.weights() * self.innovation) ** 2))
        return state + self.gamma * float(r @ problem.obs_cov.data.solve(r))

    def with_gamma(self, gamma: float) -> "Information":
        """Return the same factors, read at `gamma`."""
        return dataclasses.replace(self, gamma=gamma)

    def with_data(self, problem: Problem) -> "Information":
        """
        Return the factors of `problem`, this one's with other observation values.

        Only :math:`b` changes: a solve with the model-data mismatch, a
        product with :math:`H^T` and one with :math:`G^T` (``n x n``).
        """
        innovation = _read_only(_carried(problem, self.G))
        return dataclasses.replace(self, problem=problem, innovation=innovation)

    def __setstate__(self, state: dict) -> None:
        """Restore from a pickle or a copy, the arrays read-only again."""
        self.__dict__.update(state)
        for array in (self.eigenvalues, self.G, self.innovation):
            _read_only(array)


class _NoSquareRoot(LinAlgError):
    """The prior error covariance has no square root, so the information form cannot factor the problem."""


def _innovation(problem: Problem) -> np.ndarray:
    """Return the innovation :math:`d = z - H x_0 - c`, through ``problem.predict``."""
    return problem.obs.values - problem.predict(problem.prior).values


def _carried(problem: Problem, G: np.ndarray) -> np.ndarray:
    r"""Return :math:`b = G^T H^T S_z^{-1} d`, the innovation carried into the information form's eigenvectors."""
    y = problem.obs_cov.data.solve(_innovation(problem))
    return G.T @ (problem.forward.data._transpose() @ y)


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
            "The whitened form eigendecomposes an m x m matrix, with m = %d "
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

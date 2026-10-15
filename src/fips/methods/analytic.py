r"""
The analytic method: the closed-form Gaussian posterior of a linear problem.

``_analytic`` factors the problem's whitened system once, in a ``Whitened``,
and returns a :class:`~fips.Posterior` that asks the factors for everything it
reports. The factors depend neither on the regularization factor
:math:`\gamma` nor on the values of the observations, so the posterior of
another :math:`\gamma` (``post.with_gamma``) or of other observations
(``post.with_data``) shares them instead of solving again.
:class:`AnalyticOptions` holds :math:`\gamma`.
"""

import dataclasses
import logging
import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import cholesky, eigh, solve_triangular

from fips.linalg import DiagonalOperator, LinearOperator
from fips.posterior import Posterior
from fips.problem import Problem

logger = logging.getLogger(__name__)

_LARGE_M = 1000
"""
The number of observations from which a solve with more observations than state elements logs a warning.

The whitened solve eigendecomposes an ``m x m`` matrix. When ``m > n`` the
state-space (information) form, ``n x n``, would be cheaper; below this size
the difference is too small to mention.
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
    The analytic method's solution: the factors of one problem, read at one gamma.

    ``_analytic`` factors its problem into this once (:meth:`factor`), and the
    :class:`~fips.Posterior` asks it for every array it reports: it is the
    analytic method's :class:`~fips.posterior.Solution`. The factors depend
    neither on gamma nor on the values of the observations, so
    :meth:`with_gamma` and :meth:`with_data` share them. Nothing
    :math:`n \times n` is formed unless the full posterior covariance or the
    gain is asked for. The arrays are read-only by convention.

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
    HS_0H: LinearOperator
    r""":math:`H Q H^T` (``m x m``) in the cheapest form of its factors: ``factor`` formed it, and the posterior reuses it."""

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
        d = problem.obs.values - problem.predict(problem.prior).values
        HQ = HS_0.dense()
        HQH = HS_0H.dense()
        S_z = problem.obs_cov.data
        m, n = HQ.shape
        if m > n and m >= _LARGE_M:
            logger.warning(
                "The whitened solve eigendecomposes an m x m matrix, with m = %d "
                "observations, more than the n = %d state elements: its time grows "
                "as m^3. The state-space (information) form, n x n, would be "
                "cheaper; fips does not have it yet. Aggregating the observations "
                "makes m smaller.",
                m,
                n,
            )
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
        w, V = eigh(
            M, overwrite_a=True
        )  # driver "evr": less memory than "evd", as fast
        eigenvalues = np.asarray(w)
        _clip(eigenvalues)
        B = whiten(V)
        return cls(
            problem=problem,
            gamma=gamma,
            eigenvalues=eigenvalues,
            whitener=B,
            W=B @ HQ,
            innovation=B @ d,
            prior_variances=np.asarray(S_0.diagonal(), dtype=float),
            HS_0H=HS_0H,
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
        d = problem.obs.values - problem.predict(problem.prior).values
        return dataclasses.replace(self, problem=problem, innovation=self.whitener @ d)


def _clip(eigenvalues: np.ndarray) -> None:
    """
    Set the negative eigenvalues of a positive semidefinite matrix to zero, in place.

    Roundoff leaves eigenvalues of order ``m * eps * max|lambda|`` that should
    be zero, some negative. One beyond that is logged: the inputs are not what
    they should be (a covariance that is not positive semidefinite).
    """
    low = eigenvalues.min(initial=0.0)
    if low < 0:
        roundoff = len(eigenvalues) * np.finfo(float).eps * np.abs(eigenvalues).max()
        if -low > roundoff:
            logger.warning(
                "The whitened prior in observation space has a negative eigenvalue, "
                "%.3g, larger than roundoff (%.3g); it and the other negative "
                "eigenvalues are set to zero. Check that the prior error covariance "
                "is positive semidefinite.",
                low,
                roundoff,
            )
        np.maximum(eigenvalues, 0.0, out=eigenvalues)


def _analytic(problem: Problem, options: AnalyticOptions) -> Posterior:
    """Factor the problem's whitened system once; the posterior asks the factors for everything."""
    return Posterior(problem, Whitened.factor(problem, options.gamma), options)

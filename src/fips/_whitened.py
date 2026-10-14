r"""
The whitened factorization of a linear Gaussian inverse problem.

:class:`~fips.estimators.BayesianSolver` factors its problem here, once, and
derives everything it reports from the factors, for any regularization factor
:math:`\gamma`. Nothing :math:`n \times n` is formed unless the full posterior
covariance or averaging kernel is asked for.

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

import logging
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import cholesky, eigh, solve_triangular

from fips.linalg import DiagonalOperator, LinearOperator

logger = logging.getLogger(__name__)

_LARGE_M = 1000
"""
The number of observations from which a solve with more observations than state elements logs a warning.

The whitened solve eigendecomposes an ``m x m`` matrix. When ``m > n`` the
state-space (information) form, ``n x n``, would be cheaper; below this size
the difference is too small to mention.
"""


@dataclass(frozen=True, eq=False)
class Whitened:
    """
    The factors of one problem, which do not depend on gamma.

    Build it with :meth:`factor`. The arrays are read-only by convention: a
    :meth:`~fips.estimators.BayesianSolver.with_gamma` solver shares them.
    """

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
    def factor(
        cls,
        HQ: np.ndarray,
        HQH: np.ndarray,
        S_z: LinearOperator,
        d: np.ndarray,
        prior_variances: np.ndarray,
    ) -> "Whitened":
        """
        Factor a problem.

        Parameters
        ----------
        HQ : np.ndarray
            :math:`H Q`, dense, ``m x n``.
        HQH : np.ndarray
            :math:`H Q H^T`, dense, ``m x m``.
        S_z : LinearOperator
            The model-data mismatch, before gamma divides it.
        d : np.ndarray
            The innovation :math:`z - H x_0 - c`.
        prior_variances : np.ndarray
            The diagonal of :math:`Q`.

        Returns
        -------
        Whitened
            The factors.

        Raises
        ------
        numpy.linalg.LinAlgError
            If ``S_z`` is not positive-definite.
        """
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
            eigenvalues=eigenvalues,
            whitener=B,
            W=B @ HQ,
            innovation=B @ d,
            prior_variances=np.asarray(prior_variances, dtype=float),
        )

    def weights(self, gamma: float) -> np.ndarray:
        r"""Return :math:`w_k = \gamma / (\gamma \lambda_k + 1)`."""
        return gamma / (gamma * self.eigenvalues + 1.0)

    def increment(self, gamma: float) -> np.ndarray:
        r"""Return :math:`\hat{x} - x_0 = W^T (w \circ e)`."""
        return self.W.T @ (self.weights(gamma) * self.innovation)

    def variances(self, gamma: float) -> np.ndarray:
        r"""Return the diagonal of :math:`\hat{S}`, :math:`O(m n)`."""
        w = self.weights(gamma)
        return self.prior_variances - np.einsum("k,kj,kj->j", w, self.W, self.W)

    def reduction(self, gamma: float) -> np.ndarray:
        r"""Return :math:`Q - \hat{S} = W^T \operatorname{diag}(w) W`, ``n x n``."""
        Y = np.sqrt(self.weights(gamma))[:, None] * self.W
        return Y.T @ Y

    def gain(self, gamma: float) -> np.ndarray:
        r"""Return the gain :math:`K = W^T \operatorname{diag}(w) B`, ``n x m``."""
        return self.W.T @ (self.weights(gamma)[:, None] * self.whitener)

    def dofs(self, gamma: float) -> float:
        r"""Return :math:`\sum_k \gamma \lambda_k / (\gamma \lambda_k + 1)`."""
        return float(np.sum(self.weights(gamma) * self.eigenvalues))

    def chi2(self, gamma: float) -> float:
        r"""Return :math:`d^T (H Q H^T + S_z/\gamma)^{-1} d = \sum_k w_k e_k^2`."""
        return float(np.sum(self.weights(gamma) * self.innovation**2))

    def to_dict(self) -> dict[str, Any]:
        """Return the arrays by name, for a pickle that any fips can read."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


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

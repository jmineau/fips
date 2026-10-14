"""
Inversion estimators for solving inverse problems.

This module contains Bayesian and regularized estimators for state estimation
in linear inverse problems, computing posterior distributions and diagnostics.
"""

import logging
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.linalg import LinAlgError
from scipy.linalg import inv, solve

from fips._whitened import Whitened
from fips.operator import (
    DenseOperator,
    DiagonalOperator,
    Operator,
    OperatorLike,
    SparseOperator,
    _solve_psd,
    as_operator,
)

logger = logging.getLogger(__name__)

# The matrices an estimator computes with, as operators under private names,
# with the public attribute that gives each as before (an array, or a CSR
# matrix for a sparse input). A pickle stores each under its public name as
# the array (or SciPy sparse matrix) it holds, so it reads on a fips that has
# no operators, and one written by such a fips reads here. A Kronecker
# covariance is stored as itself: its array is the n x n it never forms.
_OPERATOR_NAMES = {
    "H": "_H_op",
    "S_0": "_S_0_op",
    "S_z": "_S_z_op",
    "_S_z_orig": "_S_z_orig_op",
}
# Products cached as operators, pickled the same way under their own names.
_CACHED_OPERATORS = ("_HS_0", "_HS_0H")


def _times_pseudo_inverse(XWt: np.ndarray, W: Operator) -> np.ndarray:
    """
    Return ``X W*`` given ``X W^T``, where ``W* = W^T (W W^T)^-1`` is the pseudo-inverse of `W`.

    Raises
    ------
    ValueError
        If the rows of `W` are linearly dependent, so ``W W^T`` is singular.
    """
    G = (W @ W.T).dense()
    try:
        # X W^T (W W^T)^-1 is the transpose of (W W^T)^-1 (X W^T)^T
        return solve(G, XWt.T, assume_a="pos").T
    except LinAlgError as err:
        raise ValueError(
            "The rows of the aggregation are linearly dependent, so it has no "
            "pseudo-inverse."
        ) from err


def _raw(op: Operator):
    """Return the array or SciPy sparse matrix an operator holds; a form of its own (Kronecker) stays itself."""
    if isinstance(op, DenseOperator):
        return op.array
    if isinstance(op, (SparseOperator, DiagonalOperator)):
        return op._csr()
    return op


def _public(op: Operator) -> "np.ndarray | sp.csr_matrix":
    """
    Return an operator as the estimator's public attributes give a matrix.

    A dense operator gives its array, read-only, so that an edit cannot change
    the estimator's later outputs; a sparse or diagonal one a CSR matrix (what
    a sparse input gave before the estimator held operators); any other form
    its dense matrix, formed on each call.
    """
    if isinstance(op, DenseOperator):
        view = op.array.view()
        view.flags.writeable = False
        return view
    csr = op._csr()
    if csr is not None:
        return sp.csr_matrix(csr)
    return op.dense()


# An alias, because pyrefly reads ``type[...]`` written inside a base class's
# arguments as ``type[Any]``, so a lookup would lose its type.
_EstimatorClass = type["Estimator"]


class EstimatorRegistry(dict[str, _EstimatorClass]):
    """
    Registry for estimator classes.
    """

    def register(self, name: str):
        """
        Register an estimator class under a given name.

        Parameters
        ----------
        name : str
            Name to register the estimator under.

        Returns
        -------
        decorator : function
            Decorator to register the class.
        """

        def decorator(cls: type[Estimator]) -> type[Estimator]:
            """Register ``cls`` under ``name`` and return it unchanged."""
            self[name] = cls
            return cls

        return decorator


ESTIMATOR_REGISTRY = EstimatorRegistry()


def available_estimators() -> list[str]:
    """
    Return the names of all registered estimators.

    Returns
    -------
    list of str
        Names that can be passed to :func:`fips.solve` and
        :meth:`~fips.InverseProblem.solve`.
    """
    return list(ESTIMATOR_REGISTRY.keys())


class Estimator(ABC):
    """
    Base inversion estimator class.

    The matrices may be dense arrays, SciPy sparse matrices or
    :class:`~fips.operator.Operator`; the estimator computes with each as an
    operator (:func:`~fips.operator.as_operator`), so a sparse or diagonal
    covariance stays sparse or diagonal through the solve. Its attributes
    :attr:`H`, :attr:`S_0` and :attr:`S_z` give each matrix as an array (a
    CSR matrix for a sparse one).

    Parameters
    ----------
    z : np.ndarray
        Observed data.
    x_0 : np.ndarray
        Prior model state estimate.
    H : np.ndarray, SciPy sparse matrix or Operator
        Forward operator.
    S_0 : np.ndarray, SciPy sparse matrix or Operator
        Prior error covariance.
    S_z : np.ndarray, SciPy sparse matrix or Operator
        Model-data mismatch covariance.
    c : np.ndarray or float, optional
        Constant data, defaults to 0.0.
    """

    z: np.ndarray
    """Observed data."""
    x_0: np.ndarray
    """Prior model state estimate."""
    _H_op: Operator
    _S_0_op: Operator
    _S_z_op: Operator
    c: np.ndarray
    """Constant data, which can be a scalar or an array matching the shape of z."""
    n_z: int
    """Number of observations (length of z)."""
    n_x: int
    """Number of state variables (length of x_0)."""

    def __init__(
        self,
        z: np.ndarray,
        x_0: np.ndarray,
        H: OperatorLike,
        S_0: OperatorLike,
        S_z: OperatorLike,
        c: np.ndarray | float | None = None,
    ):
        self.z = z.astype(float)
        self.x_0 = x_0.astype(float)
        self._H_op = as_operator(H)
        self._S_0_op = as_operator(S_0)
        self._S_z_op = as_operator(S_z)

        # Handle optional constant data
        if c is None:
            c = 0.0
        if isinstance(c, (int, float)):
            self.c = np.full(self.z.shape, float(c)).astype(float)
        elif isinstance(c, np.ndarray):
            self.c = c.astype(float)
        else:
            raise TypeError("c must be None, a float, or a numpy ndarray.")

        self.n_z = z.shape[0]
        self.n_x = x_0.shape[0]

    @property
    def H(self) -> "np.ndarray | sp.csr_matrix":
        """
        Forward operator.

        An array (read-only), or a CSR matrix when it is sparse. The estimator
        computes with it as an :class:`~fips.operator.Operator`.
        """
        return _public(self._H_op)

    @property
    def S_0(self) -> "np.ndarray | sp.csr_matrix":
        """
        Prior error covariance.

        An array (read-only), or a CSR matrix when it is sparse or diagonal. A
        covariance kept in another form (as Kronecker factors) is formed as a
        dense array each time this is read; the estimator computes with its
        :class:`~fips.operator.Operator`.
        """
        return _public(self._S_0_op)

    @property
    def S_z(self) -> "np.ndarray | sp.csr_matrix":
        """
        Model-data mismatch covariance.

        An array (read-only), or a CSR matrix when it is sparse or diagonal.
        The estimator computes with it as an :class:`~fips.operator.Operator`.
        """
        return _public(self._S_z_op)

    def __getstate__(self) -> dict[str, Any]:
        """Return the state for pickling, with each operator as the array it holds."""
        state = self.__dict__.copy()
        for public, private in _OPERATOR_NAMES.items():
            if private in state:
                state[public] = _raw(state.pop(private))
        for name in _CACHED_OPERATORS:
            if name in state:
                state[name] = _raw(state[name])
        return state

    def __setstate__(self, state: dict[str, Any]):
        """Restore the state, holding each matrix as an operator again."""
        for public, private in _OPERATOR_NAMES.items():
            if public in state:
                state[private] = as_operator(state.pop(public))
        for name in _CACHED_OPERATORS:
            if name in state:
                state[name] = as_operator(state[name])
        state.pop("_H_T", None)  # cached by earlier versions; no longer used
        self.__dict__.update(state)

    def forward(self, x) -> np.ndarray:
        """
        Forward model calculation.

        .. math::
            y = Hx + c

        Parameters
        ----------
        x : np.ndarray
            State vector.

        Returns
        -------
        np.ndarray
            Model output (Hx + c).
        """
        logger.debug("Performing forward calculation...")
        return self._H_op @ x + self.c

    def residual(self, x) -> np.ndarray:
        """
        Forward model residual.

        .. math::
            r = z - (Hx + c)

        Parameters
        ----------
        x : np.ndarray
            State vector.

        Returns
        -------
        np.ndarray
            Residual (z - (Hx + c)).
        """
        logger.debug("Performing residual calculation...")
        return self.z - self.forward(x)

    def leverage(self, x) -> np.ndarray:
        """
        Calculate the leverage matrix.

        Which observations are likely to have more impact on the solution.

        .. math::
            L = Hx ((Hx)^T (H S_0 H^T + S_z)^{-1} Hx)^{-1} (Hx)^T (H S_0 H^T + S_z)^{-1}

        This is the generalized-least-squares hat matrix of the one regressor
        :math:`Hx`: the projection onto :math:`Hx` in the metric
        :math:`(H S_0 H^T + S_z)^{-1}`. It is idempotent and its trace is 1, so
        the diagonal shares one unit of leverage among the observations. When
        the errors are correlated an entry of the diagonal can be negative.

        Parameters
        ----------
        x : np.ndarray
            State vector, of length ``n_x``.

        Returns
        -------
        np.ndarray
            Leverage matrix, ``n_z`` by ``n_z``.

        Notes
        -----
        ``Hx`` is ``forward(x)``, which includes the constant ``c``.
        """
        logger.debug("Calculating Leverage matrix...")
        Hx = np.asarray(self.forward(x), dtype=float)
        if Hx.ndim == 1:
            # One regressor: the vector is the single column of the design
            # matrix, so (Hx)^T A^-1 Hx below is 1 x 1, not a scalar.
            Hx = Hx[:, None]
        Hx_T = Hx.T

        # Equation: L = Hx (Hx^T (H S_0 H^T + S_z)^-1 Hx)^-1 Hx^T (H S_0 H^T + S_z)^-1
        # Let A = (H S_0 H^T + S_z)
        A = (self._HS_0H + self._S_z_op).dense()

        # 1. Compute term_1 = A^-1 @ Hx using solve
        term_1 = _solve_psd(A, Hx)

        # 2. Compute the inner inverse: (Hx^T @ term_1)^-1
        # We still have to use standard inv() here because Hx^T @ term_1 isn't
        # guaranteed to be a standard symmetric positive-definite covariance matrix.
        inner_inv = inv(Hx_T @ term_1)

        # 3. Assemble: Hx @ inner_inv @ term_1^T
        return Hx @ inner_inv @ term_1.T

    @abstractmethod
    def cost(self, x) -> float:
        """
        Cost/loss/misfit function.

        Parameters
        ----------
        x : np.ndarray
            State vector.

        Returns
        -------
        float
            Cost value.
        """
        logger.debug("Performing cost calculation...")
        raise NotImplementedError

    @property
    @abstractmethod
    def x_hat(self) -> np.ndarray:
        """
        Posterior mean model state estimate (solution).

        Returns
        -------
        np.ndarray
            Posterior state estimate.
        """
        logger.debug("Calculating Posterior Mean Model State Estimate...")
        raise NotImplementedError

    @property
    @abstractmethod
    def S_hat(self) -> np.ndarray:
        """
        Posterior error covariance matrix.

        Returns
        -------
        np.ndarray
            Posterior error covariance matrix.
        """
        logger.debug("Calculating Posterior Error Covariance Matrix...")
        raise NotImplementedError

    @cached_property
    def y_hat(self) -> np.ndarray:
        r"""
        Posterior mean observation estimate.

        .. math::
            \\hat{y} = H \\hat{x} + c

        Returns
        -------
        np.ndarray
            Posterior observation estimate.
        """
        logger.debug("Calculating Posterior Mean Observation Estimate...")
        return self.forward(self.x_hat)

    @cached_property
    def y_0(self) -> np.ndarray:
        r"""
        Prior mean data estimate.

        .. math::
            \\hat{y}_0 = H x_0 + c

        Returns
        -------
        np.ndarray
            Prior data estimate.
        """
        logger.debug("Calculating Prior Mean Data Estimate...")
        return self.forward(self.x_0)

    @cached_property
    def K(self):
        """
        Kalman gain matrix.

        .. math::
            K = (H S_0)^T (H S_0 H^T + S_z)^{-1}

        Returns
        -------
        np.ndarray
            Kalman gain matrix.
        """
        logger.debug("Calculating Kalman Gain Matrix...")
        # We want K = (H S_0)^T (H S_0 H^T + S_z)^-1
        # Let A = (H S_0 H^T + S_z) and B = (H S_0).
        # Since A is symmetric, solve(A, B) computes A^-1 B.
        # So K = (A^-1 B)^T = solve(A, B).T
        A = self._HS_0H + self._S_z_op
        return A.solve(self._HS_0).T

    @cached_property
    def A(self):
        """
        Averaging kernel matrix.

        .. math::
            A = KH = (H S_0)^T (H S_0 H^T + S_z)^{-1} H

        Returns
        -------
        np.ndarray
            Averaging kernel matrix.
        """
        logger.debug("Calculating Averaging Kernel Matrix...")
        return self.K @ self._H_op

    @cached_property
    def S_hat_diagonal(self) -> np.ndarray:
        r"""
        Posterior error variances, the diagonal of :math:`\hat{S}`.

        Read from :attr:`S_hat` here; an estimator that can compute them
        without the ``n x n`` matrix (:class:`~fips.estimators.BayesianSolver`)
        does.

        Returns
        -------
        np.ndarray
            The ``n_x`` variances.
        """
        return np.diag(np.asarray(self.S_hat)).copy()

    def aggregate_S_hat(self, W: OperatorLike) -> np.ndarray:
        r"""
        Posterior error covariance of an aggregate of the state, :math:`W \hat{S} W^T`.

        Parameters
        ----------
        W : array-like, SciPy sparse matrix or Operator
            The aggregation, ``k x n_x``: row ``i`` gives the weight of each
            state element in aggregate ``i`` (ones for a sum, ``1/count`` for
            a mean).

        Returns
        -------
        np.ndarray
            The ``k x k`` covariance.
        """
        W = as_operator(W)
        return (W @ (W @ np.asarray(self.S_hat)).T).T

    def aggregate_A(self, W: OperatorLike) -> np.ndarray:
        r"""
        Averaging kernel of an aggregate of the state, :math:`W A W^*`.

        :math:`W^* = W^T (W W^T)^{-1}` is the pseudo-inverse of :math:`W`: it
        spreads a change of the aggregates back over the state elements (for a
        sum over a set of elements, evenly over them), so that :math:`W A W^*`
        is the sensitivity of the aggregated estimate to the aggregated true
        state. This is the form IMI uses; :math:`W A W^T` is not an averaging
        kernel (its trace is not the aggregate's DOFS).

        Parameters
        ----------
        W : array-like, SciPy sparse matrix or Operator
            The aggregation, ``k x n_x``, with rows that are linearly
            independent.

        Returns
        -------
        np.ndarray
            The ``k x k`` averaging kernel.

        Raises
        ------
        ValueError
            If the rows of `W` are linearly dependent.
        """
        W = as_operator(W)
        WA = W @ np.asarray(self.A)
        return _times_pseudo_inverse((W @ WA.T).T, W)

    @cached_property
    def _HS_0(self) -> Operator:
        r"""
        Compute H S_0, in the cheapest form of the two.

        .. math::
            H S_0
        """
        return self._H_op @ self._S_0_op

    @cached_property
    def _HS_0H(self) -> Operator:
        r"""
        Compute H S_0 H^T, in the cheapest form of the three.

        .. math::
            H S_0 H^T
        """
        return self._HS_0 @ self._H_op.T

    @cached_property
    def S_y0(self) -> np.ndarray:
        r"""
        Prior error covariance in observation space.

        .. math::
            S_{y_0} = H S_0 H^T

        The covariance of the prior modelled observations ``y_0`` that the
        prior error ``S_0`` implies: how uncertain each modelled observation
        is because the prior state is, and how those uncertainties co-vary
        between observations that see the same state. The square root of its
        diagonal is the prior's share of the error budget at each
        observation, alongside the model-data mismatch ``S_z``.
        A dense array.
        """
        return self._HS_0H.dense()

    @cached_property
    def DOFS(self) -> float:
        """
        Degrees Of Freedom for Signal (DOFS).

        .. math::
            DOFS = Tr(A)

        Returns
        -------
        float
            Degrees of Freedom value.
        """
        return float(np.trace(self.A))

    @cached_property
    def reduced_chi2(self) -> float:
        r"""
        Reduced Chi-squared statistic. Tarantola (1987).

        .. math::
            \\chi^2 = \\frac{1}{n_z} ((z - H\\hat{x})^T S_z^{-1} (z - H\\hat{x}) + (\\hat{x} - x_0)^T S_0^{-1} (\\hat{x} - x_0))

        .. note::
            I can't find a copy of Tarantola (1987) to verify this equation, but it appears in
            Kunik et al. (2019) https://doi.org/10.1525/elementa.375

        Returns
        -------
        float
            Reduced Chi-squared value.
        """
        logger.debug("Calculating Reduced Chi-squared statistic...")
        data_residual = self.residual(self.x_hat)
        model_residual = self.x_hat - self.x_0

        # The Chi-square involves Mahalanobis distance terms: r^T * S^-1 * r
        # Explicitly inverting the S_z and S_0 matrices just to multiply a vector
        # is slow and memory-intensive.
        #
        # Instead, we solve the system: S * y = r  -->  y = S^-1 * r
        # Then we compute: r^T @ y. Operator.solve does it the way the
        # covariance is stored (a division for a diagonal, a sparse LU for a
        # sparse one); a dense one falls back to a symmetric LDL^T solve if
        # roundoff (e.g. after aggregation) tips it off positive-definite.

        # 1. Scaled Data Misfit: data_residual^T @ (S_z^-1 @ data_residual)
        scaled_data_misfit = data_residual.T @ self._S_z_op.solve(data_residual)

        # 2. Scaled Model Misfit: model_residual^T @ (S_0^-1 @ model_residual)
        scaled_model_misfit = model_residual.T @ self._S_0_op.solve(model_residual)

        return float((scaled_data_misfit + scaled_model_misfit) / self.n_z)

    @cached_property
    def R2(self) -> float:
        r"""
        Coefficient of determination (R-squared).

        .. math::
            R^2 = corr(z, H\\hat{x})^2

        Returns
        -------
        float
            R-squared value.
        """
        logger.debug("Calculating Coefficient of determination (R-squared)...")
        return float(np.corrcoef(self.z, self.y_hat)[0, 1] ** 2)

    @cached_property
    def RMSE(self) -> float:
        r"""
        Root mean square error (RMSE).

        .. math::
            RMSE = \\sqrt{\\frac{(z - H\\hat{x})^2}{n_z}}

        Returns
        -------
        float
            RMSE value.
        """
        logger.debug("Calculating Root Mean Square Error (RMSE)...")
        r = self.residual(self.x_hat)
        return float(np.sqrt((r**2).sum() / self.n_z))

    @cached_property
    def uncertainty_reduction(self) -> float:
        r"""
        Uncertainty reduction metric.

        .. math::
            U_{red} = 1 - \\frac{\\sqrt{trace(\\hat{S})}}{\\sqrt{trace(S_0)}}

        Returns
        -------
        float
            Uncertainty reduction value.
        """
        logger.debug("Calculating uncertainty reduction metric...")
        return float(
            1 - (np.sqrt(self.S_hat_diagonal.sum()) / np.sqrt(self._S_0_op.trace()))
        )

    @cached_property
    def desroziers(self) -> np.ndarray:
        r"""
        Desroziers et al. (2005) diagnosed observation error covariance.

        Estimates the observation error covariance from the innovation
        (prior departure) and analysis departure (posterior residual):

        .. math::
            R_{diag} = \frac{1}{2}(d^a {d^b}^T + d^b {d^a}^T)

        where :math:`d^b = z - Hx_0 - c` (innovation) and
        :math:`d^a = z - H\hat{x} - c` (analysis departure).

        Symmetrized since the single-realization outer product is not
        symmetric but the expectation is.

        Returns
        -------
        np.ndarray
            Diagnosed observation error covariance matrix (n_z × n_z).

        References
        ----------
        Desroziers, G., Berre, L., Chapnik, B., & Poli, P. (2005).
        Diagnosis of observation, background and analysis-error statistics
        in observation space. Q.J.R. Meteorol. Soc., 131, 3385–3396.
        """
        d_b = self.residual(self.x_0)  # innovation
        d_a = self.residual(self.x_hat)  # analysis departure
        raw = np.outer(d_a, d_b)
        return 0.5 * (raw + raw.T)

    @cached_property
    def U_red(self) -> np.ndarray:
        r"""
        Uncertainty reduction vector.

        .. math::
            U_{red} = \\left( 1 - \\frac{\\sqrt{diag(\\hat{S})}}{\\sqrt{diag(S_0)}} \\right) * 100\\%

        Returns
        -------
        np.ndarray
            Uncertainty reduction vector.
        """
        logger.debug("Calculating uncertainty reduction vector...")
        return (
            1 - (np.sqrt(self.S_hat_diagonal) / np.sqrt(self._S_0_op.diagonal()))
        ) * 100.0


@ESTIMATOR_REGISTRY.register("bayesian")
class BayesianSolver(Estimator):
    r"""
    Bayesian inversion estimator class.

    This class implements a Bayesian inversion framework for solving inverse problems,
    also known as the batch method.

    The regularization factor (gamma) controls the balance between fitting observations and
    staying close to the prior. In the cost function, gamma (γ) multiplies the observation term:
    J(x) = (x - x_0)^T S_0^{-1} (x - x_0) + gamma*(z - Hx)^T S_z^{-1} (z - Hx)

    This amounts to scaling S_z by 1/gamma.

    - gamma > 1: Increases weight on data fitting (less regularization, fits observations more closely)
    - gamma = 1: Standard Bayesian inversion (default)
    - gamma < 1: Decreases weight on data fitting (more regularization, stays closer to prior)

    **The solve.** The solver whitens the problem once: with :math:`S_z = L L^T`
    (Cholesky, or the square root of a diagonal), it eigendecomposes the
    ``m x m`` matrix :math:`L^{-1} H S_0 H^T L^{-T} = V \Lambda V^T` and keeps
    :math:`\Lambda`, :math:`V^T L^{-1}` and :math:`W = V^T L^{-1} H S_0`
    (``m x n``). The estimate, the posterior variances
    (:attr:`S_hat_diagonal`), the aggregated covariances and averaging kernels,
    :attr:`DOFS` and :attr:`reduced_chi2` come from these factors, and nothing
    ``n x n`` is formed unless :attr:`S_hat` or :attr:`A` is read.
    :meth:`with_gamma` reuses the factors for another gamma. The eigenvalues
    of that matrix, which is positive semidefinite, are clipped at zero, and a
    clip larger than roundoff is logged. With more observations than state
    elements (``m > n``) the state-space form would be cheaper; it is not
    built, and a large such solve logs a warning.

    Parameters
    ----------
    z : np.ndarray
        Observed data
    x_0 : np.ndarray
        Prior model estimate
    H : np.ndarray, SciPy sparse matrix or Operator
        Forward operator
    S_0 : np.ndarray, SciPy sparse matrix or Operator
        Prior error covariance
    S_z : np.ndarray, SciPy sparse matrix or Operator
        Model-data mismatch covariance, positive-definite
    c : np.ndarray | float, optional
        Constant data, defaults to 0.0
    gamma : float, optional
        Regularization factor (γ) that multiplies the observation term in the cost function, by default 1.0
    """

    gamma: float
    """
    Regularization factor (γ) that multiplies the observation term in the cost function.
    Values > 1 increase weight on data fitting (less regularization),
    values < 1 decrease weight on data fitting (more regularization).
    """
    _S_z_orig_op: Operator

    def __init__(
        self,
        z: np.ndarray,
        x_0: np.ndarray,
        H: OperatorLike,
        S_0: OperatorLike,
        S_z: OperatorLike,
        c: np.ndarray | float | None = None,
        gamma: float = 1.0,
    ):
        # Store original S_z and gamma, then pass regularized S_z to parent
        # Note: S_z is scaled by 1/gamma so that S_z^{-1} is scaled by gamma in the cost function
        self._S_z_orig_op = as_operator(S_z)
        self.gamma = gamma
        S_z_reg = self._S_z_orig_op / gamma
        super().__init__(z=z, x_0=x_0, H=H, S_0=S_0, S_z=S_z_reg, c=c)

    @property
    def _S_z_orig(self) -> "np.ndarray | sp.csr_matrix":
        """Original observation error covariance before regularization, as :attr:`S_z` gives it."""
        return _public(self._S_z_orig_op)

    def __repr__(self) -> str:
        """Return string representation."""
        solved = "x_hat" in self.__dict__
        return (
            f"BayesianSolver("
            f"n_x={self.n_x}, n_z={self.n_z}, "
            f"gamma={self.gamma}, solved={solved})"
        )

    def __getstate__(self) -> dict[str, Any]:
        """
        Return the state for pickling: the factors, and nothing ``n x n``.

        The factors are stored as a dict of arrays. The cached ``S_hat``,
        ``A``, ``K`` and ``H S_0`` are dropped, since the factors give them
        again; a solver that has not factored keeps what it has.
        """
        state = super().__getstate__()
        whitened = state.pop("_whitened", None)
        if whitened is not None:
            state["_whitened"] = whitened.to_dict()
            for name in ("S_hat", "A", "K", "_HS_0"):
                state.pop(name, None)
        return state

    def __setstate__(self, state: dict[str, Any]):
        """Restore the state, the factors included."""
        whitened = state.pop("_whitened", None)
        super().__setstate__(state)
        if whitened is not None:
            self.__dict__["_whitened"] = Whitened(**whitened)

    @cached_property
    def _whitened(self) -> Whitened:
        """The whitened factorization of the problem, which does not depend on gamma."""
        logger.debug("Factoring the whitened problem...")
        HS_0 = self._H_op @ self._S_0_op
        HS_0H = self.__dict__.get("_HS_0H")
        if HS_0H is None:
            HS_0H = HS_0 @ self._H_op.T
            self.__dict__["_HS_0H"] = HS_0H  # S_y0 reads it
        return Whitened.factor(
            HS_0.dense(),
            HS_0H.dense(),
            self._S_z_orig_op,
            self.residual(self.x_0),
            self._S_0_op.diagonal(),
        )

    def with_gamma(self, gamma: float) -> "BayesianSolver":
        """
        Return the solver for another regularization factor, without factoring again.

        The new solver shares this one's inputs and factors, so it costs
        :math:`O(m n)` rather than a new solve.

        Parameters
        ----------
        gamma : float
            The new regularization factor.

        Returns
        -------
        BayesianSolver
            A solver with the same problem and ``gamma``; its outputs are
            those of ``BayesianSolver(..., gamma=gamma)``.

        Examples
        --------
        >>> import numpy as np
        >>> from fips.estimators import BayesianSolver
        >>> H = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        >>> solver = BayesianSolver(
        ...     np.array([1.0, 2.0, 3.0]), np.zeros(2), H, np.eye(2), 0.1 * np.eye(3)
        ... )
        >>> half = solver.with_gamma(0.5)
        >>> fresh = BayesianSolver(
        ...     np.array([1.0, 2.0, 3.0]),
        ...     np.zeros(2),
        ...     H,
        ...     np.eye(2),
        ...     0.1 * np.eye(3),
        ...     gamma=0.5,
        ... )
        >>> bool(np.allclose(half.x_hat, fresh.x_hat))
        True
        """
        new = type(self)(
            z=self.z,
            x_0=self.x_0,
            H=self._H_op,
            S_0=self._S_0_op,
            S_z=self._S_z_orig_op,
            c=self.c,
            gamma=gamma,
        )
        new.__dict__["_whitened"] = self._whitened
        if "_HS_0H" in self.__dict__:
            new.__dict__["_HS_0H"] = self.__dict__["_HS_0H"]
        return new

    def cost(self, x):
        r"""
        Cost function with regularization.

        .. math::
            J(x) = \\frac{1}{2}(x - x_0)^T S_0^{-1}(x - x_0) + λ\\frac{1}{2}(z - Hx - c)^T S_z^{-1}(z - Hx - c)

        Note: S_z has already been scaled by the regularization factor gamma during initialization.
        """
        logger.debug("Performing cost calculation...")
        diff_model = x - self.x_0
        diff_data = self.residual(x)

        # Like the Chi-square, the cost function relies on r^T * S^-1 * r.
        # We replace inv(S) @ r with solve(S, r) to avoid calculating the full inverse.

        cost_model = diff_model.T @ self._S_0_op.solve(diff_model)
        cost_data = diff_data.T @ self._S_z_op.solve(diff_data)

        return 0.5 * (cost_model + cost_data)

    @cached_property
    def x_hat(self):  # type: ignore[override]
        r"""
        Posterior Mean Model Estimate (solution).

        .. math::
            \hat{x} = x_0 + K(z - Hx_0 - c) = x_0 + W^T \operatorname{diag}(w) e

        From the factors: :math:`e` is the whitened innovation and
        :math:`w_k = \gamma / (\gamma \lambda_k + 1)` (see the class).
        """
        logger.debug("Calculating Posterior Mean Model Estimate...")
        return self.x_0 + self._whitened.increment(self.gamma)

    @cached_property
    def K(self):
        r"""
        Kalman gain matrix.

        .. math::
            K = (H S_0)^T (H S_0 H^T + S_z)^{-1} = W^T \operatorname{diag}(w) V^T L^{-1}

        From the factors, ``n x m``.
        """
        logger.debug("Calculating Kalman Gain Matrix...")
        return self._whitened.gain(self.gamma)

    @cached_property
    def S_hat(self):  # type: ignore[override]
        r"""
        Posterior Error Covariance Matrix with regularization.

        .. math::
            \hat{S} = (H^T S_z^{-1} H + S_0^{-1})^{-1}
                = S_0 - (H S_0)^T(H S_0 H^T + S_z)^{-1}(H S_0)
                = S_0 - W^T \operatorname{diag}(w) W

        Dense, ``n x n``: reading it forms the matrix (and a Kronecker
        :math:`S_0` too). :attr:`S_hat_diagonal` and :meth:`aggregate_S_hat`
        do not.

        Note: S_z has already been scaled by the regularization factor gamma during initialization.
        """
        logger.debug("Calculating Posterior Error Covariance Matrix...")
        S_hat = self._whitened.reduction(self.gamma)
        np.subtract(self._S_0_op.dense(), S_hat, out=S_hat)
        return S_hat

    @cached_property
    def S_hat_diagonal(self) -> np.ndarray:
        r"""
        Posterior error variances, the diagonal of :math:`\hat{S}`.

        .. math::
            \operatorname{diag}(\hat{S})_j = (S_0)_{jj} - \sum_k w_k W_{kj}^2

        From the factors, in :math:`O(m n)`, without :attr:`S_hat`.

        Returns
        -------
        np.ndarray
            The ``n_x`` variances.
        """
        return self._whitened.variances(self.gamma)

    def aggregate_S_hat(self, W: OperatorLike) -> np.ndarray:
        r"""
        Posterior error covariance of an aggregate of the state, :math:`W \hat{S} W^T`.

        .. math::
            W \hat{S} W^T = W S_0 W^T - (W W_f^T) \operatorname{diag}(w) (W W_f^T)^T

        where :math:`W_f` is the factors' ``m x n`` matrix. Nothing ``n x n``
        is formed: :math:`S_0` is applied to the ``k`` columns of
        :math:`W^T` (through its factors, for a Kronecker covariance).

        Parameters
        ----------
        W : array-like, SciPy sparse matrix or Operator
            The aggregation, ``k x n_x``: row ``i`` gives the weight of each
            state element in aggregate ``i`` (ones for a sum, ``1/count`` for
            a mean).

        Returns
        -------
        np.ndarray
            The ``k x k`` covariance.
        """
        W = as_operator(W)
        prior = np.asarray(W @ (self._S_0_op @ W.T))
        WWf = W @ self._whitened.W.T  # k x m
        return prior - (WWf * self._whitened.weights(self.gamma)) @ WWf.T

    def aggregate_A(self, W: OperatorLike) -> np.ndarray:
        r"""
        Averaging kernel of an aggregate of the state, :math:`W A W^*`.

        :math:`W^* = W^T (W W^T)^{-1}` is the pseudo-inverse of :math:`W` (see
        :meth:`Estimator.aggregate_A`). Computed as :math:`(W K)(H W^T)(W
        W^T)^{-1}` from the factors, without the ``n x n`` :attr:`A`.

        Parameters
        ----------
        W : array-like, SciPy sparse matrix or Operator
            The aggregation, ``k x n_x``, with rows that are linearly
            independent.

        Returns
        -------
        np.ndarray
            The ``k x k`` averaging kernel.

        Raises
        ------
        ValueError
            If the rows of `W` are linearly dependent.
        """
        W = as_operator(W)
        f = self._whitened
        WK = ((W @ f.W.T) * f.weights(self.gamma)) @ f.whitener  # k x m
        HWt = np.asarray(self._H_op @ W.T)  # m x k
        return _times_pseudo_inverse(WK @ HWt, W)

    @cached_property
    def DOFS(self) -> float:
        r"""
        Degrees Of Freedom for Signal (DOFS).

        .. math::
            DOFS = Tr(A) = \sum_k \frac{\gamma \lambda_k}{\gamma \lambda_k + 1}

        From the eigenvalues :math:`\lambda_k` of the factors, without :attr:`A`.

        Returns
        -------
        float
            Degrees of Freedom value.
        """
        return self._whitened.dofs(self.gamma)

    @cached_property
    def reduced_chi2(self) -> float:
        r"""
        Reduced Chi-squared statistic. Tarantola (1987).

        .. math::
            \chi^2 = \frac{1}{n_z} \left((z - H\hat{x} - c)^T S_z^{-1} (z - H\hat{x} - c) + (\hat{x} - x_0)^T S_0^{-1} (\hat{x} - x_0)\right)

        with :math:`S_z` divided by gamma. The two misfits sum to
        :math:`d^T (H S_0 H^T + S_z)^{-1} d` with :math:`d = z - H x_0 - c`, the
        minimum of twice the cost, which the factors give as
        :math:`\sum_k w_k e_k^2`: no solve with :math:`S_0` or :math:`S_z`.

        Returns
        -------
        float
            Reduced Chi-squared value.
        """
        logger.debug("Calculating Reduced Chi-squared statistic...")
        return self._whitened.chi2(self.gamma) / self.n_z

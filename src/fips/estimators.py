"""Inversion estimators for solving inverse problems.

This module contains Bayesian and regularized estimators for state estimation
in linear inverse problems, computing posterior distributions and diagnostics.
"""

import logging
from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np
from scipy.linalg import inv, solve

logger = logging.getLogger(__name__)

# TODO
# - implement bayesian regularization factor usage


class EstimatorRegistry(dict):
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
            self[name] = cls
            return cls

        return decorator


ESTIMATOR_REGISTRY: dict[str, type["Estimator"]] = EstimatorRegistry()


class Estimator(ABC):
    """
    Base inversion estimator class.

    Attributes
    ----------
    z : np.ndarray
        Observed data.
    x_0 : np.ndarray
        Prior model state estimate.
    H : np.ndarray
        Forward operator.
    S_0 : np.ndarray
        Prior error covariance.
    S_z : np.ndarray
        Model-data mismatch covariance.
    c : np.ndarray or float, optional
        Constant data, defaults to 0.0.
    n_z : int
        Number of observations.
    n_x : int
        Number of state variables.
    x_hat : np.ndarray
        Posterior mean model state estimate (solution).
    S_hat : np.ndarray
        Posterior error covariance.
    y_hat : np.ndarray
        Posterior modeled observations.
    y_0 : np.ndarray
        Prior modeled observations.
    K : np.ndarray
        Kalman gain.
    A : np.ndarray
        Averaging kernel.
    chi2 : float
       Chi-squared statistic.
    R2 : float
       Coefficient of determination.
    RMSE : float
       Root mean square error.
    U_red : np.ndarray
       Reduced uncertainty.

    Methods
    -------
    cost(x: np.ndarray) -> float
        Cost/loss/misfit function.
    forward(x: np.ndarray) -> np.ndarray
        Forward model calculation.
    residual(x: np.ndarray) -> np.ndarray
        Forward model residual.
    leverage(x: np.ndarray) -> np.ndarray
        Calculate the leverage matrix.
    """

    _output_meta = {
        # attr : (row_space, col_space, is_covariance)
        "x_hat": ("state", None, False),
        "S_hat": ("state", "state", True),
        "y_hat": ("obs", None, False),
        "y_0": ("obs", None, False),
        "K": ("state", "obs", False),
        "A": ("state", "state", False),
        "U_red": ("state", None, False),
    }

    def __init__(
        self,
        z: np.ndarray,
        x_0: np.ndarray,
        H: np.ndarray,
        S_0: np.ndarray,
        S_z: np.ndarray,
        c: np.ndarray | float | None = None,
    ):
        """
        Initialize the Estimator object.

        Parameters
        ----------
        z : np.ndarray
            Observed data.
        x_0 : np.ndarray
            Prior model state estimate.
        H : np.ndarray
            Forward operator.
        S_0 : np.ndarray
            Prior error covariance.
        S_z : np.ndarray
            Model-data mismatch covariance.
        c : np.ndarray or float, optional
            Constant data, defaults to 0.0.
        """
        self.z = z.astype(float)
        self.x_0 = x_0.astype(float)
        self.H = H.astype(float)
        self.S_0 = S_0.astype(float)
        self.S_z = S_z.astype(float)

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
        return self.H @ x + self.c

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

        Parameters
        ----------
        x : np.ndarray
            State vector.

        Returns
        -------
        np.ndarray
            Leverage matrix.
        """
        logger.debug("Calculating Leverage matrix...")
        Hx = self.forward(x)
        Hx_T = Hx.T

        # Equation: L = Hx (Hx^T (H S_0 H^T + S_z)^-1 Hx)^-1 Hx^T (H S_0 H^T + S_z)^-1
        # Let A = (H S_0 H^T + S_z)
        A = self._HS_0H + self.S_z

        # 1. Compute term_1 = A^-1 @ Hx using solve
        term_1 = solve(A, Hx, assume_a="pos")

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
        """
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
        """
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
        A = self._HS_0H + self.S_z
        B = self._HS_0
        return solve(A, B, assume_a="pos").T

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
        return self.K @ self.H

    @cached_property
    def _H_T(self):
        """
        Transpose of the forward operator
        """
        return self.H.T

    @cached_property
    def _HS_0(self):
        """
        ... math::
            H S_0
        """
        return self.H @ self.S_0

    @cached_property
    def _HS_0H(self):
        """
        ... math::
            H S_0 H^T
        """
        return self._HS_0 @ self._H_T

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
        """
        Reduced Chi-squared statistic. Tarantola (1987)

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
        # Then we compute: r^T @ y

        # 1. Scaled Data Misfit: data_residual^T @ (S_z^-1 @ data_residual)
        scaled_data_misfit = data_residual.T @ solve(
            self.S_z, data_residual, assume_a="pos"
        )

        # 2. Scaled Model Misfit: model_residual^T @ (S_0^-1 @ model_residual)
        scaled_model_misfit = model_residual.T @ solve(
            self.S_0, model_residual, assume_a="pos"
        )

        return float((scaled_data_misfit + scaled_model_misfit) / self.n_z)

    @cached_property
    def R2(self) -> float:
        """
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
        """
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
        """
        Uncertainty reduction metric.

        .. math::
            U_{red} = 1 - \\frac{\\sqrt{trace(\\hat{S})}}{\\sqrt{trace(S_0)}}

        Returns
        -------
        float
            Uncertainty reduction value.
        """
        logger.debug("Calculating uncertainty reduction metric...")
        return float(1 - (np.sqrt(np.trace(self.S_hat)) / np.sqrt(np.trace(self.S_0))))

    @cached_property
    def U_red(self) -> np.ndarray:
        """
        Uncertainty reduction vector.

        .. math::
            U_{red} = \\left( 1 - \\frac{\\sqrt{diag(\\hat{S})}}{\\sqrt{diag(S_0)}} \\right) * 100\\%

        Returns
        -------
        np.ndarray
            Uncertainty reduction vector.
        """
        logger.debug("Calculating uncertainty reduction vector...")
        return (1 - (np.sqrt(np.diag(self.S_hat)) / np.sqrt(np.diag(self.S_0)))) * 100.0


@ESTIMATOR_REGISTRY.register("bayesian")
class BayesianSolver(Estimator):
    """
    Bayesian inversion estimator class
    This class implements a Bayesian inversion framework for solving inverse problems,
    also known as the batch method.
    """

    def __init__(
        self,
        z: np.ndarray,
        x_0: np.ndarray,
        H: np.ndarray,
        S_0: np.ndarray,
        S_z: np.ndarray,
        c: np.ndarray | float | None = None,
        rf: float = 1.0,
    ):
        """
        Initialize inversion object

        Parameters
        ----------
        z : np.ndarray
            Observed data
        x_0 : np.ndarray
            Prior model estimate
        H : np.ndarray
            Forward operator
        S_0 : np.ndarray
            Prior error covariance
        S_z : np.ndarray
            Model-data mismatch covariance
        c : np.ndarray | float, optional
            Constant data, defaults to 0.0
        rf : float, optional
            Regularization factor, by default 1.0
        """
        super().__init__(z=z, x_0=x_0, H=H, S_0=S_0, S_z=S_z, c=c)
        self.rf = rf  # TOOD implement usage of regularization factor

    def cost(self, x):
        """
        Cost function

        .. math::
            J(x) = \\frac{1}{2}(x - x_0)^T S_0^{-1}(x - x_0) + \\frac{1}{2}(z - Hx - c)^T S_z^{-1}(z - Hx - c)
        """
        logger.debug("Performing cost calculation...")
        diff_model = x - self.x_0
        diff_data = self.residual(x)

        # Like the Chi-square, the cost function relies on r^T * S^-1 * r.
        # We replace inv(S) @ r with solve(S, r) to avoid calculating the full inverse.

        cost_model = diff_model.T @ solve(self.S_0, diff_model, assume_a="pos")
        cost_data = diff_data.T @ solve(self.S_z, diff_data, assume_a="pos")

        return 0.5 * (cost_model + cost_data)

    @cached_property
    def x_hat(self):
        """
        Posterior Mean Model Estimate (solution)

        .. math::
            \\hat{x} = x_0 + K(z - Hx_0 - c)
        """
        logger.debug("Calculating Posterior Mean Model Estimate...")
        return self.x_0 + self.K @ self.residual(self.x_0)

    @cached_property
    def S_hat(self):
        """
        Posterior Error Covariance Matrix

        .. math::
            \\hat{S} = (H^T S_z^{-1} H + S_0^{-1})^{-1}
                = S_0 - (H S_0)^T(H S_0 H^T + S_z)^{-1}(H S_0)
        """
        logger.debug("Calculating Posterior Error Covariance Matrix...")
        # Mathematically, we want to subtract: B^T * A^-1 * B
        # where A = (H S_0 H^T + S_z) and B = (H S_0)
        #
        # Using scipy.linalg.solve(A, B), we compute the (A^-1 * B) term without
        # ever building the explicit inverse matrix.
        # The equation simplifies to: S_0 - B^T @ solve(A, B)
        A = self._HS_0H + self.S_z
        B = self._HS_0

        return self.S_0 - B.T @ solve(A, B, assume_a="pos")

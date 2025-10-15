"""
Inversion estimators.

This module contains various inversion estimators for solving inverse problems.
"""

from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np
from numpy.linalg import inv as invert

# TODO
# - implement bayesian regularization factor usage


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
        self.z = z
        self.x_0 = x_0
        self.H = H
        self.S_0 = S_0
        self.S_z = S_z
        self.c = c if c is not None else 0.0

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
        print("Performing forward calculation...")
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
        print("Performing residual calculation...")
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
        print("Calculating Leverage matrix...")
        Hx = self.forward(x)
        Hx_T = Hx.T
        HS_0H_Sz_inv = invert(self._HS_0H + self.S_z)
        return Hx @ invert(Hx_T @ HS_0H_Sz_inv @ Hx) @ Hx_T @ HS_0H_Sz_inv

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
        print("Performing cost calculation...")
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
        print("Calculating Posterior Mean Model State Estimate...")
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
        print("Calculating Posterior Error Covariance Matrix...")
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
        print("Calculating Posterior Mean Observation Estimate...")
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
        print("Calculating Prior Mean Data Estimate...")
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
        print("Calculating Kalman Gain Matrix...")
        return self._HS_0.T @ invert(self._HS_0H + self.S_z)

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
        print("Calculating Averaging Kernel Matrix...")
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
    def _S_0_inv(self):
        """
        Inverse of prior error covariance matrix
        """
        return invert(self.S_0)

    @cached_property
    def _S_z_inv(self):
        """
        Inverse of model-data mismatch covariance matrix
        """
        return invert(self.S_z)

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
        return np.trace(self.A)

    @cached_property
    def chi2(self) -> float:
        """
        Reduced Chi-squared statistic.

        .. math::
            \\chi^2 = \\frac{1}{n_z} ((z - H\\hat{x})^T S_z^{-1} (z - H\\hat{x}) + (\\hat{x} - x_0)^T S_0^{-1} (\\hat{x} - x_0))

        Returns
        -------
        float
            Reduced Chi-squared value.
        """
        # TBH im not 100% sure this is right
        return (self.chi2_obs + self.chi2_state) / self.n_z

    @cached_property
    def chi2_obs(self) -> float:
        """
        Chi-squared statistic for observation params

        .. math::
            \\chi^2 = (z - H\\hat{x})^T S_z^{-1} (z - H\\hat{x})

        Returns
        -------
        float
            Chi-squared value.
        """
        r = self.residual(self.x_hat)
        return (r.T @ self._S_z_inv @ r) / self.n_z

    @cached_property
    def chi2_state(self) -> float:
        """
        Chi-squared statistic for state params

        .. math::
            \\chi^2 = (\\hat{x} - x_0)^T S_0^{-1} (\\hat{x} - x_0)
        """
        r = self.x_hat - self.x_0
        return (r.T @ self._S_0_inv @ r) / self.n_x

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
        print("Calculating Coefficient of determination (R-squared)...")
        return np.corrcoef(self.z, self.y_hat)[0, 1] ** 2

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
        print("Calculating Root Mean Square Error (RMSE)...")
        r = self.residual(self.x_hat)
        return np.sqrt((r**2) / self.n_z)

    @cached_property
    def U_red(self):
        """
        Uncertainty reduction metric.

        .. math::
            U_{red} = 1 - \\frac{\\sqrt{trace(\\hat{S})}}{\\sqrt{trace(S_0)}}

        Returns
        -------
        float
            Uncertainty reduction value.
        """
        print("Calculating Uncertainty reduction metric...")
        return 1 - (np.sqrt(np.trace(self.S_hat)) / np.sqrt(np.trace(self.S_0)))


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


ESTIMATOR_REGISTRY = EstimatorRegistry()


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
        print("Performing cost calculation...")
        diff_model = x - self.x_0
        diff_data = self.residual(x)
        cost_model = diff_model.T @ self._S_0_inv @ diff_model
        cost_data = diff_data.T @ self._S_z_inv @ diff_data
        return 0.5 * (cost_model + cost_data)

    @cached_property
    def x_hat(self):
        """
        Posterior Mean Model Estimate (solution)

        .. math::
            \\hat{x} = x_0 + K(z - Hx_0 - c)
        """
        print("Calculating Posterior Mean Model Estimate...")
        return self.x_0 + self.K @ self.residual(self.x_0)

    @cached_property
    def S_hat(self):
        """
        Posterior Error Covariance Matrix

        .. math::
            \\hat{S} = (H^T S_z^{-1} H + S_0^{-1})^{-1}
                = S_0 - (H S_0)^T(H S_0 H^T + S_z)^{-1}(H S_0)
        """
        print("Calculating Posterior Error Covariance Matrix...")
        # Both methods return the same result
        # return invert(self.H_T @ self.S_z_inv @ self.H + self.S_0_inv)
        return (
            self.S_0 - self._HS_0.T @ invert(self._HS_0H + self.S_z) @ self._HS_0
        )  # this one only has one invert call

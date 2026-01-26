"""
Inversion estimators.
This module contains various inversion estimators for solving inverse problems.
"""

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Optional, Union, Callable

import numpy as np
from numpy.linalg import inv as invert
from scipy.optimize import minimize

class Estimator(ABC):
    """
    Base inversion estimator class.
    
    Supports both Matrix-based (Linear) and Function-based (Non-Linear) forward models.
    """

    def __init__(
        self,
        z: np.ndarray,
        x_0: np.ndarray,
        H: Optional[np.ndarray] = None,
        S_0: Optional[np.ndarray] = None,
        S_z: Optional[np.ndarray] = None,
        c: Optional[Union[np.ndarray, float]] = None,
        forward_model: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        bounds: Optional[list] = None
    ):
        self.z = z.astype(float)
        self.x_0 = x_0.astype(float)
        
        # Matrix operators (Optional if using functional forward model)
        self.H = H.astype(float) if H is not None else None
        self.S_0 = S_0.astype(float) if S_0 is not None else None
        self.S_z = S_z.astype(float) if S_z is not None else None
        
        # Custom forward model function f(x)
        self._custom_forward = forward_model
        
        # Bounds for constrained optimization (used in Variational/MCMC)
        self.bounds = bounds

        # Handle optional constant data
        if c is None:
            c = 0.0
        if isinstance(c, (int, float)):
            self.c = np.full(self.z.shape, float(c)).astype(float)
        elif isinstance(c, np.ndarray):
            self.c = c.astype(float)
        else:
            # Only strictly required if H is used
            self.c = 0.0

        self.n_z = z.shape[0]
        self.n_x = x_0.shape[0]

    def forward(self, x) -> np.ndarray:
        """
        Calculates y = F(x). 
        Uses Matrix H if available, otherwise uses custom function.
        """
        if self._custom_forward is not None:
            return self._custom_forward(x)
        elif self.H is not None:
            return self.H @ x + self.c
        else:
            raise ValueError("No Forward Operator (H or function) defined.")

    def residual(self, x) -> np.ndarray:
        return self.z - self.forward(x)

    @abstractmethod
    def cost(self, x) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def x_hat(self) -> np.ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def S_hat(self) -> np.ndarray:
        raise NotImplementedError

    # --- Common Cached Properties (Linear Assumption) ---
    # These properties (K, A) are strictly valid only for Linear problems.
    # For Non-Linear, these represent the linearized approximation at the prior.

    @cached_property
    def y_hat(self) -> np.ndarray:
        return self.forward(self.x_hat)

    @cached_property
    def K(self):
        """Kalman gain matrix (Linear Only)."""
        if self.H is None: raise NotImplementedError("K requires Matrix H")
        return self._HS_0.T @ invert(self._HS_0H + self.S_z)

    @cached_property
    def A(self):
        """Averaging kernel matrix (Linear Only)."""
        if self.H is None: raise NotImplementedError("A requires Matrix H")
        return self.K @ self.H

    @cached_property
    def _HS_0(self):
        return self.H @ self.S_0

    @cached_property
    def _HS_0H(self):
        return self._HS_0 @ self.H.T

    @cached_property
    def _S_0_inv(self):
        return invert(self.S_0)

    @cached_property
    def _S_z_inv(self):
        return invert(self.S_z)
    
    @cached_property
    def DOFS(self) -> float:
        return float(np.trace(self.A))


class EstimatorRegistry(dict):
    def register(self, name: str):
        def decorator(cls: type[Estimator]) -> type[Estimator]:
            self[name] = cls
            return cls
        return decorator

ESTIMATOR_REGISTRY = EstimatorRegistry()

@ESTIMATOR_REGISTRY.register("bayesian")
class BayesianSolver(Estimator):
    """
    Standard Analytical Solution (BLUE).
    Requires H, S_0, S_z to be explicit matrices.
    """
    def cost(self, x):
        diff_model = x - self.x_0
        diff_data = self.residual(x)
        cost_model = diff_model.T @ self._S_0_inv @ diff_model
        cost_data = diff_data.T @ self._S_z_inv @ diff_data
        return 0.5 * (cost_model + cost_data)

    @cached_property
    def x_hat(self):
        return self.x_0 + self.K @ self.residual(self.x_0)

    @cached_property
    def S_hat(self):
        return self.S_0 - self._HS_0.T @ invert(self._HS_0H + self.S_z) @ self._HS_0

@ESTIMATOR_REGISTRY.register("variational")
class VariationalSolver(Estimator):
    """
    Solves the problem using Numerical Optimization (BFGS/L-BFGS-B).
    This is effectively '3D-Var' or '4D-Var'.
    
    Advantages:
    1. Can handle bounds (e.g. Flux > 0).
    2. Does not require K matrix inversion (good for large N).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._optimization_result = None

    def cost(self, x):
        # Explicit implementation of J(x)
        diff_model = x - self.x_0
        diff_data = self.residual(x)
        
        # If matrices are available, use them. 
        # For huge problems, these @ operators would be replaced by linear operators.
        cost_model = diff_model.T @ self._S_0_inv @ diff_model
        cost_data = diff_data.T @ self._S_z_inv @ diff_data
        
        return 0.5 * (cost_model + cost_data)

    def _run_optimization(self):
        if self._optimization_result is not None:
            return

        print("Running Variational Optimization (L-BFGS-B)...")
        
        # We use L-BFGS-B which supports bounds
        res = minimize(
            fun=self.cost,
            x0=self.x_0,
            method='L-BFGS-B',
            bounds=self.bounds, # Passed from init
            options={'disp': True}
        )
        self._optimization_result = res

    @property
    def x_hat(self):
        self._run_optimization()
        return self._optimization_result.x

    @property
    def S_hat(self):
        """
        In variational methods, the posterior covariance is the 
        inverse Hessian of the cost function at the minimum.
        Approximating this is expensive for large problems.
        """
        # For this example, we fall back to the analytical calculation 
        # assuming linearity at the solution point.
        if self.H is not None:
            return self.S_0 - self._HS_0.T @ invert(self._HS_0H + self.S_z) @ self._HS_0
        else:
            raise NotImplementedError("S_hat calculation requires explicit Hessian or H matrix.")
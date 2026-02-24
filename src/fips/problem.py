"""Inverse problem framework for state estimation.

This module provides the core InverseProblem class which combines observations,
prior estimates, forward operators, and error covariances into a unified framework
for state estimation.
"""

import logging

import pandas as pd
from typing_extensions import Self

from fips.base import Pickleable
from fips.covariance import CovarianceMatrix
from fips.estimators import ESTIMATOR_REGISTRY, Estimator
from fips.matrix import Matrix, MatrixBlock, MatrixLike
from fips.operators import ForwardOperator
from fips.vector import Block, Vector, VectorLike

logger = logging.getLogger(__name__)


class InverseProblem(Pickleable):
    """Inverse problem combining observations, priors, and forward model.

    Organizes state vectors, observations, forward operators, and error covariances
    into a unified framework for solving inverse problems via different estimators.
    """

    def __init__(
        self,
        obs: VectorLike,
        prior: VectorLike,
        forward_operator: MatrixLike,
        modeldata_mismatch: MatrixLike,
        prior_error: MatrixLike,
        constant: "VectorLike | float | None" = None,
        round_index: int | None = 6,
    ):
        def promote_1d(data, default_block: str):
            if isinstance(data, pd.Series) and "block" not in data.index.names:
                # Wrap naked Series in a Block
                return Block(data, name=default_block)
            return data

        def promote_2d(data, default_row: str, default_col: str):
            if isinstance(data, pd.DataFrame):
                has_row = "block" in data.axes[0].names
                has_col = "block" in data.axes[1].names
                if not has_row or not has_col:
                    # Wrap naked DataFrame in a MatrixBlock
                    return MatrixBlock(
                        data, row_block=default_row, col_block=default_col
                    )
            return data

        def getname(obj, default):
            return getattr(obj, "name", default)

        obs = Vector(promote_1d(obs, "obs"), name=getname(obs, "obs"))
        prior = Vector(promote_1d(prior, "state"), name=getname(prior, "prior"))

        forward_operator = ForwardOperator(
            promote_2d(forward_operator, default_row="obs", default_col="state"),
            name=getname(forward_operator, "forward_operator"),
        )

        modeldata_mismatch = CovarianceMatrix(
            promote_2d(modeldata_mismatch, default_row="obs", default_col="obs"),
            name=getname(modeldata_mismatch, "modeldata_mismatch"),
        )

        prior_error = CovarianceMatrix(
            promote_2d(prior_error, default_row="state", default_col="state"),
            name=getname(prior_error, "prior_error"),
        )

        if constant is not None:
            constant_index = obs.index if isinstance(constant, (int, float)) else None
            constant = Vector(
                constant, name=getname(constant, "constant"), index=constant_index
            )

        # Round to specified precision if provided
        if round_index is not None:
            obs = obs.round_index(round_index)
            prior = prior.round_index(round_index)
            forward_operator = forward_operator.round_index(round_index, axis="both")
            modeldata_mismatch = modeldata_mismatch.round_index(
                round_index, axis="both"
            )
            prior_error = prior_error.round_index(round_index, axis="both")
            if constant is not None:
                constant = constant.round_index(round_index)

        # Reindex matrices to obs and prior (state) indices
        def reindex(matrix, row_idx, col_idx):
            return matrix.reindex(
                index=row_idx, columns=col_idx, fill_value=0.0, verify_overlap=True
            )

        forward_operator = reindex(
            forward_operator, row_idx=obs.index, col_idx=prior.index
        )
        modeldata_mismatch = reindex(
            modeldata_mismatch, row_idx=obs.index, col_idx=obs.index
        )
        prior_error = reindex(prior_error, row_idx=prior.index, col_idx=prior.index)
        if constant is not None:
            constant = constant.reindex(
                index=obs.index, fill_value=0.0, verify_overlap=True
            )

        self.obs = obs
        self.prior = prior
        self.forward_operator = forward_operator
        self.modeldata_mismatch = modeldata_mismatch
        self.prior_error = prior_error
        self.constant = constant
        self._estimator: Estimator | None = None  # init empty estimator

    @property
    def state_index(self) -> pd.Index:
        return self.prior.index

    @property
    def obs_index(self) -> pd.Index:
        return self.obs.index

    @property
    def n_state(self) -> int:
        return len(self.state_index)

    @property
    def n_obs(self) -> int:
        return len(self.obs_index)

    @property
    def estimator(self) -> Estimator:
        if self._estimator is None:
            raise RuntimeError("Problem has not been solved. Call .solve() first.")
        return self._estimator

    def get_block(
        self, component: str, block: str, crossblock: str | None = None
    ) -> pd.DataFrame | pd.Series:
        obj = getattr(self, component)

        if isinstance(obj, Vector):
            return obj[block]
        elif isinstance(obj, Matrix):
            if crossblock is None:
                crossblock = block
            return obj[block, crossblock]
        else:
            raise TypeError(f"Object '{component}' is neither a Vector nor a Matrix.")

    def solve(self, estimator: str | type[Estimator], **kwargs) -> Self:
        # Get estimator class
        if isinstance(estimator, str):
            if estimator not in ESTIMATOR_REGISTRY:
                raise ValueError(f"Estimator '{estimator}' is not registered.")
            estimator_cls = ESTIMATOR_REGISTRY[estimator]
        elif isinstance(estimator, type) and issubclass(estimator, Estimator):
            estimator_cls = estimator
        else:
            raise TypeError("Estimator must be a string or a subclass of Estimator.")

        logger.info(f"Solving using {estimator_cls.__name__}...")
        z = self.obs.values
        x_0 = self.prior.values
        H = self.forward_operator.values
        S_0 = self.prior_error.values
        S_z = self.modeldata_mismatch.values
        c = getattr(self.constant, "values", None)

        self._estimator = estimator_cls(
            z=z, x_0=x_0, H=H, S_0=S_0, S_z=S_z, c=c, **kwargs
        )

        return self

    def _wrap(self, math_attr: str, friendly_name: str):
        """Fetches raw math from the estimator and wraps it using the Estimator's space manifest."""
        # Get the raw numpy data
        raw_data = getattr(self.estimator, math_attr)

        # Look up the metadata from the Estimator's manifest
        meta = self.estimator._output_meta.get(math_attr)

        # If it's not in the manifest (e.g., chi2, DOFS), just return the raw float/array
        if not meta:
            return raw_data

        row_space, col_space, is_covariance = meta

        # Wrap 1D Vectors
        if col_space is None:
            index = self.state_index if row_space == "state" else self.obs_index
            return Vector(raw_data, index=index, name=friendly_name)

        # Wrap 2D Matrices
        idx_0 = self.state_index if row_space == "state" else self.obs_index
        idx_1 = self.state_index if col_space == "state" else self.obs_index

        if is_covariance:
            return CovarianceMatrix(
                raw_data, index=idx_0, columns=idx_1, name=friendly_name
            )
        return Matrix(raw_data, index=idx_0, columns=idx_1, name=friendly_name)

    def __getstate__(self):
        """Explicit pickle support: return state as dict."""
        return {
            "prior": self.prior,
            "obs": self.obs,
            "forward_operator": self.forward_operator,
            "prior_error": self.prior_error,
            "modeldata_mismatch": self.modeldata_mismatch,
            "constant": self.constant,
            "_estimator": self._estimator,
        }

    def __setstate__(self, state):
        """Explicit pickle support: restore state from dict."""
        self.prior = state["prior"]
        self.obs = state["obs"]
        self.forward_operator = state["forward_operator"]
        self.prior_error = state["prior_error"]
        self.modeldata_mismatch = state["modeldata_mismatch"]
        self.constant = state["constant"]
        self._estimator = state.get("_estimator", None)

    @property
    def posterior(self) -> Vector:
        """Posterior state estimate."""
        return self._wrap("x_hat", "posterior")

    @property
    def posterior_error(self) -> CovarianceMatrix:
        """Posterior error covariance."""
        return self._wrap("S_hat", "posterior_error")

    @property
    def posterior_obs(self) -> Vector:
        """Modeled observations (H @ posterior)."""
        return self._wrap("y_hat", "modeled_obs")

    @property
    def prior_obs(self) -> Vector:
        """Modeled observations using the prior (H @ prior)."""
        return self._wrap("y_0", "prior_obs")

    @property
    def kalman_gain(self) -> Matrix:
        """Kalman gain matrix (K)."""
        return self._wrap("K", "kalman_gain")

    @property
    def averaging_kernel(self) -> Matrix:
        """Averaging kernel matrix (A)."""
        return self._wrap("A", "averaging_kernel")

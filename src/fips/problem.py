"""
Inverse problem framework for state estimation.

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
from fips.vector import Vector, VectorLike

logger = logging.getLogger(__name__)


class InverseProblem(Pickleable):
    """
    Inverse problem combining observations, priors, and forward model.

    Organizes state vectors, observations, forward operators, and error covariances
    into a unified framework for solving inverse problems via different estimators.

    Parameters
    ----------
    obs : VectorLike
        Observation vector.
    prior : VectorLike
        Prior state vector.
    forward_operator : MatrixLike
        Forward operator mapping state space to observation space.
    modeldata_mismatch : MatrixLike
        Covariance matrix representing model-data mismatch (observation error).
    prior_error : MatrixLike
        Covariance matrix representing prior error.
    constant : VectorLike or float, optional
        Optional constant term added to the forward model (e.g., background or bias).
    round_index : int, optional
        Number of decimal places to round to. If None, no rounding is performed.
    """

    obs: Vector
    """Observation vector."""
    prior: Vector
    """Prior state vector."""
    forward_operator: ForwardOperator
    """Forward operator mapping state space to observation space."""
    prior_error: CovarianceMatrix
    """Covariance matrix representing prior error."""
    modeldata_mismatch: CovarianceMatrix
    """Covariance matrix representing model-data mismatch (observation error)."""
    constant: Vector | float | None
    """Optional constant term added to the forward model (e.g., background or bias)."""

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
        obs = Vector(obs)
        prior = Vector(prior)

        def promote_2d(
            data, target_index: pd.MultiIndex, target_columns: pd.MultiIndex
        ):
            """Promote 2D DataFrame to MatrixBlock, using block levels from target index/columns if needed."""
            if isinstance(data, pd.DataFrame):
                has_row = "block" in data.axes[0].names
                has_col = "block" in data.axes[1].names

                if not has_row:
                    row_blks = target_index.get_level_values("block").unique()
                    if len(row_blks) > 1:
                        raise ValueError(
                            "Data has no row 'block' level but target index has multiple blocks."
                        )
                    row_block = str(row_blks[0]) if len(row_blks) == 1 else None
                else:
                    row_block = None

                if not has_col:
                    col_blks = target_columns.get_level_values("block").unique()
                    if len(col_blks) > 1:
                        raise ValueError(
                            "Data has no col 'block' level but target columns has multiple blocks."
                        )
                    col_block = str(col_blks[0]) if len(col_blks) == 1 else None
                else:
                    col_block = None

                if not has_row or not has_col:
                    # Wrap naked DataFrame in a MatrixBlock
                    return MatrixBlock(data, row_block=row_block, col_block=col_block)
            return data

        forward_operator = ForwardOperator(
            promote_2d(
                forward_operator, target_index=obs.index, target_columns=prior.index
            )
        )
        prior_error = CovarianceMatrix(
            promote_2d(
                prior_error, target_index=prior.index, target_columns=prior.index
            )
        )
        modeldata_mismatch = CovarianceMatrix(
            promote_2d(
                modeldata_mismatch, target_index=obs.index, target_columns=obs.index
            )
        )

        if constant is not None:
            constant_index = obs.index if isinstance(constant, (int, float)) else None
            constant = Vector(constant, index=constant_index)

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
        """Return the state space index."""
        return self.prior.index

    @property
    def obs_index(self) -> pd.Index:
        """Return the observation space index."""
        return self.obs.index

    @property
    def n_state(self) -> int:
        """Return number of state variables."""
        return len(self.state_index)

    @property
    def n_obs(self) -> int:
        """Return number of observations."""
        return len(self.obs_index)

    @property
    def estimator(self) -> Estimator:
        """Return the fitted estimator (raises if not solved)."""
        if self._estimator is None:
            raise RuntimeError("Problem has not been solved. Call .solve() first.")
        return self._estimator

    def get_block(
        self, component: str, block: str, crossblock: str | None = None
    ) -> pd.DataFrame | pd.Series:
        """
        Get block from a component (Vector or Matrix).

        Parameters
        ----------
        component : str
            Name of the component ('obs', 'prior', 'forward_operator', 'modeldata_mismatch', 'prior_error', or 'constant').
        block : str
            Name of the block to retrieve.
        crossblock : str, optional
            For matrices, the name of the cross block (e.g., 'state' for forward_operator). If None, defaults to the same as 'block'.

        Returns
        -------
        pd.Series or pd.DataFrame
            The requested block of data.
        """
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
        """
        Solve the inverse problem using the specified estimator.

        Parameters
        ----------
        estimator : str or type[Estimator]
            Estimator to use for solving the inverse problem. Can be a string key for registered estimators or a subclass of Estimator.
        **kwargs
            Additional keyword arguments to pass to the estimator.

        Returns
        -------
        Self
            The InverseProblem instance with the estimator fitted.
        """
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

        # Trigger computation of posterior
        _ = self.posterior

        return self

    def _wrap(self, math_attr: str, friendly_name: str):
        """Fetch raw math from the estimator and wrap it using the Estimator's space manifest."""
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

    def __repr__(self) -> str:
        """Return string representation."""
        solved = self._estimator is not None
        return (
            f"{self.__class__.__name__}("
            f"n_obs={self.n_obs}, n_state={self.n_state}, solved={solved})"
        )

    @property
    def posterior(self) -> Vector:
        """Posterior state estimate."""
        return self._wrap("x_hat", "posterior")  # type: ignore[return-value]

    @property
    def posterior_error(self) -> CovarianceMatrix:
        """Posterior error covariance."""
        return self._wrap("S_hat", "posterior_error")  # type: ignore[return-value]

    @property
    def posterior_obs(self) -> Vector:
        """Modeled observations (H @ posterior)."""
        return self._wrap("y_hat", "modeled_obs")  # type: ignore[return-value]

    @property
    def prior_obs(self) -> Vector:
        """Modeled observations using the prior (H @ prior)."""
        return self._wrap("y_0", "prior_obs")  # type: ignore[return-value]

    @property
    def kalman_gain(self) -> Matrix:
        """Kalman gain matrix (K)."""
        return self._wrap("K", "kalman_gain")  # type: ignore[return-value]

    @property
    def averaging_kernel(self) -> Matrix:
        """Averaging kernel matrix (A)."""
        return self._wrap("A", "averaging_kernel")  # type: ignore[return-value]

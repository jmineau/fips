from pathlib import Path

import pandas as pd

from fips.estimators import ESTIMATOR_REGISTRY, Estimator
from fips.interfaces import PD, XR, EstimatorOutput
from fips.matrices import CovarianceMatrix, ForwardOperator, Matrix, prepare_matrix
from fips.serialization import Pickleable
from fips.vectors import Vector, prepare_vector


class InverseProblem(EstimatorOutput, Pickleable):
    def __init__(
        self,
        prior: str | Path | Vector | pd.Series,
        obs: str | Path | Vector | pd.Series,
        forward_operator: str | Path | ForwardOperator | pd.DataFrame,
        prior_error: str | Path | CovarianceMatrix | pd.DataFrame,
        modeldata_mismatch: str | Path | CovarianceMatrix | pd.DataFrame,
        constant: str | Path | Vector | pd.Series | float | None = None,
        float_precision: int | None = None,
    ):
        super().__init__()

        # Prepare obs and prior vectors (handles file paths via load_or_pass)
        self.obs = prepare_vector(
            name="obs", vector=obs, float_precision=float_precision
        )
        self.prior = prepare_vector(
            name="prior", vector=prior, float_precision=float_precision
        )

        # Prepare forward operator and covariance matrices (handles file paths via load_or_pass)
        self.forward_operator = prepare_matrix(
            matrix=forward_operator,
            matrix_class=ForwardOperator,
            row_index=self.obs_index,
            col_index=self.state_index,
            float_precision=float_precision,
        )

        self.prior_error = prepare_matrix(
            matrix=prior_error,
            matrix_class=CovarianceMatrix,
            row_index=self.state_index,
            col_index=self.state_index,
            float_precision=float_precision,
        )

        self.modeldata_mismatch = prepare_matrix(
            matrix=modeldata_mismatch,
            matrix_class=CovarianceMatrix,
            row_index=self.obs_index,
            col_index=self.obs_index,
            float_precision=float_precision,
        )

        # Prepare constant (scalar or vector aligned to obs index, handles file paths)
        if constant is not None:
            try:
                constant = float(constant)
            except (TypeError, ValueError):
                constant = prepare_vector(
                    name="constant", vector=constant, float_precision=float_precision
                )
        self.constant = constant

        self.float_precision = float_precision
        self._estimator: Estimator | None = None  # init empty estimator

    def __getstate__(self):
        """Explicit pickle support: return state as dict."""
        return {
            "prior": self.prior,
            "obs": self.obs,
            "forward_operator": self.forward_operator,
            "prior_error": self.prior_error,
            "modeldata_mismatch": self.modeldata_mismatch,
            "constant": self.constant,
            "float_precision": self.float_precision,
            "_estimator": self._estimator,
            "_output_cache": self._output_cache,
        }

    def __setstate__(self, state):
        """Explicit pickle support: restore state from dict."""
        self.prior = state["prior"]
        self.obs = state["obs"]
        self.forward_operator = state["forward_operator"]
        self.prior_error = state["prior_error"]
        self.modeldata_mismatch = state["modeldata_mismatch"]
        self.constant = state["constant"]
        self.float_precision = state["float_precision"]
        self._estimator = state.get("_estimator", None)
        self._output_cache = state.get("_output_cache", {})

    @property
    def state_index(self) -> pd.Index:
        return self.prior.index

    @property
    def obs_index(self) -> pd.Index:
        return self.obs.index

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
            return obj.data.loc[block, crossblock]
        else:
            raise TypeError(f"Object '{component}' is neither a Vector nor a Matrix.")

    def solve(
        self, estimator: str | type[Estimator], **kwargs
    ) -> dict[str, Vector | CovarianceMatrix]:
        # Get estimator class
        if isinstance(estimator, str):
            if estimator not in ESTIMATOR_REGISTRY:
                raise ValueError(f"Estimator '{estimator}' is not registered.")
            estimator_cls = ESTIMATOR_REGISTRY[estimator]
        elif isinstance(estimator, type) and issubclass(estimator, Estimator):
            estimator_cls = estimator
        else:
            raise TypeError("Estimator must be a string or a subclass of Estimator.")

        print(f"Solving using {estimator_cls.__name__}...")
        z = self.obs.values
        x_0 = self.prior.values
        H = self.forward_operator.values
        S_0 = self.prior_error.values
        S_z = self.modeldata_mismatch.values
        c = (
            self.constant
            if not isinstance(self.constant, Vector)
            else self.constant.values
        )

        self._estimator = estimator_cls(
            z=z, x_0=x_0, H=H, S_0=S_0, S_z=S_z, c=c, **kwargs
        )

        return {
            "posterior": self.posterior,
            "posterior_error": self.posterior_error,
            "posterior_obs": self.posterior_obs,
        }

    @property
    def pd(self) -> PD:
        return PD(self)

    @property
    def xr(self) -> XR:
        return XR(self)

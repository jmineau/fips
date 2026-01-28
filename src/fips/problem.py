import pandas as pd

from fips.estimators import ESTIMATOR_REGISTRY, Estimator
from fips.interfaces import PD, XR, EstimatorOutput
from fips.matrices import CovarianceMatrix, ForwardOperator, Matrix, prepare_matrix
from fips.vectors import Vector, prepare_vector


class InverseProblem(EstimatorOutput):
    def __init__(
        self,
        prior: Vector,
        obs: Vector,
        forward_operator: ForwardOperator,
        prior_error: CovarianceMatrix,
        modeldata_mismatch: CovarianceMatrix,
        float_precision: int | None = None,
    ):
        super().__init__()

        # Prepare obs and prior vectors
        self.obs = prepare_vector(
            name="obs", vector=obs, float_precision=float_precision
        )
        self.prior = prepare_vector(
            name="prior", vector=prior, float_precision=float_precision
        )

        # Prepare forward operator and covariance matrices
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

        self.float_precision = float_precision
        self._estimator: Estimator | None = None  # init empty estimator

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
        self, estimator: str | type[Estimator] = "bayesian", **kwargs
    ) -> dict[str, pd.Series | CovarianceMatrix | pd.Series]:
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
        self._estimator = estimator_cls(z=z, x_0=x_0, H=H, S_0=S_0, S_z=S_z, **kwargs)

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

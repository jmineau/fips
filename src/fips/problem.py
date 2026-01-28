from functools import cached_property

import pandas as pd

from fips.estimators import ESTIMATOR_REGISTRY, Estimator
from fips.interfaces import XR
from fips.matrices import CovarianceMatrix, ForwardOperator, Matrix, prepare_matrix
from fips.vectors import Vector, prepare_vector


class InverseProblem:
    def __init__(
        self,
        prior: Vector | pd.Series,
        obs: Vector | pd.Series,
        forward_operator: pd.DataFrame | ForwardOperator,
        prior_error: pd.DataFrame | CovarianceMatrix,
        modeldata_mismatch: pd.DataFrame | CovarianceMatrix,
        float_precision: int | None = None,
    ):
        # Store vectors and matrices
        self._vectors = {}
        self._matrices = {}

        # Prepare obs and prior vectors
        self._vectors["obs"] = prepare_vector(
            name="obs", vector=obs, float_precision=float_precision
        )
        self._vectors["prior"] = prepare_vector(
            name="prior", vector=prior, float_precision=float_precision
        )

        # Prepare forward operator and covariance matrices
        self._matrices["forward_operator"] = prepare_matrix(
            matrix=forward_operator,
            matrix_class=ForwardOperator,
            row_index=self.obs_index,
            col_index=self.state_index,
            float_precision=float_precision,
        )

        self._matrices["prior_error"] = prepare_matrix(
            matrix=prior_error,
            matrix_class=CovarianceMatrix,
            row_index=self.state_index,
            col_index=self.state_index,
            float_precision=float_precision,
        )

        self._matrices["modeldata_mismatch"] = prepare_matrix(
            matrix=modeldata_mismatch,
            matrix_class=CovarianceMatrix,
            row_index=self.obs_index,
            col_index=self.obs_index,
            float_precision=float_precision,
        )

        self.float_precision = float_precision
        self._estimator: Estimator | None = None  # init empty estimator

    @property
    def obs(self) -> pd.Series:
        return self._vectors["obs"].data

    @property
    def prior(self) -> pd.Series:
        return self._vectors["prior"].data

    @property
    def forward_operator(self) -> pd.DataFrame:
        return self._matrices["forward_operator"].data

    @property
    def modeldata_mismatch(self) -> pd.DataFrame:
        return self._matrices["modeldata_mismatch"].data

    @property
    def prior_error(self) -> pd.DataFrame:
        return self._matrices["prior_error"].data

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

    def get_vector(self, component) -> Vector:
        if self._vectors.get(component) is None and getattr(self, component) is None:
            raise KeyError(f"Vector '{component}' not found in problem.")

        return self._vectors[component]

    def get_matrix(self, component) -> Matrix:
        if self._matrices.get(component) is None and getattr(self, component) is None:
            raise KeyError(f"Matrix '{component}' not found in problem.")

        return self._matrices[component]

    def get_block(
        self, component: str, block: str, crossblock: str | None = None
    ) -> pd.DataFrame | pd.Series:
        try:
            obj = self.get_vector(component)
        except KeyError:
            try:
                obj = self.get_matrix(component)
            except KeyError:
                raise KeyError(
                    f"Component '{component}' not found in problem."
                ) from None

        if isinstance(obj, Vector):
            return obj[block]
        elif isinstance(obj, Matrix):
            if crossblock is None:
                crossblock = block
            return obj.data.loc[block, crossblock]
        else:
            raise TypeError(f"Object '{component}' is neither a Vector nor a Matrix.")

    @property
    def xr(self) -> XR:
        return XR(self)

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
        z = self.obs.to_numpy()
        x_0 = self.prior.to_numpy()
        H = self.forward_operator.to_numpy()
        S_0 = self.prior_error.to_numpy()
        S_z = self.modeldata_mismatch.to_numpy()
        self._estimator = estimator_cls(z=z, x_0=x_0, H=H, S_0=S_0, S_z=S_z, **kwargs)

        return {
            "posterior": self.posterior,
            "posterior_error": self.posterior_error,
            "posterior_obs": self.posterior_obs,
        }

    @cached_property
    def posterior(self) -> pd.Series:
        """
        Posterior state estimate.

        Returns
        -------
        pd.Series
            Pandas series with the posterior mean model estimate.
        """
        if self._vectors.get("posterior") is None:
            self._vectors["posterior"] = Vector.from_series(
                pd.Series(
                    self.estimator.x_hat, index=self.state_index, name="posterior"
                )
            )

        return self._vectors["posterior"].data

    @cached_property
    def posterior_obs(self) -> pd.Series:
        """
        Posterior observation estimates.

        Returns
        -------
        pd.Series
            Pandas series with the posterior observation estimates.
        """
        if self._vectors.get("posterior_obs") is None:
            self._vectors["posterior_obs"] = Vector.from_series(
                pd.Series(
                    self.estimator.y_hat, index=self.obs_index, name="posterior_obs"
                )
            )

        return self._vectors["posterior_obs"].data

    @cached_property
    def posterior_error(self) -> pd.DataFrame:
        """
        Posterior error covariance matrix.

        Returns
        -------
        pd.DataFrame
            CovarianceMatrix instance with the posterior error covariance matrix.
        """
        if self._matrices.get("posterior_error") is None:
            self._matrices["posterior_error"] = CovarianceMatrix(
                pd.DataFrame(
                    self.estimator.S_hat,
                    index=self.state_index,
                    columns=self.state_index,
                )
            )

        return self._matrices["posterior_error"].data

    @cached_property
    def prior_obs(self) -> pd.Series:
        """
        Prior observation estimates.

        Returns
        -------
        pd.Series
            Pandas series with the prior observation estimates.
        """
        if self._vectors.get("prior_obs") is None:
            self._vectors["prior_obs"] = Vector.from_series(
                pd.Series(self.estimator.y_0, index=self.obs_index, name="prior_obs")
            )

        return self._vectors["prior_obs"].data

    @cached_property
    def U_red(self) -> pd.Series:
        U_red = self.estimator.U_red
        return pd.Series(U_red, index=self.state_index, name="U_red")

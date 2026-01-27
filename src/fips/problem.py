from typing import Hashable, Optional, Type

import numpy as np
import pandas as pd

from fips.estimators import Estimator, ESTIMATOR_REGISTRY
from fips.indices import check_overlap
from fips.interfaces import EstimatorOutput, XR
from fips.vectors import Vector, prepare_vector
from fips.matrices import (
    ForwardOperator,
    CovarianceMatrix,
    Matrix,
    prepare_matrix
)


class InverseProblem(EstimatorOutput):
    def __init__(
        self,
        prior: Vector | pd.Series,
        obs: Vector | pd.Series,
        forward_operator: pd.DataFrame | ForwardOperator,
        prior_error: pd.DataFrame | CovarianceMatrix,
        modeldata_mismatch: pd.DataFrame | CovarianceMatrix,
        float_precision: int | None = None
    ):
        # Store vectors and matrices
        self.vectors = {}
        self.matrices = {}

        prior, promote_prior = prepare_vector(vector=prior, default_name="prior", float_precision=float_precision)
        obs, promote_obs = prepare_vector(vector=obs, default_name="obs", float_precision=float_precision)

        self.vectors['prior'] = prior
        self.vectors['obs'] = obs

        # Unwrap Matrices
        def unwrap_matrix(mat):
            if isinstance(mat, Matrix):
                return mat.data
            else:
                return mat

        forward_operator = unwrap_matrix(forward_operator)
        prior_error = unwrap_matrix(prior_error)
        modeldata_mismatch = unwrap_matrix(modeldata_mismatch)

        # Prepare matrices (sanitize & align names)
        forward_operator = prepare_matrix(matrix=forward_operator,
                                          row_promote=promote_obs, row_asm=obs,
                                          col_promote=promote_prior, col_asm=prior,
                                          float_precision=float_precision)
        prior_error = prepare_matrix(matrix=prior_error,
                                     row_promote=promote_prior, row_asm=prior,
                                     col_promote=promote_prior, col_asm=prior,
                                     float_precision=float_precision)
        modeldata_mismatch = prepare_matrix(matrix=modeldata_mismatch,
                                            row_promote=promote_obs, row_asm=obs,
                                            col_promote=promote_obs, col_asm=obs,
                                            float_precision=float_precision)

        idx_prior = prior.data.index
        idx_obs = obs.data.index

        # Check Overlap before strict reindexing
        check_overlap(idx_obs, forward_operator.index, "Observation")
        check_overlap(idx_prior, forward_operator.columns, "Prior/State")

        # Reindex matrices to ensure full coverage, filling missing with zeros
        # This is a left join with respect to the assembly vectors
        H_final = forward_operator.reindex(index=idx_obs, columns=idx_prior).fillna(0.0)
        prior_error = prior_error.reindex(index=idx_prior, columns=idx_prior).fillna(0.0)
        modeldata_mismatch = modeldata_mismatch.reindex(index=idx_obs, columns=idx_obs).fillna(0.0)

        self.matrices['prior_error'] = CovarianceMatrix(prior_error)
        self.matrices['modeldata_mismatch'] = CovarianceMatrix(modeldata_mismatch)
        self.matrices['forward_operator'] = ForwardOperator(H_final)
        self.float_precision = float_precision
        self._estimator: Optional[Estimator] = None

    @property
    def obs(self) -> pd.Series:
        return self.vectors['obs'].data

    @property
    def prior(self) -> pd.Series:
        return self.vectors['prior'].data

    @property
    def forward_operator(self) -> pd.DataFrame:
        return self.matrices['forward_operator'].data

    @property
    def modeldata_mismatch(self) -> pd.DataFrame:
        return self.matrices['modeldata_mismatch'].data

    @property
    def prior_error(self) -> pd.DataFrame:
        return self.matrices['prior_error'].data

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

    def aggregate_obs(self, func: str = "mean", **kwargs) -> "InverseProblem":
        """Create a new InverseProblem with aggregated observations (super-obbing)."""
        obs_idx = self.obs_index
        
        if 'rule' in kwargs:
            time_level = kwargs.get('level', None)
            if time_level is None:
                if isinstance(obs_idx, pd.MultiIndex):
                    for i, level in enumerate(obs_idx.levels):
                        if pd.api.types.is_datetime64_any_dtype(level):
                            time_level = obs_idx.names[i]
                            break
                    if time_level is None and len(obs_idx.levels) > 1:
                        time_level = 1
                else:
                    time_level = None

            resample_kwargs = kwargs.copy()
            resample_kwargs.pop('level', None)
            grouper = [pd.Grouper(level=0), pd.Grouper(level=time_level, **resample_kwargs)]
            grouped = self.obs.groupby(grouper)
        else:
            grouped = self.obs.groupby(**kwargs)

        group_mapper = grouped.ngroup() 
        n_new = grouped.ngroups
        n_old = len(self.obs)
        
        W = np.zeros((n_new, n_old))
        W[group_mapper.values, np.arange(n_old)] = 1.0
        
        if func == "mean":
            group_sizes = grouped.size().values 
            group_sizes[group_sizes == 0] = 1
            W = W / group_sizes[:, None]
        elif func != "sum":
            raise ValueError(f"Unsupported aggregation function '{func}'. Use 'mean' or 'sum'.")

        y_old = self.obs.full_vector.values
        y_new_vals = W @ y_old

        if func == "mean":
            y_new_series = grouped.mean()
        else:
            y_new_series = grouped.sum()

        H_old = self.forward_operator.values
        H_new_vals = W @ H_old

        S_old = self.modeldata_mismatch.values
        S_new_vals = W @ S_old @ W.T

        H_new = pd.DataFrame(H_new_vals, index=y_new_series.index, columns=self.state_index)
        S_new = pd.DataFrame(S_new_vals, index=y_new_series.index, columns=y_new_series.index)

        return InverseProblem(
            prior=self.prior, 
            obs=y_new_series, 
            forward_operator=H_new,
            prior_error=self.prior_error,
            modeldata_mismatch=S_new,
            float_precision=self.float_precision
        )

    def get_block(self, name: str, block: Hashable,
                  crossblock: Hashable | None = None) -> pd.DataFrame | pd.Series:
        objs = {
            **self.vectors,
            **self.matrices
        }
        if name not in objs:
            raise KeyError(f"Object '{name}' not found in problem.")

        obj = objs[name]
        if isinstance(obj, Vector):
            return obj[block]
        elif isinstance(obj, Matrix):
            if crossblock is None:
                crossblock = block
            return obj.data.loc[block, crossblock]
        else:
            raise TypeError(f"Object '{name}' is neither a Vector nor a Matrix.")

    @property
    def xr(self) -> XR:
        return XR(self)
    
    @property
    def state_blocks(self) -> list[str | Hashable]:
        return self.prior.names

    @property
    def obs_blocks(self) -> list[str | Hashable]:
        return self.obs.names

    def solve(self, estimator: str | Type[Estimator] = 'bayesian', **kwargs
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

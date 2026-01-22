"""
Core inversion problem class.
"""

from functools import partial

import pandas as pd

from fips.estimators import ESTIMATOR_REGISTRY, Estimator
from fips.interfaces import XR, EstimatorOutput
from fips.matrices import CovarianceMatrix
from fips.operators import ForwardOperator
from fips.utils import round_index, validate_single_column_df

# TODO
# - Obs aggregation
# - file io


class InverseProblem(EstimatorOutput):
    """
    Inverse problem class for estimating model states from observations.

    Represents a statistical inverse problem for estimating model states from observed data
    using Bayesian inference and linear forward operators.

    An inverse problem seeks to infer unknown model parameters (the "state") from observed data,
    given prior knowledge and a mathematical relationship (the forward operator) that links the state
    to the observations. This class provides a flexible interface for formulating and solving such
    problems using various estimators.

    Parameters
    ----------
    estimator : str or type[Estimator]
        The estimator to use for solving the inverse problem. Can be the name of a registered estimator
        or an Estimator class.
    obs : pd.Series
        Observed data as a pandas Series, indexed by observation dimensions.
    prior : pd.Series
        Prior estimate of the model state as a pandas Series, indexed by state dimensions.
    forward_operator : ForwardOperator or pd.DataFrame
        Linear operator mapping model state to observations. Can be a ForwardOperator instance or a
        pandas DataFrame with appropriate indices and columns.
    prior_error : CovarianceMatrix
        Covariance matrix representing uncertainty in the prior state estimate.
    modeldata_mismatch : CovarianceMatrix
        Covariance matrix representing uncertainty in the observed data (model-data mismatch).
    constant : float or pd.Series or None, optional
        Optional constant term added to the forward model output. If not provided, defaults to zero.
    state_index : pd.Index or None, optional
        Index for the state variables. If None, uses the index from the prior.
    estimator_kwargs : dict, optional
        Additional keyword arguments to pass to the estimator.
    coord_decimals : int, optional
        Number of decimal places to round coordinate values for alignment (default is 6).

    Raises
    ------
    TypeError
        If input types are incorrect.
    ValueError
        If input data dimensions are incompatible or indices do not align.

    Attributes
    ----------
    obs_index : pd.Index
        Index of the observations used in the problem.
    state_index : pd.Index
        Index of the state variables used in the problem.
    obs_dims : tuple
        Names of the observation dimensions.
    state_dims : tuple
        Names of the state dimensions.
    n_obs : int
        Number of observations.
    n_state : int
        Number of state variables.
    posterior : pd.Series
        Posterior mean estimate of the model state.
    posterior_error : CovarianceMatrix
        Posterior error covariance matrix.
    posterior_obs : pd.Series
        Posterior mean estimate of the observations.
    prior_obs : pd.Series
        Prior mean estimate of the observations.
    xr : InverseProblem._XR
        Xarray interface for accessing inversion results as xarray DataArrays.

    Methods
    -------
    solve() -> dict[str, pd.Series | CovarianceMatrix | pd.Series]
        Solves the inverse problem and returns a dictionary with posterior state, posterior error
        covariance, and posterior observation estimates.

    Notes
    -----
    This class is designed for linear inverse problems with Gaussian error models, commonly encountered
    in geosciences, remote sensing, and other fields where model parameters are inferred from indirect
    measurements. It supports flexible input formats and provides robust alignment and validation of
    input data.
    """

    def __init__(
        self,
        estimator: str | type[Estimator],
        obs: pd.Series,
        prior: pd.Series,
        forward_operator: ForwardOperator | pd.DataFrame,
        prior_error: CovarianceMatrix,
        modeldata_mismatch: CovarianceMatrix,
        constant: float | pd.Series | None = None,
        state_index: pd.Index | None = None,
        estimator_kwargs: dict | None = None,
        coord_decimals: int = 6,
    ) -> None:
        """
        Initialize the InverseProblem.

        Parameters
        ----------
        estimator : str or type[Estimator]
            Estimator class or its registered name as a string.
        obs : pd.Series
            Observed data.
        prior : pd.Series
            Prior model state estimate.
        forward_operator : pd.DataFrame
            Forward operator matrix.
        prior_error : CovarianceMatrix
            Prior error covariance matrix.
        modeldata_mismatch : CovarianceMatrix
            Model-data mismatch covariance matrix.
        constant : float or pd.Series, optional
            Constant data, defaults to 0.0.
        state_index : pd.Index, optional
            Index for the state variables.
        estimator_kwargs : dict, optional
            Additional keyword arguments for the estimator.
        obs_aggregation : optional
            Observation aggregation method.
        coord_decimals : int, optional
            Number of decimal places for rounding coordinates.

        Raises
        ------
        TypeError
            If any of the inputs are of the wrong type.
        ValueError
            If there are issues with the input data (e.g., incompatible dimensions).
        """
        self._initialized = False  # prevent premature alignment calls
        self._coord_decimals = coord_decimals

        # Set problem dimensions
        self.obs_dims = tuple(obs.index.names)
        self.state_dims = tuple(prior.index.names)

        # Set state index argument
        if state_index is not None:
            if not all(dim in state_index.names for dim in self.state_dims):
                raise ValueError("State index must contain all state dimensions.")
            self._state_index_arg = round_index(
                state_index, decimals=self._coord_decimals
            )
        else:
            self._state_index_arg = None

        # Handle forward operator
        if isinstance(forward_operator, ForwardOperator):
            forward_operator = forward_operator.data

        # Handle dataframe input that should be series
        obs = validate_single_column_df("obs", obs)
        prior = validate_single_column_df("prior", prior)
        constant = validate_single_column_df("constant", constant)

        # Handle constant data
        if not isinstance(constant, pd.Series):
            constant_series = obs.copy(deep=True)
            constant_series[:] = constant if constant is not None else 0.0
            constant = constant_series

        # Assert dimensions are in indices
        if not all(dim in forward_operator.index.names for dim in self.obs_dims):
            raise ValueError(
                "Observation dimensions must be in the forward operator index."
            )
        if not all(dim in constant.index.names for dim in self.obs_dims):
            raise ValueError("Observation dimensions must be in the constant index.")
        if not all(dim in forward_operator.columns.names for dim in self.state_dims):
            raise ValueError(
                "State dimensions must be in the forward operator columns."
            )

        # Order levels if indexes are MultiIndex
        if isinstance(forward_operator.index, pd.MultiIndex):
            forward_operator = forward_operator.reorder_levels(
                list(self.obs_dims), axis="index"
            )
            obs = obs.reorder_levels(list(self.obs_dims))
            modeldata_mismatch = modeldata_mismatch.reorder_levels(self.obs_dims)
            constant = constant.reorder_levels(list(self.obs_dims))
        if isinstance(forward_operator.columns, pd.MultiIndex):
            forward_operator = forward_operator.reorder_levels(
                list(self.state_dims), axis="columns"
            )
            prior = prior.reorder_levels(list(self.state_dims))
            prior_error = prior_error.reorder_levels(list(self.state_dims))

        # Align inputs
        self.obs: pd.Series = obs
        self.prior: pd.Series = prior
        self.forward_operator: pd.DataFrame = forward_operator
        self.prior_error: CovarianceMatrix = prior_error
        self.modeldata_mismatch: CovarianceMatrix = modeldata_mismatch
        self.constant: pd.Series = constant
        self.align()

        # Store estimator details for solve()
        self._estimator_cls_or_name = estimator
        self._estimator_kwargs = (
            estimator_kwargs if estimator_kwargs is not None else {}
        )
        self._estimator: Estimator | None = None

        # Build xarray interface
        self.xr = XR(self)

        self._initialized = True

    @property
    def n_obs(self) -> int:
        """
        Number of observations.

        Returns
        -------
        int
            Number of observations.
        """
        return len(self.obs_index)

    @property
    def n_state(self) -> int:
        """
        Number of state variables.

        Returns
        -------
        int
            Number of state variables.
        """
        return len(self.state_index)

    def __setattr__(self, name, value):
        """Custom setattr to trigger alignment on data changes."""
        super().__setattr__(name, value)
        if self._initialized and name in {
            "obs_index",
            "state_index",
            "obs",
            "prior",
            "forward_operator",
            "prior_error",
            "modeldata_mismatch",
            "constant",
        }:
            self.align()

    def align(self):
        """Align all data components to the current obs_index and state_index."""
        # Round index coordinates to avoid floating point issues during alignment
        self._round_indices()

        # Re-compute indices in case underlying data has changed
        self.obs_index = self._compute_obs_index()
        self.state_index = self._compute_state_index()

        # Align inputs
        self.obs = self.obs.reindex(self.obs_index).dropna()
        self.prior = self.prior.reindex(self.state_index).dropna()
        self.forward_operator = self.forward_operator.reindex(
            index=self.obs_index, columns=self.state_index
        ).fillna(0.0)
        self.prior_error = self.prior_error.reindex(self.state_index)
        self.modeldata_mismatch = self.modeldata_mismatch.reindex(self.obs_index)
        self.constant = self.constant.reindex(self.obs_index).fillna(0.0)

    def _compute_obs_index(self) -> pd.Index:
        """Compute the observation index from the intersection of obs and forward operator."""
        obs_index = self.obs.index.intersection(self.forward_operator.index)
        if obs_index.empty:
            raise ValueError(
                "No overlapping indices between observations and forward operator."
            )
        return obs_index

    def _compute_state_index(self) -> pd.Index:
        """Compute the state index from the prior or the provided state_index."""
        state_index = self.prior.index.intersection(self.forward_operator.columns)
        if self._state_index_arg is not None:
            state_index = state_index.intersection(self._state_index_arg)

        if state_index.empty:
            raise ValueError("No overlapping indices for state variables.")
        return state_index

    def _round_indices(self):
        """Round all indices to the specified number of decimal places."""
        round_func = partial(round_index, decimals=self._coord_decimals)

        self.obs.index = round_func(self.obs.index)
        self.prior.index = round_func(self.prior.index)
        self.forward_operator.index = round_func(self.forward_operator.index)
        self.forward_operator.columns = round_func(self.forward_operator.columns)
        self.prior_error.index = round_func(self.prior_error.index)
        self.modeldata_mismatch.index = round_func(self.modeldata_mismatch.index)
        self.constant.index = round_func(self.constant.index)

    def solve(self) -> dict[str, pd.Series | CovarianceMatrix | pd.Series]:
        """
        Solve the inversion problem using the configured estimator.

        Returns
        -------
        dict[str, CovarianceMatrix | Data]
            A dictionary containing the posterior estimates:
            - 'posterior': Pandas series with the posterior mean model estimate.
            - 'posterior_error': CovarianceMatrix object with the posterior error covariance matrix.
            - 'posterior_obs': Pandas series with the posterior observation estimates.
        """
        estimator_input = {
            "z": self.obs.to_numpy(),
            "x_0": self.prior.to_numpy(),
            "H": self.forward_operator.to_numpy(),
            "S_0": self.prior_error.values,
            "S_z": self.modeldata_mismatch.values,
            "c": self.constant.to_numpy(),
            **self._estimator_kwargs,
        }
        self._estimator = self._init_estimator(
            self._estimator_cls_or_name,
            **estimator_input,
        )

        return {
            "posterior": self.posterior,
            "posterior_error": self.posterior_error,
            "posterior_obs": self.posterior_obs,
        }

    @property
    def estimator(self) -> Estimator:
        if self._estimator is None:
            raise AttributeError(
                "Estimator has not been initialized. Call the 'solve' method first."
            )
        return self._estimator

    def _init_estimator(self, estimator: str | type[Estimator], **kwargs) -> Estimator:
        """
        Initialize the estimator.

        Parameters
        ----------
        estimator : str or type[Estimator]
            The estimator class or its name as a string.
        kwargs : dict
            Input parameters for the estimator, including:
            - 'z': Observed data
            - 'x_0': Prior state estimate
            - 'H': Forward operator
            - 'S_0': Prior error covariance
            - 'S_z': Model-data mismatch covariance
            - 'c': Constant data (optional)

        Returns
        -------
        Estimator
            An instance of the specified estimator class.
        """
        if isinstance(estimator, str):
            if estimator not in ESTIMATOR_REGISTRY:
                raise ValueError(f"Estimator '{estimator}' is not registered.")
            estimator_cls = ESTIMATOR_REGISTRY[estimator]
        elif isinstance(estimator, type) and issubclass(estimator, Estimator):
            estimator_cls = estimator
        else:
            raise TypeError("Estimator must be a string or a subclass of Estimator.")

        return estimator_cls(**kwargs)

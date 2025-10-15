"""
Core inversion classes and functions.

This module provides core classes and utilities for formulating and solving inverse problems, including:
    - Abstract base classes for inversion estimators.
    - Registry for estimator implementations.
    - Forward operator and covariance matrix wrappers with index-aware functionality.
    - Utilities for convolving state vectors with forward operators.
    - The `InverseProblem` class, which orchestrates the alignment of data, prior information, error covariances, and the solution process.
"""

from functools import cached_property, partial

import pandas as pd
import xarray as xr

from fips.estimators import ESTIMATOR_REGISTRY, Estimator
from fips.matrices import CovarianceMatrix, SymmetricMatrix
from fips.operator import ForwardOperator
from fips.utils import round_index

# TODO
# - Obs aggregation
# - file io


class InverseProblem:
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
            Estimator class or its name as a string.
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
        # Validate state_index
        if state_index is None:
            state_index = prior.index
        if not isinstance(state_index, pd.Index):
            raise TypeError("state_index must be a pandas Index.")

        # Set problem dimensions
        self.obs_dims = tuple(obs.index.names)
        self.state_dims = tuple(prior.index.names)

        # Handle forward operator
        if isinstance(forward_operator, ForwardOperator):
            forward_operator = forward_operator.data

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
        if not all(dim in state_index.names for dim in self.state_dims):
            raise ValueError("State dimensions must be in the state index.")

        # Order levels if indexes are MultiIndex
        if isinstance(forward_operator.index, pd.MultiIndex):
            forward_operator = forward_operator.reorder_levels(
                self.obs_dims, axis="index"
            )
            obs = obs.reorder_levels(self.obs_dims)
            modeldata_mismatch = modeldata_mismatch.reorder_levels(self.obs_dims)
            constant = constant.reorder_levels(self.obs_dims)
        if isinstance(forward_operator.columns, pd.MultiIndex):
            forward_operator = forward_operator.reorder_levels(
                self.state_dims, axis="columns"
            )
            prior = prior.reorder_levels(self.state_dims)
            prior_error = prior_error.reorder_levels(self.state_dims)

        # Round index coordinates to avoid floating point issues during alignment
        round_coords = partial(round_index, decimals=coord_decimals)
        state_index = round_coords(state_index)
        obs.index = round_coords(obs.index)
        prior.index = round_coords(prior.index)
        forward_operator.index = round_coords(forward_operator.index)
        forward_operator.columns = round_coords(forward_operator.columns)
        prior_error.index = round_coords(prior_error.index)
        modeldata_mismatch.index = round_coords(modeldata_mismatch.index)
        constant.index = round_coords(constant.index)

        # Define the obs index as the intersection of the observation and forward operator obs indices
        obs_index = obs.index.intersection(forward_operator.index)
        if obs_index.empty:
            raise ValueError(
                "No overlapping indices between observations and forward operator."
            )

        # Align inputs
        self.obs = obs.reindex(obs_index).dropna()
        self.prior = prior.reindex(state_index).dropna()
        self.forward_operator = forward_operator.reindex(
            index=obs_index, columns=state_index
        ).fillna(0.0)
        self.prior_error = prior_error.reindex(state_index)
        self.modeldata_mismatch = modeldata_mismatch.reindex(obs_index)
        self.constant = constant.reindex(obs_index).fillna(0.0)

        # Store the problem indices
        self.obs_index = obs_index
        self.state_index = state_index

        # Initialize the estimator
        estimator_input = {
            "z": self.obs.values,
            "x_0": self.prior.values,
            "H": self.forward_operator.values,
            "S_0": self.prior_error.values,
            "S_z": self.modeldata_mismatch.values,
            "c": self.constant.values,
        }
        if estimator_kwargs is None:
            estimator_kwargs = {}

        self.estimator = self._init_estimator(
            estimator, estimator_input=estimator_input, **estimator_kwargs
        )

        # Build xarray interface
        self.xr = self._XR(self)

    def _init_estimator(
        self, estimator: str | type[Estimator], estimator_input: dict, **kwargs
    ) -> Estimator:
        """
        Initialize the estimator.

        Parameters
        ----------
        estimator : str or type[Estimator]
            The estimator class or its name as a string.
        estimator_input : dict
            Input parameters for the estimator, including:
            - 'z': Observed data
            - 'x_0': Prior state estimate
            - 'H': Forward operator
            - 'S_0': Prior error covariance
            - 'S_z': Model-data mismatch covariance
            - 'c': Constant data (optional)
        kwargs : dict
            Additional keyword arguments to pass to the estimator constructor.

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

        z = estimator_input["z"]
        x_0 = estimator_input["x_0"]
        H = estimator_input["H"]
        S_0 = estimator_input["S_0"]
        S_z = estimator_input["S_z"]
        c = estimator_input.get("c")

        return estimator_cls(z=z, x_0=x_0, H=H, S_0=S_0, S_z=S_z, c=c, **kwargs)

    def solve(self) -> dict[str, pd.Series | SymmetricMatrix | pd.Series]:
        """
        Solve the inversion problem using the configured estimator.

        Returns
        -------
        dict[str, State | Covariance | Data]
            A dictionary containing the posterior estimates:
            - 'posterior': Pandas series with the posterior mean model estimate.
            - 'posterior_error': Covariance object with the posterior error covariance matrix.
            - 'posterior_obs': Pandas series with the posterior observation estimates.
        """
        return {
            "posterior": self.posterior,
            "posterior_error": self.posterior_error,
            "posterior_obs": self.posterior_obs,
        }

    @property
    def n_obs(self) -> int:
        """
        Number of observations.

        Returns
        -------
        int
            Number of observations.
        """
        return self.estimator.n_z

    @property
    def n_state(self) -> int:
        """
        Number of state variables.

        Returns
        -------
        int
            Number of state variables.
        """
        return self.estimator.n_x

    @cached_property
    def posterior(self) -> pd.Series:
        """
        Posterior state estimate.

        Returns
        -------
        pd.Series
            Pandas series with the posterior mean model estimate.
        """
        x_hat = self.estimator.x_hat
        return pd.Series(x_hat, index=self.state_index, name="posterior")

    @cached_property
    def posterior_obs(self) -> pd.Series:
        """
        Posterior observation estimates.

        Returns
        -------
        pd.Series
            Pandas series with the posterior observation estimates.
        """
        y_hat = self.estimator.y_hat
        return pd.Series(y_hat, index=self.obs_index, name="posterior_obs")

    @cached_property
    def posterior_error(self) -> SymmetricMatrix:
        """
        Posterior error covariance matrix.

        Returns
        -------
        CovarianceMatrix
            CovarianceMatrix instance with the posterior error covariance matrix.
        """
        S_hat = self.estimator.S_hat
        return SymmetricMatrix(
            pd.DataFrame(S_hat, index=self.state_index, columns=self.state_index)
        )

    @cached_property
    def prior_obs(self) -> pd.Series:
        """
        Prior observation estimates.

        Returns
        -------
        pd.Series
            Pandas series with the prior observation estimates.
        """
        y_0 = self.estimator.y_0
        return pd.Series(y_0, index=self.obs_index, name="prior_obs")

    class _XR:
        """
        Xarray interface for Inversion data.
        """

        def __init__(self, inversion: "InverseProblem"):
            self._inversion = inversion

        def __getattr__(self, attr):
            """
            Get an xarray representation of an attribute from the inversion object.

            Parameters
            ----------
            attr : str
                Attribute name.

            Returns
            -------
            xr.DataArray
                Xarray representation of the attribute.

            Raises
            ------
            AttributeError
                If the attribute does not exist.
            TypeError
                If the attribute type is not supported.
            """
            if attr == "_inversion":
                return self._inversion
            if hasattr(self._inversion, attr):
                obj = getattr(self._inversion, attr)
                if isinstance(obj, pd.Series):
                    return self._series_to_xarray(series=obj, attr=attr)
                elif isinstance(obj, pd.DataFrame):
                    return self._dataframe_to_xarray(df=obj, attr=attr)
                else:
                    raise TypeError(f"Unable to represent {type(obj)} as Xarray.")
            else:
                raise AttributeError(
                    f"'{type(self._inversion).__name__}' object has no attribute '{attr}'"
                )

        def __setattr__(self, *args):
            """
            Prevent setting attributes on the Xarray interface.

            Parameters
            ----------
            *args : tuple
                Attribute name and value.

            Raises
            ------
            AttributeError
                If attempting to set an attribute.
            """
            if args[0] == "_inversion":
                super().__setattr__(*args)
            else:
                raise AttributeError(
                    f"Cannot set attribute '{args[0]}' on Xarray interface."
                )

        @staticmethod
        def _series_to_xarray(series: pd.Series, attr) -> xr.DataArray:
            """
            Convert a Pandas Series to an Xarray DataArray.

            Parameters
            ----------
            series : pd.Series
                Pandas Series to convert.
            attr : str
                Attribute name.

            Returns
            -------
            xr.DataArray
                Xarray DataArray representation of the series.
            """
            series = series.copy()
            series.name = attr
            return series.to_xarray()

        @staticmethod
        def _dataframe_to_xarray(df: pd.DataFrame, attr) -> xr.DataArray:
            """
            Convert a Pandas DataFrame to an Xarray DataArray.

            Parameters
            ----------
            df : pd.DataFrame
                Pandas DataFrame to convert.
            attr : str
                Attribute name.

            Returns
            -------
            xr.DataArray
                Xarray DataArray representation of the DataFrame.
            """
            df = df.copy()
            if isinstance(df.columns, pd.MultiIndex):
                # Stack all levels of the columns MultiIndex into the index
                n_levels = len(df.columns.levels)
                s = df.stack(list(range(n_levels)), future_stack=True)
            else:
                s = df.stack(future_stack=True)
            return InverseProblem._XR._series_to_xarray(series=s, attr=attr)

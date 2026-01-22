from functools import cached_property
from typing import TYPE_CHECKING

import pandas as pd
import xarray as xr

from fips.estimators import Estimator
from fips.matrices import CovarianceMatrix, SymmetricMatrix
from fips.utils import dataframe_to_xarray, series_to_xarray

if TYPE_CHECKING:
    from fips.problem import InverseProblem


class EstimatorOutput:
    """Estimator output interface for inversion attributes."""

    obs_index: pd.Index
    state_index: pd.Index
    estimator: Estimator | property

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
    def posterior_error(self) -> CovarianceMatrix:
        """
        Posterior error covariance matrix.

        Returns
        -------
        CovarianceMatrix
            CovarianceMatrix instance with the posterior error covariance matrix.
        """
        S_hat = self.estimator.S_hat
        return CovarianceMatrix(
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

    @cached_property
    def U_red(self) -> pd.Series:
        U_red = self.estimator.U_red
        return pd.Series(U_red, index=self.state_index, name="U_red")


class XR:
    """
    Xarray interface for inversion attributes.
    """

    def __init__(self, inversion: "InverseProblem"):
        self._inversion = inversion

    def __getattr__(self, attr) -> xr.DataArray:
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
        obj = getattr(self._inversion, attr)
        if isinstance(obj, pd.Series):
            return series_to_xarray(series=obj, name=attr)
        elif isinstance(obj, (pd.DataFrame, SymmetricMatrix)):
            return dataframe_to_xarray(df=obj.data, name=attr)
        else:
            raise TypeError(f"Unable to represent {type(obj)} as Xarray.")

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

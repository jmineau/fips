from functools import cached_property
from typing import TYPE_CHECKING

import pandas as pd
import xarray as xr

from fips.estimators import Estimator
from fips.matrices import CovarianceMatrix

if TYPE_CHECKING:
    from fips.problem import InverseProblem


class EstimatorOutput:
    """Estimator output interface for inversion attributes."""

    obs_index: pd.Index | property
    state_index: pd.Index | property
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

    def __getattr__(self, attr) -> xr.DataArray | xr.Dataset:
        """
        Get an xarray representation of an attribute from the inversion object.

        Parameters
        ----------
        attr : str
            Attribute name.

        Returns
        -------
        xr.DataArray | xr.Dataset
            Xarray representation of the attribute.

        Raises
        ------
        AttributeError
            If the attribute does not exist.
        TypeError
            If the attribute type is not supported.
        """
        obj = getattr(self._inversion, attr)
        return convert_to_xarray(obj, name=attr)

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


#  --- XARRAY HELPERS ---

def series_to_xarray(series: pd.Series, name=None) -> xr.DataArray:
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
    if name is not None:
        series.name = name
    return series.to_xarray()


def dataframe_to_xarray(df: pd.DataFrame, name=None) -> xr.DataArray:
    """
    Convert a Pandas DataFrame to an Xarray DataArray.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame to convert.
    name : str
        Name for the resulting DataArray.

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
    if isinstance(s, pd.DataFrame):
        raise ValueError("DataFrame could not be stacked into a Series.")
    return series_to_xarray(series=s, name=name)


def convert_to_xarray(obj: pd.Series | pd.DataFrame, name: str
                        ) -> xr.DataArray | xr.Dataset:
    # Handle Series with blocks -> Dataset
    if isinstance(obj, pd.Series):
        if isinstance(obj.index, pd.MultiIndex) and 'block' in obj.index.names:
            ds = xr.Dataset()
            blocks = obj.index.get_level_values('block').unique()
            
            for block in blocks:
                # Extract subset for this block
                # .xs drops the 'block' level, leaving the inner index (e.g. time/lat/lon)
                subset = obj.xs(block, level='block')
                
                # Convert to DataArray
                da = series_to_xarray(subset, name=block)
                ds[block] = da
            return ds
        else:
            # Flat Series -> DataArray
            return series_to_xarray(obj, name=name)

    # Handle DataFrame (Matrices) -> DataArray
    elif isinstance(obj, pd.DataFrame):
        return dataframe_to_xarray(obj, name=name)

    raise TypeError(f"Cannot convert object of type {type(obj)} to xarray.")
"""Interfaces and mixins for inverse problem outputs.

This module provides base classes for managing estimator outputs including
caching, type conversion to pandas and xarray, and metadata.
"""

from enum import Enum
from typing import TYPE_CHECKING, Literal

import pandas as pd
import xarray as xr

from fips.converters import dataframe_to_xarray, series_to_xarray
from fips.covariance import CovarianceMatrix
from fips.estimators import OUTPUT_PROPERTY_NAMES, Estimator
from fips.indices import xs
from fips.serialization import Pickleable
from fips.structures import Matrix, Vector

if TYPE_CHECKING:
    from fips.problem import InverseProblem


class ComponentType(Enum):
    """Metadata for inverse components."""

    SCALAR = "scalar"  # single value
    VECTOR = "vector"  # 1D state-indexed
    MATRIX = "matrix"  # 2D state x state
    COVMATRIX = "covariance_matrix"  # 2D covariance matrix


class EstimatorOutput(Pickleable):
    state_index: pd.Index | property
    obs_index: pd.Index | property
    estimator: Estimator | property

    OUTPUTS = {
        # component name: (ComponentType, row_index, col_index)
        "posterior": (ComponentType.VECTOR, "state"),
        "posterior_error": (ComponentType.COVMATRIX, "state", "state"),
        "posterior_obs": (ComponentType.VECTOR, "obs"),
        "prior_obs": (ComponentType.VECTOR, "obs"),
        "kalman_gain": (ComponentType.MATRIX, "obs", "state"),
        "averaging_kernel": (ComponentType.MATRIX, "state", "state"),
        "U_red": (ComponentType.VECTOR, "state"),
        "leverage": (ComponentType.VECTOR, "obs"),
        "reduced_chi2": (ComponentType.SCALAR,),
        "R2": (ComponentType.SCALAR,),
        "RMSE": (ComponentType.SCALAR,),
        "DOFS": (ComponentType.SCALAR,),
        "uncertainty_reduction": (ComponentType.SCALAR,),
    }

    def __init__(self):
        """Initialize output cache for caching computed results."""
        self._output_cache: dict = {}

    def _wrap_output(
        self,
        attr,
        component,
        component_type: ComponentType,
        index: Literal["state", "obs"] | None = None,
        column: Literal["state", "obs"] | None = None,
    ):
        # Direct return for scalars
        if component_type == ComponentType.SCALAR:
            return component

        # Determine row index
        if index == "state":
            idx = self.state_index
        elif index == "obs":
            idx = self.obs_index
        else:
            raise ValueError(f"Unknown index type: {index}")

        # Wrap Vectors
        if component_type == ComponentType.VECTOR:
            # Create Vector with index that already has 'block' level from Vector inputs
            s = pd.Series(component, index=idx, name=attr)
            return Vector.from_series(s, name=attr)

        # Determine Matrix class
        if component_type == ComponentType.MATRIX:
            Matrix_Class = Matrix
        elif component_type == ComponentType.COVMATRIX:
            Matrix_Class = CovarianceMatrix
        else:
            raise ValueError(f"Unknown component type: {component_type}")

        # Determine column index
        if column is None:
            raise ValueError("Column index must be specified for MATRIX components.")
        elif column == "state":
            col = self.state_index
        elif column == "obs":
            col = self.obs_index
        else:
            raise ValueError(f"Unknown column type: {column}")

        # Create Matrix
        df = pd.DataFrame(component, index=idx, columns=col)
        return Matrix_Class(df)

    def __getattr__(self, attr: str):
        if attr in self.OUTPUTS:
            # Check cache first
            if attr in self._output_cache:
                return self._output_cache[attr]

            # Compute and cache the output
            meta = self.OUTPUTS[attr]
            component_type = meta[0]
            row_index = meta[1] if len(meta) > 1 else None
            col_index = meta[2] if len(meta) > 2 else None

            estimator_attr = OUTPUT_PROPERTY_NAMES.get(attr, attr)
            result = self._wrap_output(
                attr=attr,
                component=getattr(self.estimator, estimator_attr),
                component_type=component_type,
                index=row_index,
                column=col_index,
            )

            # Cache the result
            self._output_cache[attr] = result
            return result

        return super().__getattribute__(attr)


class _ReadOnly:
    """
    Mixin class to prevent setting attributes.
    """

    __slots__ = ("_inversion",)

    def __init__(self, inversion: "InverseProblem"):
        object.__setattr__(self, "_inversion", inversion)

    def __setattr__(self, name: str, value) -> None:
        """Prevent all attribute assignments."""
        raise AttributeError(f"Cannot set attribute '{name}' on read-only interface.")


class PD(_ReadOnly):
    """
    Pandas interface for inversion attributes.
    """

    def __getattr__(self, attr) -> pd.Series | pd.DataFrame:
        """
        Get a pandas representation of an attribute from the inversion object.

        Parameters
        ----------
        attr : str
            Attribute name.

        Returns
        -------
        pd.Series | pd.DataFrame
            Pandas representation of the attribute.

        Raises
        ------
        AttributeError
            If the attribute does not exist.
        TypeError
            If the attribute type is not supported.
        """
        obj = getattr(self._inversion, attr)
        if isinstance(obj, (Vector, Matrix)):
            return obj.data
        else:
            raise TypeError(f"Cannot convert object of type {type(obj)} to pandas.")


class XR(_ReadOnly):
    """
    Xarray interface for inversion attributes.
    """

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
        obj = getattr(self._inversion.pd, attr)
        return convert_to_xarray(obj, name=attr)


#  --- XARRAY HELPERS ---


def convert_to_xarray(
    obj: pd.Series | pd.DataFrame, name: str
) -> xr.DataArray | xr.Dataset:
    # Handle Series with blocks -> Dataset
    if isinstance(obj, pd.Series):
        if isinstance(obj.index, pd.MultiIndex) and "block" in obj.index.names:
            ds = xr.Dataset()
            blocks = obj.index.get_level_values("block").unique()

            for block in blocks:
                # Extract subset for this block
                # .xs drops the 'block' level, leaving the inner index (e.g. time/lat/lon)
                subset = xs(obj, block, level="block")

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

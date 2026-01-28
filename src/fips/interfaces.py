from enum import Enum
from typing import TYPE_CHECKING, Literal

import pandas as pd
import xarray as xr

from fips.estimators import Estimator
from fips.matrices import CovarianceMatrix, Matrix
from fips.vectors import Vector

if TYPE_CHECKING:
    from fips.problem import InverseProblem


class ComponentType(Enum):
    """Metadata for inverse components."""

    SCALAR = "scalar"  # single value
    VECTOR = "vector"  # 1D state-indexed
    MATRIX = "matrix"  # 2D state x state
    COVMATRIX = "covariance_matrix"  # 2D covariance matrix


# INPUTS = {
#     # component name: (ComponentType, row_index, col_index, [aliases])
#     'obs': (ComponentType.VECTOR, 'obs', None, ['observations']),
#     'prior': (ComponentType.VECTOR, 'state', None, ['prior_state']),
#     'forward_operator': (ComponentType.MATRIX, 'obs', 'state', ['jacobian', 'forwardoperater']),
#     'prior_error': (ComponentType.COVMATRIX, 'state', 'state', ['prior_cov']),
#     'modeldata_mismatch': (ComponentType.COVMATRIX, 'obs', 'obs', ['obs_error', 'obs_cov', 'mdm']),
# }
# OUTPUTS = {
#     'posterior': (ComponentType.VECTOR, 'state', None, ['posterior_state', 'analysis']),
#     'posterior_error': (ComponentType.COVMATRIX, 'state', 'state', ['posterior_cov']),
#     'posterior_obs': (ComponentType.VECTOR, 'obs', None, ['posterior_observations']),
#     'prior_obs': (ComponentType.VECTOR, 'obs', None, ['prior_observations']),
#     'kalman_gain': (ComponentType.MATRIX, 'state', 'obs', []),
#     'averaging_kernel': (ComponentType.MATRIX, 'state', 'state', []),
#     'U_red': (ComponentType.MATRIX, 'state', 'state', []),
#     'leverage': (ComponentType.VECTOR, 'state', None, []),
# }

INPUTS = {
    # component name: (ComponentType, row_index, col_index)
    "obs": (ComponentType.VECTOR, "obs", None),
    "prior": (ComponentType.VECTOR, "state", None),
    "forward_operator": (ComponentType.MATRIX, "obs", "state"),
    "prior_error": (ComponentType.COVMATRIX, "state", "state"),
    "modeldata_mismatch": (ComponentType.COVMATRIX, "obs", "obs"),
}

# Aliases mapping -> canonical name
ALIASES = {
    "observations": "obs",
    "prior_state": "prior",
    "jacobian": "forward_operator",
    "forwardoperator": "forward_operator",
    "prior_cov": "prior_error",
    "obs_error": "modeldata_mismatch",
    "obs_cov": "modeldata_mismatch",
    "mdm": "modeldata_mismatch",
    "posterior_state": "posterior",
    "analysis": "posterior",
    "posterior_cov": "posterior_error",
    "posterior_observations": "posterior_obs",
    "prior_observations": "prior_obs",
}

# Combined lookup: canonical names + aliases
# _ALL_COMPONENTS = {**INPUTS, **OUTPUTS, **ALIASES}


def canonicalize(name: str) -> str:
    """
    Convert an alias to its canonical component name.

    Parameters
    ----------
    name : str
        Component name or alias.

    Returns
    -------
    str
        Canonical component name.

    Raises
    ------
    KeyError
        If the name is not a valid component or alias.
    """
    if name in ALIASES:
        return ALIASES[name]
    elif name in _ALL_COMPONENTS:
        return name
    else:
        valid = list(INPUTS.keys()) + list(OUTPUTS.keys())
        raise KeyError(
            f"Unknown component '{name}'. Valid components are: {valid}. "
            f"Aliases: {list(ALIASES.keys())}"
        )


class EstimatorOutput:
    state_index: pd.Index
    obs_index: pd.Index
    estimator: Estimator

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
            # Create Vector
            s = pd.Series(component, index=idx, name=attr)
            return Vector.from_series(s)

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
            component_type, row_index, col_index = self.OUTPUTS[attr]
            return self._wrap_output(
                attr=attr,
                component=getattr(self.estimator, attr),
                component_type=component_type,
                index=row_index,
                column=col_index,
            )

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
        if isinstance(obj, Vector | Matrix):
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


def series_to_xarray(series: pd.Series, name=None) -> xr.DataArray:
    """
    Convert a Pandas Series to an Xarray DataArray.

    Parameters
    ----------
    series : pd.Series
        Pandas Series to convert.
    name : str
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
                subset = obj.xs(block, level="block")

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

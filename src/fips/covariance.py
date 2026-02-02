import logging
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr
from typing_extensions import Self

from fips.indices import xs
from fips.structures import SymmetricMatrix
from fips.utils.spacetime import build_latlon_corr_matrix, build_temporal_corr_matrix

logger = logging.getLogger(__name__)


def variances_as_series(
    variances: pd.Series | xr.DataArray | np.ndarray | float,
    index: pd.Index | None = None,
) -> pd.Series:
    """
    Convert variances to a pandas Series with the given index.

    Parameters
    ----------
    variances : scalar | sequence | xr.DataArray | pd.Series
        Variances.
    index : pd.Index
        Index for the variances if variances is a scalar or sequence.
    Returns
    -------
    pd.Series
        Prior error variances as a pandas Series.
    """

    # If already a pandas Series, use it (warn if index argument is provided)
    if isinstance(variances, pd.Series):
        if index is not None:
            warnings.warn(
                message="Provided 'index' is ignored when 'variances' is a pandas Series; "
                "the Series' own index will be used.",
                category=UserWarning,
                stacklevel=3,
            )
        logger.debug(f"Using variances from Series with {len(variances)} entries")
        return variances

    # Xarray -> pandas Series
    if isinstance(variances, xr.DataArray):
        return variances.to_series()

    # Scalar (any numeric scalar, numpy or python)
    if np.isscalar(variances):
        if index is None:
            raise ValueError("index must be provided if variances is a scalar")
        arr = np.full(len(index), variances)
        return pd.Series(arr, index=index)

    # Sequence-like (lists, tuples, numpy arrays, etc.), but exclude strings/bytes
    if isinstance(variances, (np.ndarray, Sequence)) and not isinstance(
        variances, (str, bytes)
    ):
        if index is None:
            raise ValueError("index must be provided if variances is a sequence")
        if len(variances) != len(index):
            raise ValueError("Length of variances must match length of index")
        return pd.Series(np.asarray(variances), index=index)

    raise ValueError("variances must be a scalar, sequence, xr.DataArray, or pd.Series")


class CovarianceMatrix(SymmetricMatrix):
    """
    Represents a symmetric Covariance Matrix (Prior Error or Obs Error).

    Can be initialized from a DataFrame directly, or created from a Vector
    to enable block-based construction methods using set_block().
    """

    @classmethod
    def from_variances(
        cls,
        variances: pd.Series | np.ndarray | float,
        index: pd.Index | None = None,
        **kwargs,
    ) -> Self:
        # Normalize variances to a pandas Series
        variances = variances_as_series(variances, index=index)

        # Create diagonal covariance matrix from variances
        index = variances.index
        values = np.diag(variances.to_numpy())

        logger.debug(f"Building diagonal covariance with {len(index)} elements")
        return cls.from_numpy(array=values, index=index)

    def get_variances(self, block: str | None = None) -> pd.Series:
        """
        Get the variances (diagonal elements) of the covariance matrix.

        Parameters
        ----------
        block : str, optional
            If specified, return variances only for the given block.

        Returns
        -------
        pd.Series
            Series of variances indexed by state vector index.
        """
        variances = pd.Series(
            np.diag(self.data.values), index=self.data.index, name="variance"
        )
        if block is not None:
            variances = xs(variances, block, level="block")
        return variances


class SpaceTimeCovariance(CovarianceMatrix):
    @classmethod
    def from_variances(
        cls,
        variances: pd.Series | np.ndarray | float,
        index: pd.Index | None = None,
        spatial_corr: dict | None = None,
        temporal_corr: dict | None = None,
        time_dim: str = "time",
        x_dim: str = "lon",
        y_dim: str = "lat",
        **kwargs,
    ) -> Self:
        """
        Create a SpaceTimeCovariance instance from variances and correlation matrices.

        Parameters
        ----------
        variances : float | np.ndarray
            Prior error variances for the inversion.
        index : pd.MultiIndex
            MultiIndex for flux state.
        spatial_corr : dict, optional
            Spatial correlation parameters for the inversion, by default None.
            If None, the spatial error correlation matrix is an identity matrix.
            See `build_spatial_corr` for more options.
        temporal_corr : dict, optional
            Temporal correlation parameters for the inversion, by default None.
            If None, the temporal error correlation matrix is an identity matrix.
            See `build_temporal_corr` for more options.
        time_dim : str
            Name of the time dimension in the index, by default "time".
        x_dim : str
            Name of the x (longitude) dimension in the index, by default "lon".
        y_dim : str
            Name of the y (latitude) dimension in the index, by default "lat".

        Returns
        -------
        SpaceTimeCovariance
            A new SpaceTimeCovariance instance
        """
        variances = variances_as_series(variances=variances, index=index)

        index = variances.index
        variances = variances.to_numpy()

        if spatial_corr is None and temporal_corr is None:
            # If no correlations specified, return diagonal covariance matrix
            S_0 = np.diag(variances)
            return cls.from_numpy(S_0, index=index)

        # Get unique times, lats, and lons
        times = index.get_level_values(time_dim).unique()
        lats = index.get_level_values(y_dim).unique().to_numpy()
        lons = index.get_level_values(x_dim).unique().to_numpy()

        # Build spatial correlation matrix
        if spatial_corr is None:
            spatial_corr_matrix = np.eye(len(lats) * len(lons))
        elif isinstance(spatial_corr, dict):
            spatial_corr_matrix = build_latlon_corr_matrix(
                lats=lats, lons=lons, **spatial_corr
            )
        else:
            raise ValueError("spatial_corr must be a dict or None")
        logger.debug(
            f"Built spatial correlation matrix with shape {spatial_corr_matrix.shape}"
        )
        # Build temporal correlation matrix
        if temporal_corr is None:
            temporal_corr_matrix = np.eye(len(times))
        elif isinstance(temporal_corr, dict):
            temporal_corr_matrix = build_temporal_corr_matrix(
                times=times, **temporal_corr
            )
        else:
            raise ValueError("temporal_corr must be a dict or None")
        logger.debug(
            f"Built temporal correlation matrix with shape {temporal_corr_matrix.shape}"
        )

        # Combine spatial and temporal correlations using Kronecker product
        sigma = np.diag(np.sqrt(variances))  # Scale by variances
        kron = np.kron(temporal_corr_matrix, spatial_corr_matrix)  # order matters
        S_0 = sigma @ kron @ sigma
        logger.debug(f"Built spatiotemporal covariance with shape {S_0.shape}")
        return cls.from_numpy(S_0, index=index)

    def apply_temporal_decay(
        self,
        length_scale: pd.Timedelta | str,
        interday: bool = True,
        time_dim: str = "obs_time",
    ) -> Self:
        """
        Apply temporal decay to the covariance matrix.

        Parameters
        ----------
        length_scale : pd.Timedelta or str
            Length scale for the temporal decay (e.g., '32h' for 32 hours)
        interday : bool
            Whether to apply decay across days. If False, decay is only applied
            within the same day.
        time_dim : str
            Name of the time dimension

        Returns
        -------
        SpaceTimeCovariance
            A new instance with time decay applied
        """

        # Define the decay function
        def calculate_temporal_decay(
            group: pd.DataFrame, time_dim, length_scale
        ) -> np.ndarray:
            times = group[time_dim]  # Extract times for this group
            return build_temporal_corr_matrix(times=times, length_scale=length_scale)

        # Find all dimensions other than the decay dimension
        groupers = [col for col in self.index.names if col != time_dim]

        # If not spanning days, add the date as a grouper
        if not interday:
            dates = self.index.get_level_values(time_dim).date
            groupers.append(dates)

        # Apply the decay
        logger.debug(
            f"Applying temporal decay (length_scale={length_scale}, interday={interday})"
        )
        return self.apply_groupwise(
            groupers=groupers,
            func=calculate_temporal_decay,
            time_dim=time_dim,
            length_scale=length_scale,
            diagonal=False,
        )

    def apply_spatial_decay(
        self,
        length_scale: float,
        location_latlon_mapper: dict[str, tuple[float, float]],
        earth_radius: float | None = None,
        spatial_dim: str = "obs_location",
    ) -> Self:
        """
        Apply spatial decay to the covariance matrix.

        Parameters
        ----------
        length_scale : float
            Length scale for the spatial decay (in km)
        location_latlon_mapper : dict
            Dictionary mapping location names to (latitude, longitude) tuples
        earth_radius : float
            Earth radius in km
        spatial_dim : str
            Name of the spatial dimension

        Returns
        -------
        SpaceTimeCovariance
            A new instance with spatial decay applied
        """

        # Define the decay function
        def calculate_spatial_decay(
            group: pd.DataFrame,
            spatial_dim: str,
            length_scale: float,
            earth_radius: float,
        ) -> np.ndarray:
            # Get lat/lon for each location in the group
            lats = [
                location_latlon_mapper[loc][0] for loc in group[spatial_dim]
            ]  # TODO make this more flexible
            lons = [location_latlon_mapper[loc][1] for loc in group[spatial_dim]]
            return build_latlon_corr_matrix(
                lats=lats,
                lons=lons,
                length_scale=length_scale,
                earth_radius=earth_radius,
            )

        # Find all dimensions other than the decay dimension
        groupers = [col for col in self.index.names if col != spatial_dim]

        # Apply the decay
        logger.debug(
            f"Applying spatial decay (length_scale={length_scale}, spatial_dim={spatial_dim})"
        )
        return self.apply_groupwise(
            groupers=groupers,
            func=calculate_spatial_decay,
            spatial_dim=spatial_dim,
            length_scale=length_scale,
            earth_radius=earth_radius,
            diagonal=False,
        )

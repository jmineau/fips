import math
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing_extensions import Self

from fips.matrices import CovarianceMatrix
from fips.problems.flux.utils import haversine_matrix, time_decay_matrix


class SpaceTimeCovariance(CovarianceMatrix):
    """
    SpaceTimeCovariance: Base class for space-time covariance matrices.

    Subclass of CovarianceMatrix for representing space-time covariance matrices
    in flux inversion problems.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def plot(self, **kwargs) -> plt.Axes:
        """Plot the covariance matrix using imshow."""
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", None))
        p = ax.imshow(self.data, **kwargs)
        ax.set_xticks(np.arange(len(self.index)))
        ax.set_xticklabels(self.index.values, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(self.index)))
        ax.set_yticklabels(self.index.values)
        ax.set_title(type(self).__name__)
        fig.colorbar(p, ax=ax)
        plt.show()

        return ax


class PriorError(SpaceTimeCovariance):
    """
    PriorError: Prior Error Covariance Matrix.

    Subclass of CovarianceMatrix for representing the prior error covariance matrix
    in flux inversion problems.
    """

    @classmethod
    def from_variances(
        cls,
        variances: pd.Series | np.ndarray | float,
        index: pd.Index | None = None,
        spatial_corr: dict | None = None,
        temporal_corr: dict | None = None,
        **kwargs,
    ) -> "PriorError":
        """
        Create a PriorError instance from variances and correlation matrices.

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

        Returns
        -------
        PriorError
            Instance of PriorError.
        """
        variances = cls._variances_as_series(variances=variances, index=index)

        # Sort index to ensure consistent ordering
        variances = variances.reorder_levels(["time", "lat", "lon"]).sort_index()

        index = variances.index
        variances = variances.to_numpy()

        # Ensure index is a MultiIndex and access level values in a type-safe way
        if not isinstance(index, pd.MultiIndex):
            raise TypeError(
                "index must be a pandas.MultiIndex with levels ['time', 'lat', 'lon']"
            )

        # Get unique times, lats, and lons
        times = index.get_level_values("time").unique()
        lats = index.get_level_values("lat").unique().to_numpy()
        lons = index.get_level_values("lon").unique().to_numpy()

        # Build spatial correlation matrix
        if spatial_corr is None:
            spatial_corr_matrix = np.eye(len(lats) * len(lons))
        elif isinstance(spatial_corr, dict):
            spatial_corr_matrix = cls.build_spatial_corr_matrix(
                lats=lats, lons=lons, **spatial_corr
            )
        else:
            raise ValueError("spatial_corr must be a dict or None")
        # Build temporal correlation matrix
        if temporal_corr is None:
            temporal_corr_matrix = np.eye(len(times))
        elif isinstance(temporal_corr, dict):
            temporal_corr_matrix = cls.build_temporal_corr_matrix(
                times=times, **temporal_corr
            )
        else:
            raise ValueError("temporal_corr must be a dict or None")

        # Combine spatial and temporal correlations using Kronecker product
        sigma = np.diag(np.sqrt(variances))  # Scale by variances
        kron = np.kron(temporal_corr_matrix, spatial_corr_matrix)  # order matters
        S_0 = sigma @ kron @ sigma
        return cls.from_numpy(S_0, index=index)

    @staticmethod
    def build_spatial_corr_matrix(
        lats: np.ndarray, lons: np.ndarray, length_scale: float
    ) -> np.ndarray:
        """
        Build the spatial error correlation matrix for the inversion.

        Parameters
        ----------
        lats : np.ndarray
            Latitude coordinates of the flux cells.
        lons : np.ndarray
            Longitude coordinates of the flux cells.
        length_scale : float
            E-folding length scale for the exponential decay of spatial correlations (in km).

        Returns
        -------
        np.ndarray
            Spatial error correlation matrix for the inversion.
        """
        # Get the lat/lon coordinates of the grid cells
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Calculate pairwise haversine distances between grid cells
        distances = haversine_matrix(lats=lat_grid.ravel(), lons=lon_grid.ravel())
        corr = np.exp((-1 * distances) / length_scale)

        return corr

    @staticmethod
    def build_temporal_corr_matrix(
        times: pd.Index | pd.DatetimeIndex,
        method: str | dict | None = None,
        length_scale: pd.Timedelta | str | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Build the temporal error correlation matrix for the inversion.
        Has dimensions of flux_times x flux_times.

        Parameters
        ----------
        times : pd.Index or pd.DatetimeIndex
            Datetime-like index of flux times. Values will be coerced to a
            pandas.DatetimeIndex internally so attributes like `.hour` and
            `.month` are available.
        method : dict, optional
            Method for calculating the temporal error correlation matrix.
            The key defines the method and the value is the weight for the method.
            Options include:
                - 'exp': exponentially decaying with an e-folding length of self.t_bins freq
                - 'diel': like 'exp', except correlations are only non-zero for the same time of day
                - 'clim': each month is highly correlated with the same month in other years

        Returns
        -------
        np.ndarray
            Temporal error correlation matrix for the inversion.
        """
        # Coerce to DatetimeIndex so we can use .hour/.month safely
        times = pd.DatetimeIndex(times)

        # Handle method input
        if method is None:
            method = {"exp": 1.0}
        elif isinstance(method, str):
            method = {method: 1.0}
        elif isinstance(method, dict):
            if not math.isclose(float(sum(method.values())), 1.0):
                raise ValueError("Weights for temporal error methods must sum to 1.0")
        else:
            raise ValueError("method must be a dict")

        # Initialize the temporal correlation matrix
        N = len(times)  # number of flux times
        corr = np.zeros((N, N))

        # Calculate and combine correlation matrices based on the method weights
        for method_name, weight in method.items():
            if method_name in ["exp", "diel"]:  # Closer times are more correlated
                if not isinstance(length_scale, (str, pd.Timedelta)):
                    raise ValueError(
                        'length_scale must be a str or pd.Timedelta for "exp" and "diel" methods'
                    )
                method_corr = time_decay_matrix(times, decay=length_scale)

                if method_name == "diel":
                    # Set the correlation values for the same hour of day
                    # use a NumPy array for safe ndarray-style indexing
                    hours = times.hour.values
                    same_time_mask = (hours[:, None] - hours[None, :]) == 0
                    method_corr[~same_time_mask] = 0

            elif (
                method_name == "clim"
            ):  # Each month is highly correlated with the same month in other years
                # Initialize the correlation matrix as identity matrix
                method_corr = np.eye(N)  # the diagonal

                # Set the correlation values for the same month in other years
                corr_val = 0.9
                months = times.month.values

                # Create a mask for the same month in different years
                same_month_mask = (months[:, None] - months[None, :]) % 12 == 0

                # Apply the correlation value using the mask
                method_corr[same_month_mask] = corr_val
            else:
                raise ValueError(f"Unknown method: {method_name}")

            # Combine the correlation matrices based on the method weights
            corr += weight * method_corr

        return corr


class ModelDataMismatch(SpaceTimeCovariance):
    """
    ModelDataMismatch: Model-Data Mismatch Covariance Matrix.

    Subclass of CovarianceMatrix for representing the model-data mismatch
    covariance matrix in flux inversion problems.
    """

    @staticmethod
    def _apply_groupwise_decay(
        matrix: np.ndarray,
        coords: pd.DataFrame,
        decay_func: Callable,
        decay_dims: list[str],
        **decay_kwargs,
    ) -> np.ndarray:
        """
        Generic method to apply decay (time or space) to the covariance matrix.

        Parameters:
        -----------
        matrix : np.ndarray
            Covariance matrix to which decay will be applied.
        coords : pd.DataFrame
            DataFrame containing the coordinates corresponding to the matrix indices.
        decay_func : Callable
            Function to compute the decay matrix.
        decay_dims : list[str]
            Dimension(s) on which to apply the decay (e.g., 'obs_time' for time decay).
        decay_kwargs : dict
            Additional keyword arguments for the decay function

        Returns:
        --------
        Self
            A new instance with the decay applied
        """
        diag_mask = np.eye(len(coords), dtype=bool)
        off_diags = np.zeros_like(matrix)

        # Find all dimensions other than the decay dimension
        group_dims = [col for col in coords.columns if col not in decay_dims]

        # If no other dimensions found, add a dummy dimension
        if not group_dims:
            coords["dummy"] = 0
            group_dims = ["dummy"]

        # Group by the identified dimensions
        groups = coords.groupby(group_dims)

        # Calculate the decay matrix for each group
        for _group_key, group in groups:
            group_indices = group.index.to_list()
            decay_matrix = decay_func(group, **decay_kwargs)
            off_diags[np.ix_(group_indices, group_indices)] = decay_matrix

        # Scale decay matrix by variances
        variances = np.diag(matrix)
        sigma = np.diag(np.sqrt(variances))
        off_diags = sigma @ off_diags @ sigma

        # Add the decayed off-diagonal terms to the original matrix
        mdm = matrix.copy()
        off_diags[diag_mask] = 0  # Zero out diagonal terms
        mdm += off_diags

        return mdm

    def apply_temporal_decay(
        self,
        length_scale: pd.Timedelta | str,
        span_days: bool = True,
        time_dim: str = "obs_time",
    ) -> Self:
        """
        Apply temporal decay to the covariance matrix.

        Parameters:
        -----------
        length_scale : pd.Timedelta or str
            Length scale for the temporal decay (e.g., '32h' for 32 hours)
        span_days : bool
            Whether to span across days
        time_dim : str
            Name of the time dimension

        Returns:
        --------
        Self
            A new instance with time decay applied
        """

        # Define the decay function
        def calculate_temporal_decay(
            group: pd.DataFrame, time_dim, length_scale
        ) -> np.ndarray:
            times = group[time_dim]
            return time_decay_matrix(times=times, decay=length_scale)

        # If not spanning days, add the date as a grouping dimension
        coords = self.index.to_frame(index=False)
        if not span_days:
            times = pd.to_datetime(coords[time_dim])
            coords["date_group"] = times.dt.date

        # Apply the decay
        matrix = self.data.to_numpy()
        decayed_matrix = self._apply_groupwise_decay(
            matrix=matrix,
            coords=coords,
            decay_func=calculate_temporal_decay,
            decay_dims=[time_dim],
            time_dim=time_dim,
            length_scale=length_scale,
        )

        return self.from_numpy(array=decayed_matrix, index=self.index)

    def apply_spatial_decay(
        self,
        length_scale: float,
        location_latlon_mapper: dict[str, tuple[float, float]],
        earth_radius: float = 6371.0,
        spatial_dim: str = "obs_location",
    ) -> Self:
        """
        Apply spatial decay to the covariance matrix.

        Parameters:
        -----------
        length_scale : float
            Length scale for the spatial decay (in km)
        location_latlon_mapper : dict
            Dictionary mapping location names to (latitude, longitude) tuples
        earth_radius : float
            Earth radius in km
        spatial_dim : str
            Name of the spatial dimension

        Returns:
        --------
        Self
            A new instance with spatial decay applied
        """

        # Define the decay function
        def calculate_spatial_decay(
            group: pd.DataFrame, length_scale: float, earth_radius: float
        ) -> np.ndarray:
            # Get lat/lon for each location in the group
            lats = group["lat"].to_numpy()
            lons = group["lon"].to_numpy()

            # Calculate distance matrix using haversine formula
            dist_matrix = haversine_matrix(
                lats=lats, lons=lons, earth_radius=earth_radius
            )

            # Calculate decay matrix
            return np.exp(-dist_matrix / length_scale)

        coords = self.index.to_frame(index=False)
        coords["lat"] = [location_latlon_mapper[loc][0] for loc in coords[spatial_dim]]
        coords["lon"] = [location_latlon_mapper[loc][1] for loc in coords[spatial_dim]]
        coords = coords.drop(columns=[spatial_dim])

        # Apply the decay
        matrix = self.data.to_numpy()
        decayed_matrix = self._apply_groupwise_decay(
            matrix=matrix,
            coords=coords,
            decay_func=calculate_spatial_decay,
            decay_dims=["lat", "lon"],
            length_scale=length_scale,
            earth_radius=earth_radius,
        )

        return self.from_numpy(array=decayed_matrix, index=self.index)

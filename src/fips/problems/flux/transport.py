import datetime as dt
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import stilt

from fips.core import ForwardOperator as Jacobian
from fips.problems.flux.utils import integrate_over_time_bins
from fips.utils import parallelize


class StiltJacobianBuilder:
    """
    Build the Jacobian matrix from STILT simulations.

    Attributes
    ----------
    simulations : list[stilt.Simulation | Path]
        List of STILT simulations or paths to simulation directories.
    failed_sims : list[str]
        List of simulation IDs that failed to process.

    Methods
    -------
    build_from_coords(coords, flux_times, resolution=None, subset_hours=None, num_processes=1)
        Build the Jacobian matrix H from specified coordinates (x, y) and flux time bins.
    """

    def __init__(self, simulations: list[stilt.Simulation | Path]):
        """
        Initialize the Jacobian builder with a list of STILT simulations
        or paths to simulation directories.

        Parameters
        ----------
        simulations : list[stilt.Simulation | Path]
            List of STILT simulations or paths to simulation directories.
        """
        self.simulations = simulations

        self.failed_sims = []

    def build_from_coords(
        self,
        coords: list[tuple[float, float]] | dict[str, list[tuple[float, float]]],
        flux_times: pd.IntervalIndex,
        resolution: str | None = None,
        subset_hours: int | list[int] | None = None,
        num_processes: int | Literal["max"] = 1,
        location_mapper: dict[str, str] | None = None,
    ) -> Jacobian | dict[str, Jacobian]:
        """
        Build the Jacobian matrix H from specified coordinates (x, y) and flux time bins.

        Parameters
        ----------
        coords : list[tuple[float, float]] | dict[str, list[tuple[float, float]]]
            Coordinates of the output grid points.
            Multiple sets of coordinates can be provided as a dictionary of lists of coordinate tuples.
            Otherwise, a single list of coordinate tuples can be provided.
            Coordinates should be in the same CRS as the STILT footprints and specified as (x, y) tuples.
        flux_times : pd.IntervalIndex
            Time bins for the fluxes
        resolution : str | None, optional
            Resolution of the footprints to use, by default None (use highest resolution available)
        subset_hours : int | list[int] | None, optional
            Subset the simulations to specific hours of the day, by default None
        num_processes : int | Literal['max'], optional
            Number of processes to use for parallel computation, by default 1
        location_mapper : dict[str, str] | None, optional
            Optional mapping of observation location IDs to new IDs.

        Returns
        -------
        Jacobian | dict[str, Jacobian]
            If coords is a dict, returns a dict of Jacobians for each set of coordinates.
            Otherwise, returns a single Jacobian.
        """

        print("Building Jacobian matrix...")

        if not isinstance(coords, dict):
            coords = {"DEFAULT": coords}

        # Build the Jacobian matrix in parallel
        H_rows = defaultdict(list)
        parallelized_builder = parallelize(
            self._build_jacobian_row_from_coords, num_processes=num_processes
        )
        results = parallelized_builder(
            self.simulations,
            coords=coords,
            flux_times=flux_times,
            resolution=resolution,
            subset_hours=subset_hours,
        )
        for row in results:
            if row is not None:
                if isinstance(row, dict):
                    for key, df in row.items():
                        H_rows[key].append(df)
                elif isinstance(row, str):
                    if row not in self.failed_sims:
                        self.failed_sims.append(row)
                else:
                    raise ValueError("Unexpected output from build_jacobian_row")

        H_dict = {}
        for key, rows in H_rows.items():
            if rows:
                print(f"Combining {len(rows)} rows for {key} jacobian...")
                H = pd.concat(rows).fillna(0)

                if location_mapper:
                    h = H.reset_index()
                    h["obs_location"] = (
                        h["obs_location"].map(location_mapper).fillna(h["obs_location"])
                    )
                    H = h.set_index(["obs_location", "obs_time"])

                H = Jacobian(H)
                H_dict[key] = H

                if key == "DEFAULT":
                    return H

        print("Jacobian matrix built successfully.")

        return H_dict

    @staticmethod
    def _build_jacobian_row_from_coords(
        simulation: stilt.Simulation | Path,
        coords: dict[str, list[tuple[float, float]]],
        flux_times: pd.IntervalIndex,
        resolution: str | None = None,
        subset_hours: int | list[int] | None = None,
    ) -> dict[str, pd.DataFrame] | str | None:
        """
        Build a row of the Jacobian matrix for a single STILT simulation
        """
        t_start, t_stop = flux_times[0].left, flux_times[-1].right

        # Get simulation object
        sim = StiltJacobianBuilder._get_sim(
            simulation=simulation,
            t_start=t_start,
            t_stop=t_stop,
            subset_hours=subset_hours,
        )
        if not isinstance(sim, stilt.Simulation):
            return sim  # could be None or sim.id if failed

        # Get footprint for the simulation
        footprint = StiltJacobianBuilder._get_footprint(
            sim=sim, t_start=t_start, t_stop=t_stop, resolution=resolution
        )
        if footprint is None:
            return None

        print(f"Computing Jacobian row for {sim.id}...")

        # Convert xarray to pandas for sparse representation
        foot = footprint.data.to_series()

        # Get the x and y dimension names
        is_latlon = "lon" in foot.index.names and "lat" in foot.index.names
        x_dim = "lon" if is_latlon else "x"
        y_dim = "lat" if is_latlon else "y"

        foot = foot.reset_index()

        # Round coordinates to avoid floating point issues
        xres, yres = footprint.xres, footprint.yres
        xdigits = StiltJacobianBuilder._calc_digits(xres)
        ydigits = StiltJacobianBuilder._calc_digits(yres)

        foot[x_dim] = foot[x_dim].round(xdigits)
        foot[y_dim] = foot[y_dim].round(ydigits)

        # Reorder dimensions to x, y, time
        foot = foot.set_index([x_dim, y_dim, "time"])  # still a df

        # Build index value for the observation
        obs_index = StiltJacobianBuilder._build_obs_index(sim=sim)

        # Build Jacobian row for each set of coordinates
        rows = {}
        for key, coord_list in coords.items():
            # Round input coordinates to match footprint rounding
            coord_index = pd.MultiIndex.from_tuples(coord_list)
            coord_index = coord_index.round([xdigits, ydigits]).set_names(
                [x_dim, y_dim]
            )

            # Filter to xy points defined by grid
            filtered_foot = (
                foot.reset_index(level="time")
                .loc[coord_index]
                .set_index("time", append=True)
            )

            if filtered_foot.size == 0:
                continue

            # Integrate each simulation over flux_time bins
            integrated_foot = integrate_over_time_bins(
                data=filtered_foot, time_bins=flux_times
            )

            # Transpose and set index as (obs_location, obs_time) multiindex
            transposed_foot = integrated_foot.T
            transposed_foot.index = obs_index

            rows[key] = transposed_foot
        return rows

    @staticmethod
    def _get_sim(
        simulation: stilt.Simulation | Path,
        t_start: dt.datetime,
        t_stop: dt.datetime,
        subset_hours: int | list[int] | None = None,
    ) -> stilt.Simulation | str | None:
        sim = StiltJacobianBuilder._load_simulation(simulation)

        if sim.status != "SUCCESS":
            return sim.id

        if not StiltJacobianBuilder._sim_in_time_range(
            sim=sim, t_start=t_start, t_stop=t_stop, subset_hours=subset_hours
        ):
            return None
        return sim

    @staticmethod
    def _load_simulation(simulation: stilt.Simulation | Path) -> stilt.Simulation:
        if isinstance(simulation, Path):
            sim = stilt.Simulation.from_path(simulation)
        elif isinstance(simulation, stilt.Simulation):
            sim = simulation
        else:
            raise ValueError("simulation must be a Path or a stilt.Simulation object")
        return sim

    @staticmethod
    def _sim_in_time_range(
        sim: stilt.Simulation,
        t_start: dt.datetime,
        t_stop: dt.datetime,
        subset_hours: int | list[int] | None = None,
    ) -> bool:
        n_hours = sim.config.n_hours
        if n_hours >= 0:
            raise ValueError("STILT must be run backwards in time (n_hours < 0)")
        n_hours = pd.to_timedelta(n_hours, unit="h")

        # Skip simulations that do not overlap with inversion time range
        sim_start = sim.receptor.time + n_hours
        sim_end = sim.receptor.time - pd.Timedelta(hours=1)
        if sim_end < t_start or sim_start >= t_stop:
            return False

        if subset_hours:
            # Subset simulations to specific hours
            if isinstance(subset_hours, int):
                subset_hours = [subset_hours]

            if sim.receptor.time.hour not in subset_hours:
                return False

        return True

    @staticmethod
    def _get_footprint(
        sim: stilt.Simulation,
        t_start: dt.datetime,
        t_stop: dt.datetime,
        resolution: str | None = None,
    ) -> stilt.Footprint | None:
        # Load footprint object
        footprint = StiltJacobianBuilder._load_footprint(sim=sim, resolution=resolution)

        if not footprint:
            return None

        # Check if footprint time range overlaps with inversion time range
        if not StiltJacobianBuilder._footprint_in_time_range(
            footprint=footprint, t_start=t_start, t_stop=t_stop
        ):
            # A footprint within a simulation might not overlap with the inversion time range
            # even if the simulation does
            return None

        return footprint

    @staticmethod
    def _load_footprint(
        sim: stilt.Simulation, resolution: str | None = None
    ) -> stilt.Footprint | None:
        # Load footprint at specified resolution if available
        # Otherwise, get the default (highest) resolution footprint
        footprint = sim.footprints[resolution] if resolution else sim.footprint
        return footprint

    @staticmethod
    def _footprint_in_time_range(
        footprint: stilt.Footprint, t_start: dt.datetime, t_stop: dt.datetime
    ) -> bool:
        # Check if footprint time range overlaps with inversion time range
        return footprint.time_range[0] < t_stop and footprint.time_range[1] > t_start

    @staticmethod
    def _build_obs_index(sim: stilt.Simulation) -> pd.MultiIndex:
        return pd.MultiIndex.from_arrays(
            [[sim.receptor.location.id], [sim.receptor.time]],
            names=["obs_location", "obs_time"],
        )

    @staticmethod
    def _calc_digits(res: float) -> int:
        if res <= 0:
            raise ValueError("Resolution must be positive")
        if res < 1:  # fractional resolution
            digits = int(np.ceil(np.abs(np.log10(res)))) + 1
        else:
            digits = int(-np.log10(res))
        return digits

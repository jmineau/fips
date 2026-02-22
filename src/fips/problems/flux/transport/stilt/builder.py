import datetime as dt
import logging
from collections import defaultdict
from pathlib import Path
from typing import overload

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from stilt import Simulation

from fips.aggregators import integrate_over_time_bins
from fips.matrix import MatrixBlock
from fips.problems.flux.transport.stilt.footprint import get_footprint
from fips.problems.flux.transport.stilt.simulation import get_sim

logger = logging.getLogger(__name__)


class JacobianBuilder:
    def __init__(
        self,
        simulations: list[Simulation | Path],
        location_dim: str = "obs_location",
        time_dim: str = "obs_time",
    ):
        """
        Initialize the Jacobian builder with a list of STILT simulations
        or paths to simulation directories.

        Parameters
        ----------
        simulations : list[stilt.Simulation | Path]
            List of STILT simulations or paths to simulation directories.
        """
        self.simulations = simulations
        self.location_dim = location_dim
        self.time_dim = time_dim

        self.failed_sims = []

    @overload
    def build_from_coords(
        self,
        coords: list[tuple[float, float]],
        flux_times: pd.IntervalIndex,
        resolution: str | None = None,
        subset_hours: int | list[int] | None = None,
        num_processes: int = 1,
        location_mapper: dict[str, str] | None = None,
    ) -> MatrixBlock: ...

    @overload
    def build_from_coords(
        self,
        coords: dict[str, list[tuple[float, float]]],
        flux_times: pd.IntervalIndex,
        resolution: str | None = None,
        subset_hours: int | list[int] | None = None,
        num_processes: int = 1,
        location_mapper: dict[str, str] | None = None,
    ) -> dict[str, MatrixBlock]: ...

    def build_from_coords(
        self,
        coords: list[tuple[float, float]] | dict[str, list[tuple[float, float]]],
        flux_times: pd.IntervalIndex,
        resolution: str | None = None,
        subset_hours: int | list[int] | None = None,
        num_processes: int = 1,
        location_mapper: dict[str, str] | None = None,
        timeout: float | int | None = None,
    ) -> MatrixBlock | dict[str, MatrixBlock]:
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
        num_processes : int, optional
            Number of processes to use for parallel computation, by default 1
            To use all cores, set num_processes=-1. Note that parallel processing may not be available on all platforms.
        location_mapper : dict[str, str] | None, optional
            Optional mapping of observation location IDs to new IDs.
        timeout : float, optional
            The maximum time (in seconds) allowed for each simulation to be processed.
            If a task exceeds this time, a TimeoutError is raised.
            Default is None (no timeout).

        Returns
        -------
        MatrixBlock | dict[str, MatrixBlock]
            If coords is a dict, returns a dict of MatrixBlocks for each set of coordinates.
            Otherwise, returns a single MatrixBlock.
        """

        logger.info("Building Jacobian matrix...")

        if not isinstance(coords, dict):
            coords = {"DEFAULT": coords}

        # Determine overall time range from flux_times
        t_start, t_stop = flux_times[0].left, flux_times[-1].right

        # Build the Jacobian matrix in parallel
        H_rows = defaultdict(list)
        if num_processes > 1:
            logger.debug(f"Building Jacobian rows with {num_processes} processes...")
        results = Parallel(n_jobs=num_processes, timeout=timeout)(
            delayed(build_jacobian_row_from_coords)(
                sim,
                coords=coords,
                location_dim=self.location_dim,
                time_dim=self.time_dim,
                flux_times=flux_times,
                t_start=t_start,
                t_stop=t_stop,
                resolution=resolution,
                subset_hours=subset_hours,
            )
            for sim in self.simulations
        )
        logger.debug("Sorting Jacobian rows...")
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
                rows: list[pd.DataFrame]
                logger.debug(f"Combining {len(rows)} rows for {key} jacobian...")
                H = pd.concat(rows).fillna(0)

                if location_mapper:
                    h = H.reset_index()
                    h[self.location_dim] = (
                        h[self.location_dim]
                        .map(location_mapper)
                        .fillna(h[self.location_dim])
                    )
                    H = h.set_index([self.location_dim, self.time_dim])

                H = MatrixBlock(
                    H, name="jacobian", row_block="concentration", col_block="flux"
                )
                H_dict[key] = H

                if key == "DEFAULT":
                    return H

        logger.info("Jacobian matrix built successfully.")

        return H_dict


def build_jacobian_row_from_coords(  # must be top-level for multiprocessing
    simulation: Simulation | Path,
    coords: dict[str, list[tuple[float, float]]],
    location_dim: str,
    time_dim: str,
    flux_times: pd.IntervalIndex,
    t_start: dt.datetime,
    t_stop: dt.datetime,
    resolution: str | None = None,
    subset_hours: int | list[int] | None = None,
) -> dict[str, pd.DataFrame] | str | None:
    """
    Build a row of the Jacobian matrix for a single STILT simulation
    """
    t0 = dt.datetime.now()

    # Get simulation object
    sim = get_sim(
        simulation=simulation, t_start=t_start, t_stop=t_stop, subset_hours=subset_hours
    )
    if not isinstance(sim, Simulation):
        return sim  # could be None or sim.id if failed

    # Get footprint for the simulation
    try:
        footprint = get_footprint(
            sim=sim, t_start=t_start, t_stop=t_stop, resolution=resolution
        )
    except Exception as e:
        foot_file = sim.paths["footprints"][str(resolution)]
        logger.exception(f"Error loading footprint file {foot_file}")
        raise e

    if footprint is None:
        return None

    # print(f"Computing Jacobian row for {sim.id}...")

    # Convert xarray to pandas for sparse representation
    foot = footprint.data.to_series()

    # Get the x and y dimension names
    is_latlon = "lon" in foot.index.names and "lat" in foot.index.names
    x_dim = "lon" if is_latlon else "x"
    y_dim = "lat" if is_latlon else "y"

    foot = foot.reset_index()

    # Round coordinates to avoid floating point issues
    xres, yres = footprint.xres, footprint.yres
    xdigits = calc_digits(xres)
    ydigits = calc_digits(yres)

    foot[x_dim] = foot[x_dim].round(xdigits)
    foot[y_dim] = foot[y_dim].round(ydigits)

    # Reorder dimensions to x, y, time
    foot = foot.set_index([x_dim, y_dim, "time"])  # still a df

    # Build index value for the observation
    obs_index = build_obs_index(sim=sim, location_dim=location_dim, time_dim=time_dim)

    # Build Jacobian row for each set of coordinates
    rows = {}
    for key, coord_list in coords.items():
        # Round input coordinates to match footprint rounding
        coord_index = pd.MultiIndex.from_tuples(coord_list)
        coord_index = coord_index.round([xdigits, ydigits]).set_names([x_dim, y_dim])

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
    logger.debug(
        "Finished computing Jacobian row for %s in %s",
        sim.id,
        dt.datetime.now() - t0,
    )
    return rows


def build_obs_index(sim: Simulation, location_dim: str, time_dim: str) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays(
        [[sim.receptor.location.id], [sim.receptor.time]],
        names=[location_dim, time_dim],
    )


def calc_digits(res: float) -> int:
    if res <= 0:
        raise ValueError("Resolution must be positive")
    if res < 1:  # fractional resolution
        digits = int(np.ceil(np.abs(np.log10(res)))) + 1
    else:
        digits = int(-np.log10(res))
    return digits

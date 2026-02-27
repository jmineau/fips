"""
Jacobian builder for STILT transport models.

This module provides the `JacobianBuilder` class, which constructs the
forward operator (Jacobian matrix) by loading and aggregating STILT
footprints over specified time bins and spatial resolutions.
"""

import datetime as dt
import logging
from collections import defaultdict
from pathlib import Path
from typing import overload

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from stilt import Simulation  # type: ignore[import]

from fips.aggregators import integrate_over_time_bins
from fips.matrix import MatrixBlock
from fips.problems.flux.transport.stilt.footprint import get_footprint
from fips.problems.flux.transport.stilt.simulation import get_sim

logger = logging.getLogger(__name__)


class JacobianBuilder:
    """
    Builds Jacobian matrices from STILT footprint simulations.

    Parameters
    ----------
    simulations : list[stilt.Simulation | Path]
        List of STILT simulations or paths to simulation directories.
    """

    simulations: list[Simulation | Path]
    """List of STILT simulations or paths to simulation directories."""
    location_dim: str
    """Name of the observation location dimension."""
    time_dim: str
    """Name of the observation time dimension."""
    failed_sims: list[str]
    """List of simulation IDs that failed to process."""

    def __init__(
        self,
        simulations: list[Simulation | Path],
        location_dim: str = "obs_location",
        time_dim: str = "obs_time",
    ):
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
        timeout: float | int | None = None,
        threshold: float | None = 1e-10,
        sparse: bool = False,
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
        timeout: float | int | None = None,
        threshold: float | None = 1e-10,
        sparse: bool = False,
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
        threshold: float | None = 1e-15,  # numpy.float64 precision is 1e-15
        sparse: bool = False,
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
        threshold : float | None, optional
            Values whose absolute value is strictly less than this threshold are set to
            zero before the MatrixBlock is assembled.  This avoids storing floating-point
            noise as explicit non-zero entries.  Default is 1e-15.  Pass None to disable.
        sparse : bool, default False
            If True, store the assembled MatrixBlock in pandas sparse format.
            Pairs naturally with ``threshold`` — threshold zeroing is applied first
            so that structural zeros are not stored as explicit entries.

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

                if threshold is not None:
                    H = H.where(H.abs() >= threshold, other=0.0)

                if location_mapper:
                    h = H.reset_index()
                    h[self.location_dim] = (
                        h[self.location_dim]
                        .map(location_mapper)
                        .fillna(h[self.location_dim])
                    )
                    H = h.set_index([self.location_dim, self.time_dim])

                H = MatrixBlock(
                    H,
                    name="jacobian",
                    row_block="concentration",
                    col_block="flux",
                    sparse=sparse,
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
    Build a row of the Jacobian matrix for a single STILT simulation.

    Parameters
    ----------
    simulation : Simulation or Path
        STILT simulation object or path to simulation directory.
    coords : dict[str, list[tuple[float, float]]]
        Dictionary mapping coordinate set names to lists of (x, y) coordinate tuples.
    location_dim : str
        Name of the observation location dimension.
    time_dim : str
        Name of the observation time dimension.
    flux_times : pd.IntervalIndex
        Time bins for flux aggregation.
    t_start : datetime
        Start of the inversion time range.
    t_stop : datetime
        End of the inversion time range.
    resolution : str, optional
        Resolution of the footprint to use. If None, uses default (highest) resolution.
    subset_hours : int or list[int], optional
        Hour(s) of day to filter simulations. If None, uses all hours.

    Returns
    -------
    dict[str, pd.DataFrame] or str or None
        Dictionary mapping coordinate set names to Jacobian row DataFrames,
        or simulation ID string if failed, or None if skipped.
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
    """
    Build observation index from simulation receptor location and time.

    Parameters
    ----------
    sim : Simulation
        STILT simulation object.
    location_dim : str
        Name of the location dimension.
    time_dim : str
        Name of the time dimension.

    Returns
    -------
    pd.MultiIndex
        Multi-index with location and time levels.
    """
    return pd.MultiIndex.from_arrays(
        [[sim.receptor.location.id], [sim.receptor.time]],
        names=[location_dim, time_dim],
    )


def calc_digits(res: float) -> int:
    """
    Calculate number of decimal digits to represent a resolution value.

    Parameters
    ----------
    res : float
        Resolution value (must be positive).

    Returns
    -------
    int
        Number of decimal digits needed to represent the resolution.
    """
    if res <= 0:
        raise ValueError("Resolution must be positive")
    if res < 1:  # fractional resolution  # noqa: SIM108
        digits = int(np.ceil(np.abs(np.log10(res)))) + 1
    else:
        digits = int(-np.log10(res))
    return digits

"""Main STILT Jacobian builder class.

This module provides the JacobianBuilder class for constructing sensitivity matrices
(Forward Operators) from STILT atmospheric transport simulations.
"""

import datetime as dt
from collections import defaultdict
from pathlib import Path
from typing import Literal, overload

import pandas as pd
import stilt

from fips.matrices import ForwardOperator as Jacobian
from fips.parallel import parallelize
from fips.problems.flux.transport.footprint import build_jacobian_row
from fips.problems.flux.transport.simulation import get_simulation


class JacobianBuilder:
    """
    Build the Jacobian matrix from STILT simulations.

    The Jacobian (also called Forward Operator or sensitivity matrix) maps
    fluxes to concentration observations using atmospheric transport model
    outputs (STILT footprints).

    Attributes
    ----------
    simulations : list[stilt.Simulation | Path]
        List of STILT simulations or paths to simulation directories.
    failed_sims : list[str]
        List of simulation IDs that failed to process.

    Methods
    -------
    build_from_coords(coords, flux_times, resolution=None, subset_hours=None, num_processes=1)
        Build the Jacobian matrix from specified coordinates and flux time bins.
    """

    def __init__(self, simulations: list[stilt.Simulation | Path]):
        """
        Initialize the Jacobian builder with STILT simulations.

        Parameters
        ----------
        simulations : list[stilt.Simulation | Path]
            List of STILT simulations or paths to simulation directories.
        """
        self.simulations = simulations
        self.failed_sims = []

    @overload
    def build_from_coords(
        self,
        coords: list[tuple[float, float]],
        flux_times: pd.IntervalIndex,
        resolution: str | None = None,
        subset_hours: int | list[int] | None = None,
        num_processes: int | Literal["max"] = 1,
        location_mapper: dict[str, str] | None = None,
    ) -> Jacobian: ...

    @overload
    def build_from_coords(
        self,
        coords: dict[str, list[tuple[float, float]]],
        flux_times: pd.IntervalIndex,
        resolution: str | None = None,
        subset_hours: int | list[int] | None = None,
        num_processes: int | Literal["max"] = 1,
        location_mapper: dict[str, str] | None = None,
    ) -> dict[str, Jacobian]: ...

    def build_from_coords(
        self,
        coords: list[tuple[float, float]] | dict[str, list[tuple[float, float]]],
        flux_times: pd.IntervalIndex,
        resolution: str | None = None,
        subset_hours: int | list[int] | None = None,
        num_processes: int | Literal["max"] = 1,
        location_mapper: dict[str, str] | None = None,
        timeout: float | int | None = None,
    ) -> Jacobian | dict[str, Jacobian]:
        """
        Build the Jacobian matrix from specified coordinates and flux time bins.

        This method processes STILT simulations in parallel to construct the
        Jacobian matrix (sensitivity matrix) that maps flux state variables
        to concentration observations. Each simulation contributes one row
        to the Jacobian.

        Parameters
        ----------
        coords : list[tuple[float, float]] | dict[str, list[tuple[float, float]]]
            Coordinates of the output grid points as (x, y) tuples.
            - If list: single set of coordinates, returns single Jacobian
            - If dict: multiple domains (e.g., {'near': [...], 'far': [...]}),
              returns dict of Jacobians
            Coordinates should be in the same CRS as STILT footprints.
        flux_times : pd.IntervalIndex
            Time bins for the fluxes (defines temporal resolution of state vector).
        resolution : str | None, optional
            Resolution of footprints to use (e.g., '0.01deg').
            If None, uses highest resolution available. Default is None.
        subset_hours : int | list[int] | None, optional
            Subset simulations to specific hours of the day (0-23).
            If int, converts to single-element list. Default is None (all hours).
        num_processes : int | Literal['max'], optional
            Number of processes for parallel computation. If 'max', uses all
            available CPUs. Default is 1 (sequential).
        location_mapper : dict[str, str] | None, optional
            Optional mapping of observation location IDs to new IDs.
            Useful for grouping or renaming locations.
        timeout : float, optional
            Maximum time (in seconds) allowed for each simulation to be processed.
            If exceeded, raises TimeoutError. Default is None (no timeout).

        Returns
        -------
        Jacobian | dict[str, Jacobian]
            - If coords is a list: returns single Jacobian
            - If coords is a dict: returns dict mapping domain names to Jacobians

        Notes
        -----
        The Jacobian is built row-by-row, where each row corresponds to one
        observation (STILT simulation). Failed simulations are tracked in
        self.failed_sims.
        """

        print("Building Jacobian matrix...")

        if not isinstance(coords, dict):
            coords = {"DEFAULT": coords}

        # Pre-process simulations: load and filter by time/hour
        t_start, t_stop = flux_times[0].left, flux_times[-1].right
        processed_sims = [
            get_simulation(sim, t_start, t_stop, subset_hours)
            for sim in self.simulations
        ]

        # Build the Jacobian matrix in parallel
        H_rows = defaultdict(list)
        parallelized_builder = parallelize(
            build_jacobian_row,
            num_processes=num_processes,
            timeout=timeout,
        )
        results = parallelized_builder(
            processed_sims,
            coords=coords,
            flux_times=flux_times,
            resolution=resolution,
        )

        print("Sorting Jacobian rows...")
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


__all__ = ["JacobianBuilder"]

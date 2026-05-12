"""
Jacobian builder for STILT transport models.

This module provides the `JacobianBuilder` class, which constructs the
forward operator (Jacobian matrix) by loading and aggregating STILT
footprints over specified time bins and spatial resolutions.
"""

import logging
from collections import defaultdict

import pandas as pd
from joblib import Parallel, delayed
from stilt.footprint import Footprint  # type: ignore[import]
from stilt.model import Model  # type: ignore[import]

from fips.matrix import MatrixBlock

logger = logging.getLogger(__name__)


class JacobianBuilder:
    """
    Builds Jacobian matrices from STILT footprints via a PYSTILT Model.

    Parameters
    ----------
    model : stilt.Model
        A configured PYSTILT Model used to load footprints via
        ``model.get_footprints()``.
    location_dim : str
        Name of the observation location dimension.
    time_dim : str
        Name of the observation time dimension.
    """

    location_dim: str
    time_dim: str
    failed_sims: list[str]

    def __init__(
        self,
        model: Model,
        location_dim: str = "obs_location",
        time_dim: str = "obs_time",
    ):
        self.model = model
        self.location_dim = location_dim
        self.time_dim = time_dim
        self.failed_sims = []

    def build_from_coords(
        self,
        coords: list[tuple[float, float]] | dict[str, list[tuple[float, float]]],
        flux_times: pd.IntervalIndex,
        footprint: str,
        *,
        mets: str | list[str] | None = None,
        time_range: tuple | None = None,
        location_ids: set[str] | None = None,
        subset_hours: int | list[int] | None = None,
        num_processes: int = 1,
        location_mapper: dict[str, str] | None = None,
        timeout: float | int | None = None,
        threshold: float | None = 1e-15,
        sparse: bool = False,
    ) -> "MatrixBlock | dict[str, MatrixBlock]":
        """
        Build the Jacobian matrix H from specified coordinates and flux time bins.

        Parameters
        ----------
        coords : list[tuple[float, float]] | dict[str, list[tuple[float, float]]]
            Coordinates of the output grid points as (x, y) tuples. Pass a
            dict to build multiple Jacobians over different coordinate sets.
        flux_times : pd.IntervalIndex
            Time bins for the fluxes.
        footprint : str
            Name of the footprint to load from each simulation.
        mets : str, list[str], or None
            Restrict to specific met configurations. None = all.
        time_range : tuple or None
            ``(start, end)`` to filter simulations by receptor time. Defaults
            to the full flux window derived from ``flux_times``.
        location_ids : set[str] or None
            Restrict to specific location IDs.
        subset_hours : int | list[int] | None
            Filter simulations to specific hours of the day (receptor time).
        num_processes : int
            Number of parallel workers (joblib). -1 = all cores.
        location_mapper : dict[str, str] | None
            Optional mapping of location IDs to new IDs (e.g. site names).
        timeout : float | None
            Per-task timeout in seconds passed to joblib.
        threshold : float | None
            Absolute value cutoff; entries below this are zeroed. Default
            1e-15. Pass None to disable.
        sparse : bool
            Store the assembled MatrixBlock in sparse format.

        Returns
        -------
        MatrixBlock | dict[str, MatrixBlock]
            Single MatrixBlock when coords is a list; dict when coords is a dict.
        """
        logger.info("Building Jacobian matrix...")

        if not isinstance(coords, dict):
            coords = {"DEFAULT": coords}

        if time_range is None:
            time_range = (flux_times[0].left, flux_times[-1].right)

        footprints = self.model.footprints[footprint].load(
            mets=mets,
            time_range=time_range,
            location_ids=location_ids,
        )

        if subset_hours is not None:
            if isinstance(subset_hours, int):
                subset_hours = [subset_hours]
            footprints = [
                fp for fp in footprints
                if fp.receptor.time.hour in subset_hours
            ]

        if not footprints:
            raise ValueError(
                f"No footprints found for '{footprint}' after filtering. "
                "Check that footprints exist and filters are not too restrictive."
            )

        logger.debug("Dispatching %d footprints...", len(footprints))
        results = Parallel(n_jobs=num_processes, timeout=timeout)(
            delayed(build_jacobian_row_from_coords)(
                fp=fp,
                coords=coords,
                location_dim=self.location_dim,
                time_dim=self.time_dim,
                flux_times=flux_times,
            )
            for fp in footprints
        )

        H_rows: dict[str, list[pd.DataFrame]] = defaultdict(list)
        for row in results:
            if row is not None:
                for key, df in row.items():
                    H_rows[key].append(df)

        if not H_rows:
            raise ValueError(
                f"No Jacobian rows were produced from {len(footprints)} footprints. "
                "Check that footprints overlap with the given coordinates."
            )

        H_dict: dict[str, MatrixBlock] = {}
        for key, rows in H_rows.items():
            H = pd.concat(rows).fillna(0)

            if threshold is not None:
                H = H.where(H.abs() >= threshold, other=0.0)

            if location_mapper:
                idx = H.index.to_frame(index=False)
                idx[self.location_dim] = (
                    idx[self.location_dim]
                    .map(location_mapper)
                    .fillna(idx[self.location_dim])
                )
                H.index = pd.MultiIndex.from_frame(idx)

            H = MatrixBlock(
                H,
                name="jacobian",
                row_block="concentration",
                col_block="flux",
                sparse=sparse,
            )
            H_dict[key] = H

            if key == "DEFAULT":
                logger.info("Jacobian matrix built successfully.")
                return H

        logger.info("Jacobian matrix built successfully.")
        return H_dict


def build_jacobian_row_from_coords(  # must be top-level for multiprocessing
    fp: Footprint,
    coords: dict[str, list[tuple[float, float]]],
    location_dim: str,
    time_dim: str,
    flux_times: pd.IntervalIndex,
) -> "dict[str, pd.DataFrame] | None":
    """
    Build a row of the Jacobian matrix for a single STILT footprint.

    Parameters
    ----------
    fp : stilt.Footprint
        Loaded STILT footprint.
    coords : dict[str, list[tuple[float, float]]]
        Coordinate sets to aggregate over; keys become keys in the output dict.
    location_dim : str
        Name of the observation location index level.
    time_dim : str
        Name of the observation time index level.
    flux_times : pd.IntervalIndex
        Flux time bins for aggregation.

    Returns
    -------
    dict[str, pd.DataFrame] or None
        Jacobian row DataFrames keyed by coordinate set name, or None if the
        footprint has no overlap with any coordinate set.
    """
    obs_index = pd.MultiIndex.from_arrays(
        [[fp.receptor.location_id], [fp.receptor.time]],
        names=[location_dim, time_dim],
    )

    rows: dict[str, pd.DataFrame] = {}
    for key, coord_list in coords.items():
        agg = fp.aggregate(coord_list, flux_times)
        if not agg.values.any():
            continue

        row = agg.stack().to_frame().T
        row.index = obs_index
        rows[key] = row

    return rows or None

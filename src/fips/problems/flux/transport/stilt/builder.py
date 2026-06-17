"""
Jacobian builder for STILT transport models.

This module provides the `JacobianBuilder` class, which constructs the
forward operator (Jacobian matrix) by loading and aggregating STILT
footprints over specified time bins and spatial resolutions.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from joblib import Parallel, delayed
from stilt.footprint import Footprint  # type: ignore[import]
from stilt.model import Model  # type: ignore[import]

from fips.matrix import MatrixBlock

if TYPE_CHECKING:
    import xarray as xr

    # An aggregation target: an (x, y) coords list or an xarray grid.
    Target = list[tuple[float, float]] | xr.DataArray | xr.Dataset

logger = logging.getLogger(__name__)


def _hour_from_sim_id(sim_id: str) -> int | None:
    """Extract receptor hour (UTC) from a PYSTILT sim_id: '{met}_{YYYYMMDDHHMM}_{loc}'."""
    parts = sim_id.split("_", 2)
    if len(parts) < 2 or len(parts[1]) < 10:
        return None
    try:
        return int(parts[1][8:10])
    except ValueError:
        return None


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
        **kwargs,
    ) -> MatrixBlock | dict[str, MatrixBlock]:
        """
        Build the Jacobian H from output-grid coordinates and flux time bins.

        Convenience wrapper over :meth:`build_from_grid` for callers that have a
        plain list of ``(x, y)`` cell centers (a regular grid is assumed; PYSTILT
        infers the cell size from the coordinate spacing). Prefer
        :meth:`build_from_grid` when you already have an xarray grid, since it
        carries resolution, bounds, CRS, and any mask explicitly.

        Parameters
        ----------
        coords : list[tuple[float, float]] | dict[str, list[tuple[float, float]]]
            Output grid cell centers as (x, y) tuples. Pass a dict to build
            multiple Jacobians over different coordinate sets.
        flux_times : pd.IntervalIndex
            Time bins for the fluxes.
        footprint : str
            Name of the footprint to load from each simulation.
        **kwargs
            Forwarded to :meth:`build_from_grid` (``mets``, ``time_range``,
            ``location_ids``, ``subset_hours``, ``num_processes``,
            ``location_mapper``, ``timeout``, ``threshold``, ``sparse``).

        Returns
        -------
        MatrixBlock | dict[str, MatrixBlock]
            Single MatrixBlock when coords is a list; dict when coords is a dict.
        """
        return self.build_from_grid(coords, flux_times, footprint, **kwargs)

    def build_from_grid(
        self,
        grid: Target | dict[str, Target],
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
    ) -> MatrixBlock | dict[str, MatrixBlock]:
        """
        Build the Jacobian matrix H over a target grid and flux time bins.

        Each footprint is conservatively regridded onto ``grid`` (see
        :meth:`stilt.Footprint.aggregate`) and its time-binned sensitivities
        become one Jacobian row.

        Parameters
        ----------
        grid : xr.DataArray | xr.Dataset | list[tuple[float, float]] | dict
            The target grid: a CF xarray grid (``lon``/``lat`` or ``x``/``y``
            coordinates; ``NaN`` cells in a 2-D DataArray are masked out) or a
            plain list of ``(x, y)`` cell centers. Pass a dict to build multiple
            Jacobians over different grids.
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
            Single MatrixBlock when ``grid`` is a single grid; dict when a dict.
        """
        logger.info("Building Jacobian matrix...")

        targets = grid if isinstance(grid, dict) else {"DEFAULT": grid}

        if time_range is None:
            time_range = (flux_times[0].left, flux_times[-1].right)

        # Get paths without loading — footprints are loaded inside each worker
        # to avoid serial NFS I/O and large object pickling overhead.
        paths = self.model.footprints[footprint].paths(
            mets=mets,
            time_range=time_range,
            location_ids=location_ids,
        )

        # Pre-filter by hour using sim_id before dispatching workers
        if subset_hours is not None:
            if isinstance(subset_hours, int):
                subset_hours = [subset_hours]
            hours_set = set(subset_hours)
            paths = [p for p in paths if _hour_from_sim_id(p.parent.name) in hours_set]

        if not paths:
            raise ValueError(
                f"No footprints found for '{footprint}' after filtering. "
                "Check that footprints exist and filters are not too restrictive."
            )

        logger.debug("Dispatching %d footprints...", len(paths))
        results = Parallel(n_jobs=num_processes, timeout=timeout)(
            delayed(_build_jacobian_row_from_path)(
                path=path,
                targets=targets,
                location_dim=self.location_dim,
                time_dim=self.time_dim,
                flux_times=flux_times,
            )
            for path in paths
        )

        H_rows: dict[str, list[pd.DataFrame]] = defaultdict(list)
        for row in results:
            if row is not None:
                for key, df in row.items():
                    H_rows[key].append(df)

        if not H_rows:
            raise ValueError(
                f"No Jacobian rows were produced from {len(paths)} footprints. "
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
                    .map(location_mapper.get)
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


def _build_jacobian_row(
    fp: Footprint,
    targets: dict[str, Target],
    location_dim: str,
    time_dim: str,
    flux_times: pd.IntervalIndex,
) -> dict[str, pd.DataFrame] | None:
    """
    Build one footprint's Jacobian row by aggregating it onto each target.

    Returns ``None`` when the footprint does not overlap any target (all-zero
    aggregate), so it contributes no row.
    """
    obs_index = pd.MultiIndex.from_arrays(
        [[fp.receptor.location_id], [fp.receptor.time]],
        names=[location_dim, time_dim],
    )

    rows: dict[str, pd.DataFrame] = {}
    for key, target in targets.items():
        agg = fp.aggregate(target, flux_times)
        if not agg.values.any():
            continue

        if agg.columns.name is None:
            agg.columns.name = "time"
        row = agg.stack().to_frame().T
        row.index = obs_index
        rows[key] = row

    return rows or None


def _build_jacobian_row_from_path(  # must be top-level for multiprocessing
    path: Path,
    targets: dict[str, Target],
    location_dim: str,
    time_dim: str,
    flux_times: pd.IntervalIndex,
) -> dict[str, pd.DataFrame] | None:
    """
    Load one footprint from disk and build its Jacobian row.

    Loading inside the worker avoids serial NFS reads and large object
    pickling that would occur if footprints were pre-loaded before dispatch.
    """
    try:
        fp = Footprint.from_netcdf(path)
    except Exception:
        return None

    return _build_jacobian_row(fp, targets, location_dim, time_dim, flux_times)

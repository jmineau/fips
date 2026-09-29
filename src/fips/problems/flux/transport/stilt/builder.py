"""
Jacobian builder for STILT transport models.

This module provides the `JacobianBuilder` class, which constructs the
forward operator (Jacobian matrix) by loading and aggregating STILT
footprints over specified time bins and spatial resolutions.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pandas as pd
from joblib import Parallel, delayed
from stilt.config import FootprintConfig  # type: ignore[import]
from stilt.footprint import Footprint  # type: ignore[import]
from stilt.model import Model  # type: ignore[import]
from stilt.trajectory import Trajectories  # type: ignore[import]

from fips.matrix import MatrixBlock

if TYPE_CHECKING:
    # The state geometry a footprint is aggregated onto: stilt.Grid,
    # stilt.Mesh, stilt.Zones, an xarray grid, or an (x, y) coords list.
    from stilt.geometry import SpatialTarget as Target  # type: ignore[import]

logger = logging.getLogger(__name__)


class JacobianBuilder:
    """
    Builds Jacobian matrices from STILT footprints via a PYSTILT Model.

    Each Jacobian row is one simulation of the model: one receptor under one
    variant (a named set of PYSTILT settings, each with at most one footprint).

    Parameters
    ----------
    model : stilt.Model
        A PYSTILT Model; its simulations are selected with
        ``model.simulations.sel(...)`` and their footprints (or trajectories)
        loaded by path.
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
        variant: str,
        **kwargs,
    ) -> MatrixBlock | dict[str, MatrixBlock]:
        """
        Build the Jacobian H from output-grid coordinates and flux time bins.

        Convenience wrapper over :meth:`build_from_target` for callers that have
        a plain list of ``(x, y)`` cell centers (a regular grid is assumed;
        PYSTILT infers the cell size from the coordinate spacing). Prefer
        :meth:`build_from_target` with a ``stilt.Grid`` or ``stilt.Mesh``,
        which carry resolution, CRS, and the state index explicitly.

        Parameters
        ----------
        coords : list[tuple[float, float]] | dict[str, list[tuple[float, float]]]
            Output grid cell centers as (x, y) tuples. Pass a dict to build
            multiple Jacobians over different coordinate sets.
        flux_times : pd.IntervalIndex
            Time bins for the fluxes.
        variant : str
            PYSTILT variant whose footprints make the rows.
        **kwargs
            Forwarded to :meth:`build_from_target` (``footprint``,
            ``time_range``, ``location_ids``, ``subset_hours``,
            ``num_processes``, ``location_mapper``, ``timeout``, ``threshold``,
            ``sparse``).

        Returns
        -------
        MatrixBlock | dict[str, MatrixBlock]
            Single MatrixBlock when coords is a list; dict when coords is a dict.
        """
        return self.build_from_target(coords, flux_times, variant, **kwargs)

    def build_from_grid(
        self,
        grid: Target | Mapping[str, Target],
        flux_times: pd.IntervalIndex,
        variant: str,
        **kwargs,
    ) -> MatrixBlock | dict[str, MatrixBlock]:
        """Alias of :meth:`build_from_target` kept for existing callers."""
        return self.build_from_target(grid, flux_times, variant, **kwargs)

    def build_from_target(
        self,
        target: Target | Mapping[str, Target],
        flux_times: pd.IntervalIndex,
        variant: str,
        *,
        footprint: FootprintConfig | None = None,
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
        Build the Jacobian matrix H over a spatial target and flux time bins.

        Each footprint is conservatively regridded onto ``target`` (see
        :meth:`stilt.Footprint.aggregate`) and its time-binned sensitivities
        become one Jacobian row.  The Jacobian's column index is the target's
        state index (``(lon, lat, time)`` for grids, ``(cell, time)`` for
        ``stilt.Mesh`` / ``stilt.Zones``).

        Parameters
        ----------
        target : stilt.Grid | stilt.Mesh | xr.DataArray | xr.Dataset | list | dict
            The state geometry: a ``stilt.Grid`` (every cell), ``stilt.Mesh``
            (polygons: shapefile, H3, point windows), ``stilt.Zones``
            (super-cells), a CF xarray grid
            (``lon``/``lat`` or ``x``/``y`` coordinates; ``NaN`` cells in a 2-D
            DataArray are masked out), or a plain list of ``(x, y)`` cell
            centers. Pass a dict to build multiple Jacobians over different
            targets.
        flux_times : pd.IntervalIndex
            Time bins for the fluxes.
        variant : str
            PYSTILT variant whose simulations make the rows (``"hrrr"``). A
            realization group's name selects every realization.
        footprint : stilt.FootprintConfig, optional
            Regenerate each footprint from the variant's stored trajectory
            particles with these settings instead of loading the stored
            footprint (full kernel fidelity when the state grid differs from
            the stored raster; slower).
        time_range : tuple or None
            ``(start, end)`` to filter simulations by receptor time, both
            inclusive. Defaults to the full flux window from ``flux_times``.
        location_ids : set[str] or None
            Restrict to specific location IDs.
        subset_hours : int | list[int] | None
            Filter simulations to specific hours of the day (receptor time,
            UTC).
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

        targets: dict[str, Target]
        if isinstance(target, dict):
            targets = dict(target)
        else:
            targets = {"DEFAULT": cast("Target", target)}

        if time_range is None:
            time_range = (flux_times[0].left, flux_times[-1].right)

        hours = None
        if subset_hours is not None:
            hours = (
                {subset_hours} if isinstance(subset_hours, int) else set(subset_hours)
            )
        sims = self.model.simulations.sel(
            variant=variant,
            time=slice(*time_range),
            location=location_ids,
            where=None if hours is None else (lambda r: r.time.hour in hours),
        )

        # Get paths without loading — footprints are loaded (or regenerated
        # from trajectories) inside each worker to avoid serial NFS I/O and
        # large object pickling overhead.
        regenerate = footprint is not None
        outputs = sims.trajectories if regenerate else sims.footprint
        paths = list(outputs.paths().values())

        if not paths:
            what = (
                f"No trajectories found for variant '{variant}'"
                if regenerate
                else f"No footprints found for variant '{variant}'"
            )
            raise ValueError(
                f"{what} after filtering. "
                "Check that outputs exist and filters are not too restrictive."
            )

        logger.debug(
            "Dispatching %d %s...",
            len(paths),
            "trajectories" if regenerate else "footprints",
        )
        results = Parallel(n_jobs=num_processes, timeout=timeout)(
            delayed(_build_jacobian_row_from_path)(
                path=path,
                targets=targets,
                location_dim=self.location_dim,
                time_dim=self.time_dim,
                flux_times=flux_times,
                footprint_config=footprint,
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
    footprint_config: FootprintConfig | None = None,
) -> dict[str, pd.DataFrame] | None:
    """
    Load one footprint from disk (or regenerate it) and build its Jacobian row.

    With ``footprint_config`` the path is a trajectory parquet and the
    footprint is recalculated from its particles on that config's grid.
    Loading inside the worker avoids serial NFS reads and large object
    pickling that would occur if outputs were pre-loaded before dispatch.
    """
    try:
        if footprint_config is not None:
            fp = Trajectories.from_parquet(path).footprint(footprint_config)
        else:
            fp = Footprint.from_netcdf(path)
    except Exception:
        return None

    return _build_jacobian_row(fp, targets, location_dim, time_dim, flux_times)

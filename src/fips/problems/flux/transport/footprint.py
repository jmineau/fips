"""STILT footprint loading and Jacobian row building utilities."""

import datetime as dt

import numpy as np
import pandas as pd
import stilt

from fips.spacetime import integrate_over_time_bins


def load_footprint(
    sim: stilt.Simulation, resolution: str | None = None
) -> stilt.Footprint | None:
    """Load footprint from a STILT simulation at specified resolution.
    
    Parameters
    ----------
    sim : stilt.Simulation
        STILT simulation object.
    resolution : str | None, optional
        Resolution string (e.g., '0.01deg'). If None, uses highest resolution.
        
    Returns
    -------
    stilt.Footprint | None
        Footprint object or None if not available.
    """
    # Load footprint at specified resolution if available
    # Otherwise, get the default (highest) resolution footprint
    footprint = sim.footprints[resolution] if resolution else sim.footprint
    return footprint


def check_footprint_in_time_range(
    footprint: stilt.Footprint, t_start: dt.datetime, t_stop: dt.datetime
) -> bool:
    """Check if footprint time range overlaps with inversion time range.
    
    Parameters
    ----------
    footprint : stilt.Footprint
        STILT footprint object.
    t_start : dt.datetime
        Start time of inversion period.
    t_stop : dt.datetime
        End time of inversion period.
        
    Returns
    -------
    bool
        True if footprint overlaps with time range.
    """
    return footprint.time_range[0] < t_stop and footprint.time_range[1] > t_start


def get_footprint(
    sim: stilt.Simulation,
    t_start: dt.datetime,
    t_stop: dt.datetime,
    resolution: str | None = None,
) -> stilt.Footprint | None:
    """Load and validate a footprint from a STILT simulation.
    
    Parameters
    ----------
    sim : stilt.Simulation
        STILT simulation object.
    t_start : dt.datetime
        Start time of inversion period.
    t_stop : dt.datetime
        End time of inversion period.
    resolution : str | None, optional
        Resolution string. If None, uses highest resolution.
        
    Returns
    -------
    stilt.Footprint | None
        Footprint object if valid and in time range, None otherwise.
    """
    # Load footprint object
    footprint = load_footprint(sim=sim, resolution=resolution)

    if not footprint:
        return None

    # Check if footprint time range overlaps with inversion time range
    if not check_footprint_in_time_range(
        footprint=footprint, t_start=t_start, t_stop=t_stop
    ):
        # A footprint within a simulation might not overlap with the inversion time range
        # even if the simulation does
        return None

    return footprint


def build_obs_index(sim: stilt.Simulation) -> pd.MultiIndex:
    """Build observation index for a STILT simulation.
    
    Parameters
    ----------
    sim : stilt.Simulation
        STILT simulation object.
        
    Returns
    -------
    pd.MultiIndex
        MultiIndex with levels (obs_location, obs_time).
    """
    return pd.MultiIndex.from_arrays(
        [[sim.receptor.location.id], [sim.receptor.time]],
        names=["obs_location", "obs_time"],
    )


def calc_rounding_digits(resolution: float) -> int:
    """Calculate number of decimal places needed for coordinate rounding.
    
    Parameters
    ----------
    resolution : float
        Grid resolution in degrees or meters.
        
    Returns
    -------
    int
        Number of decimal places for rounding.
        
    Raises
    ------
    ValueError
        If resolution is not positive.
    """
    if resolution <= 0:
        raise ValueError("Resolution must be positive")
    if resolution < 1:  # fractional resolution
        digits = int(np.ceil(np.abs(np.log10(resolution)))) + 1
    else:
        digits = int(-np.log10(resolution))
    return digits


def build_jacobian_row(
    simulation: stilt.Simulation | None,
    coords: dict[str, list[tuple[float, float]]],
    flux_times: pd.IntervalIndex,
    resolution: str | None = None,
    subset_hours: int | list[int] | None = None,
) -> dict[str, pd.DataFrame] | str | None:
    """Build Jacobian row(s) from a single STILT simulation.
    
    This function processes a STILT simulation to build one or more rows of the
    Jacobian matrix (sensitivity matrix). It filters the footprint to specified
    coordinates, integrates over time bins, and returns transposed DataFrames
    ready to be assembled into the full Jacobian.
    
    Parameters
    ----------
    simulation : stilt.Simulation | None
        STILT simulation object (already validated) or None if filtered out.
    coords : dict[str, list[tuple[float, float]]]
        Dictionary mapping domain names to lists of (x, y) coordinate tuples.
        Coordinates should match the CRS of STILT footprints.
    flux_times : pd.IntervalIndex
        Time bins for flux integration.
    resolution : str | None, optional
        Footprint resolution to use. If None, uses highest available.
    subset_hours : int | list[int] | None, optional
        Hour filter (not used here, kept for signature compatibility).
        
    Returns
    -------
    dict[str, pd.DataFrame] | str | None
        - dict: Maps domain names to Jacobian row DataFrames
        - str: Simulation ID if simulation failed
        - None: If simulation was filtered out or had no valid footprint
    """
    if simulation is None:
        return None
        
    sim = simulation
    t0 = dt.datetime.now()
    t_start, t_stop = flux_times[0].left, flux_times[-1].right

    # Get footprint for the simulation
    try:
        footprint = get_footprint(
            sim=sim, t_start=t_start, t_stop=t_stop, resolution=resolution
        )
    except Exception as e:
        foot_file = sim.paths["footprints"][str(resolution)]
        print(f"Error loading footprint file {foot_file}: {e}")
        raise e

    if footprint is None:
        return None

    # Convert xarray to pandas for sparse representation
    foot = footprint.data.to_series()

    # Get the x and y dimension names
    is_latlon = "lon" in foot.index.names and "lat" in foot.index.names
    x_dim = "lon" if is_latlon else "x"
    y_dim = "lat" if is_latlon else "y"

    foot = foot.reset_index()

    # Round coordinates to avoid floating point issues
    xres, yres = footprint.xres, footprint.yres
    xdigits = calc_rounding_digits(xres)
    ydigits = calc_rounding_digits(yres)

    foot[x_dim] = foot[x_dim].round(xdigits)
    foot[y_dim] = foot[y_dim].round(ydigits)

    # Reorder dimensions to x, y, time
    foot = foot.set_index([x_dim, y_dim, "time"])  # still a df

    # Build index value for the observation
    obs_index = build_obs_index(sim=sim)

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
    
    print(f"Finished computing Jacobian row for {sim.id} in {dt.datetime.now() - t0}")
    return rows


__all__ = [
    "load_footprint",
    "check_footprint_in_time_range",
    "get_footprint",
    "build_obs_index",
    "calc_rounding_digits",
    "build_jacobian_row",
]

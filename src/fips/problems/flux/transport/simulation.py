"""STILT simulation loading and validation utilities.

This module provides functions for loading, filtering, and validating
STILT simulations by time and location.
"""

import datetime as dt
from pathlib import Path

import pandas as pd
import stilt


def load_simulation(simulation: stilt.Simulation | Path) -> stilt.Simulation:
    """Load a STILT simulation from a path or return if already loaded.
    
    Parameters
    ----------
    simulation : stilt.Simulation | Path
        Either a STILT Simulation object or path to simulation directory.
        
    Returns
    -------
    stilt.Simulation
        Loaded simulation object.
        
    Raises
    ------
    ValueError
        If simulation is neither a Path nor stilt.Simulation object.
    """
    if isinstance(simulation, Path):
        sim = stilt.Simulation.from_path(simulation)
    elif isinstance(simulation, stilt.Simulation):
        sim = simulation
    else:
        raise ValueError("simulation must be a Path or a stilt.Simulation object")
    return sim


def check_sim_in_time_range(
    sim: stilt.Simulation,
    t_start: dt.datetime,
    t_stop: dt.datetime,
    subset_hours: int | list[int] | None = None,
) -> bool:
    """Check if a STILT simulation overlaps with the inversion time range.
    
    Parameters
    ----------
    sim : stilt.Simulation
        STILT simulation to check.
    t_start : dt.datetime
        Start time of inversion period.
    t_stop : dt.datetime
        End time of inversion period.
    subset_hours : int | list[int] | None, optional
        If provided, only include simulations at these hours of day.
        
    Returns
    -------
    bool
        True if simulation overlaps with time range and passes hour filter.
        
    Raises
    ------
    ValueError
        If STILT was run forward in time (n_hours >= 0).
    """
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


def get_simulation(
    simulation: stilt.Simulation | Path,
    t_start: dt.datetime,
    t_stop: dt.datetime,
    subset_hours: int | list[int] | None = None,
) -> stilt.Simulation | str | None:
    """Load and validate a STILT simulation.
    
    Parameters
    ----------
    simulation : stilt.Simulation | Path
        STILT simulation or path to simulation directory.
    t_start : dt.datetime
        Start time of inversion period.
    t_stop : dt.datetime
        End time of inversion period.
    subset_hours : int | list[int] | None, optional
        If provided, only include simulations at these hours of day.
        
    Returns
    -------
    stilt.Simulation | str | None
        - stilt.Simulation if valid and in time range
        - str (simulation ID) if simulation failed
        - None if simulation is outside time range
    """
    sim = load_simulation(simulation)

    if sim.status != "SUCCESS":
        return sim.id

    if not check_sim_in_time_range(
        sim=sim, t_start=t_start, t_stop=t_stop, subset_hours=subset_hours
    ):
        return None
    return sim


__all__ = ["load_simulation", "check_sim_in_time_range", "get_simulation"]
